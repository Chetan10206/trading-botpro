"""
backtest.py
─────────────────────────────────────────────────────────────────────────────
Full backtesting engine for the NY Open FVG Breakout Strategy.

Responsibilities
  • Run the strategy over historical 1-min data
  • Simulate realistic position sizing (fixed-risk % of equity)
  • Calculate a comprehensive suite of performance metrics
  • Produce equity curve, drawdown series, and per-trade log
  • Save results to CSV + HTML report

Usage
  from backtest.backtest import Backtest
  bt = Backtest(symbol="SPY", start="2024-01-01", end="2024-06-30")
  results = bt.run()
  bt.print_summary(results)
  bt.save_report(results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data.fetch_data    import DataFetcher
from strategy.moving_average import NYOpenFVGStrategy, trades_to_df

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceMetrics:
    # ── Totals ──────────────────────────────────────────────────────────────
    total_trades      : int   = 0
    winning_trades    : int   = 0
    losing_trades     : int   = 0
    breakeven_trades  : int   = 0

    # ── Rates ───────────────────────────────────────────────────────────────
    win_rate          : float = 0.0   # %
    loss_rate         : float = 0.0

    # ── P&L ─────────────────────────────────────────────────────────────────
    total_pnl_pct     : float = 0.0   # sum of % returns
    avg_win_pct       : float = 0.0
    avg_loss_pct      : float = 0.0
    largest_win_pct   : float = 0.0
    largest_loss_pct  : float = 0.0
    profit_factor     : float = 0.0   # gross profit / gross loss
    expectancy_pct    : float = 0.0   # avg pnl per trade

    # ── R-multiples ─────────────────────────────────────────────────────────
    avg_r             : float = 0.0
    total_r           : float = 0.0

    # ── Equity curve ────────────────────────────────────────────────────────
    initial_capital   : float = 10_000.0
    final_equity      : float = 10_000.0
    total_return_pct  : float = 0.0
    cagr_pct          : float = 0.0

    # ── Risk metrics ────────────────────────────────────────────────────────
    max_drawdown_pct  : float = 0.0
    max_drawdown_dur  : int   = 0   # bars
    sharpe_ratio      : float = 0.0
    sortino_ratio     : float = 0.0
    calmar_ratio      : float = 0.0

    # ── Entry breakdown ─────────────────────────────────────────────────────
    primary_fvg_count : int   = 0
    retest_fvg_count  : int   = 0
    tp_exits          : int   = 0
    sl_exits          : int   = 0
    eod_exits         : int   = 0

    # ── Streaks ─────────────────────────────────────────────────────────────
    max_consec_wins   : int   = 0
    max_consec_losses : int   = 0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  EQUITY SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def _simulate_equity(
    trades_df       : pd.DataFrame,
    initial_capital : float,
    risk_per_trade  : float,   # fraction of equity risked per trade (e.g. 0.01 = 1%)
) -> pd.DataFrame:
    """
    Apply fixed-fractional position sizing:
      shares = (equity × risk%) / (entry − stop)
      pnl    = shares × (exit − entry)

    Returns trades_df with added columns:
      equity_before, shares, dollar_pnl, equity_after
    """
    df     = trades_df.copy().reset_index(drop=True)
    equity = initial_capital
    rows   = []

    for _, r in df.iterrows():
        risk_pts  = abs(r["entry_price"] - r["stop_loss"])
        if risk_pts == 0:
            shares = 0.0
        else:
            risk_dollars = equity * risk_per_trade
            shares       = risk_dollars / risk_pts

        if r["direction"] == "long":
            dollar_pnl = shares * (r["exit_price"] - r["entry_price"])
        else:
            dollar_pnl = shares * (r["entry_price"] - r["exit_price"])

        equity_before = equity
        equity        = max(0.01, equity + dollar_pnl)

        rows.append({
            **r.to_dict(),
            "equity_before": round(equity_before, 2),
            "shares"        : round(shares, 4),
            "dollar_pnl"    : round(dollar_pnl, 2),
            "equity_after"  : round(equity, 2),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _compute_metrics(
    trades_df       : pd.DataFrame,
    initial_capital : float,
    start_date      : str,
    end_date        : str,
) -> PerformanceMetrics:
    m  = PerformanceMetrics(initial_capital=initial_capital)

    if trades_df.empty:
        return m

    m.total_trades   = len(trades_df)
    m.initial_capital = initial_capital
    m.final_equity   = trades_df["equity_after"].iloc[-1]

    wins = trades_df[trades_df["pnl_pct"] > 0]
    loss = trades_df[trades_df["pnl_pct"] < 0]
    brkn = trades_df[trades_df["pnl_pct"] == 0]

    m.winning_trades   = len(wins)
    m.losing_trades    = len(loss)
    m.breakeven_trades = len(brkn)
    m.win_rate         = m.winning_trades / m.total_trades * 100
    m.loss_rate        = m.losing_trades  / m.total_trades * 100

    m.total_pnl_pct   = trades_df["pnl_pct"].sum()
    m.avg_win_pct     = wins["pnl_pct"].mean()  if len(wins) else 0.0
    m.avg_loss_pct    = loss["pnl_pct"].mean()  if len(loss) else 0.0
    m.largest_win_pct = wins["pnl_pct"].max()   if len(wins) else 0.0
    m.largest_loss_pct= loss["pnl_pct"].min()   if len(loss) else 0.0
    m.expectancy_pct  = trades_df["pnl_pct"].mean()

    gross_profit = wins["dollar_pnl"].sum() if len(wins) else 0.0
    gross_loss   = abs(loss["dollar_pnl"].sum()) if len(loss) else 0.0
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    m.avg_r   = trades_df["r_multiple"].mean()
    m.total_r = trades_df["r_multiple"].sum()

    # ── Equity / drawdown ──────────────────────────────────────────────────
    equity_curve   = np.array([initial_capital] + list(trades_df["equity_after"]))
    running_max    = np.maximum.accumulate(equity_curve)
    drawdown       = (equity_curve - running_max) / running_max * 100
    m.max_drawdown_pct = abs(drawdown.min())

    # Drawdown duration (consecutive bars below previous peak)
    in_dd  = drawdown < 0
    dd_dur = 0
    max_dur = 0
    for flag in in_dd:
        dd_dur = dd_dur + 1 if flag else 0
        max_dur = max(max_dur, dd_dur)
    m.max_drawdown_dur = max_dur

    # ── Return metrics ─────────────────────────────────────────────────────
    m.total_return_pct = (m.final_equity - initial_capital) / initial_capital * 100

    try:
        n_years = (
            pd.Timestamp(end_date) - pd.Timestamp(start_date)
        ).days / 365.25
        if n_years > 0:
            m.cagr_pct = ((m.final_equity / initial_capital) ** (1 / n_years) - 1) * 100
    except Exception:
        m.cagr_pct = 0.0

    # ── Sharpe / Sortino (trade-level returns) ─────────────────────────────
    daily_ret = trades_df.groupby("date")["pnl_pct"].sum()

    if len(daily_ret) > 1:
        mean_r  = daily_ret.mean()
        std_r   = daily_ret.std()
        neg_std = daily_ret[daily_ret < 0].std()

        m.sharpe_ratio  = (mean_r / std_r  * np.sqrt(252)) if std_r  > 0 else 0.0
        m.sortino_ratio = (mean_r / neg_std * np.sqrt(252)) if neg_std > 0 else 0.0

    m.calmar_ratio = (
        m.cagr_pct / m.max_drawdown_pct if m.max_drawdown_pct > 0 else 0.0
    )

    # ── Entry / exit breakdown ─────────────────────────────────────────────
    m.primary_fvg_count = (trades_df["entry_type"] == "primary_fvg").sum()
    m.retest_fvg_count  = (trades_df["entry_type"] == "retest_fvg").sum()
    m.tp_exits          = (trades_df["exit_reason"] == "tp").sum()
    m.sl_exits          = (trades_df["exit_reason"] == "sl").sum()
    m.eod_exits         = (trades_df["exit_reason"] == "eod").sum()

    # ── Consecutive win / loss streaks ─────────────────────────────────────
    outcomes = (trades_df["pnl_pct"] > 0).tolist()
    win_streak = loss_streak = cur_win = cur_loss = 0
    for w in outcomes:
        if w:
            cur_win += 1; cur_loss = 0
        else:
            cur_loss += 1; cur_win = 0
        win_streak  = max(win_streak,  cur_win)
        loss_streak = max(loss_streak, cur_loss)
    m.max_consec_wins   = win_streak
    m.max_consec_losses = loss_streak

    return m


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST CLASS
# ═══════════════════════════════════════════════════════════════════════════

class Backtest:
    """
    Full backtesting orchestrator.

    Parameters
    ----------
    symbol          : str    – Ticker symbol (e.g. "SPY", "QQQ", "AAPL")
    start           : str    – Start date  "YYYY-MM-DD"
    end             : str    – End date    "YYYY-MM-DD"
    initial_capital : float  – Starting account balance ($)
    risk_per_trade  : float  – Fraction of equity risked per trade (0.01 = 1%)
    rr_ratio        : float  – Take-Profit multiplier  (default 2.0)
    rt_tolerance    : float  – Retest tolerance %       (default 10%)
    eod_exit_hhmm   : int    – Force-exit time          (default 1555)
    data_source     : str    – "yfinance" | "alpaca"
    """

    def __init__(
        self,
        symbol          : str   = "SPY",
        start           : str   = "2024-01-01",
        end             : str   = "2024-06-30",
        initial_capital : float = 10_000.0,
        risk_per_trade  : float = 0.01,
        rr_ratio        : float = 2.0,
        rt_tolerance    : float = 10.0,
        eod_exit_hhmm   : int   = 1555,
        data_source     : str   = "yfinance",
    ):
        self.symbol          = symbol
        self.start           = start
        self.end             = end
        self.initial_capital = initial_capital
        self.risk_per_trade  = risk_per_trade
        self.rr_ratio        = rr_ratio
        self.rt_tolerance    = rt_tolerance
        self.eod_exit_hhmm   = eod_exit_hhmm
        self.data_source     = data_source

        self._data     : Optional[pd.DataFrame] = None
        self._trades   : Optional[pd.DataFrame] = None
        self._metrics  : Optional[PerformanceMetrics] = None

    # ── Orchestration ──────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the full pipeline and return a results dict with keys:
          trades, equity_curve, metrics, drawdown
        """
        # 1. Fetch data
        logger.info("═══ Fetching data: %s  %s → %s", self.symbol, self.start, self.end)
        fetcher     = DataFetcher(source=self.data_source)
        self._data  = fetcher.get(self.symbol, self.start, self.end)

        # 2. Run strategy
        logger.info("═══ Running strategy …")
        strat        = NYOpenFVGStrategy(
            rr_ratio      = self.rr_ratio,
            rt_tolerance  = self.rt_tolerance,
            eod_exit_hhmm = self.eod_exit_hhmm,
        )
        raw_trades   = strat.run(self._data)
        trades_raw   = trades_to_df(raw_trades)

        if trades_raw.empty:
            logger.warning("No trades generated — check data range / symbol.")
            return {"trades": pd.DataFrame(), "metrics": PerformanceMetrics(),
                    "equity_curve": pd.Series(dtype=float), "drawdown": pd.Series(dtype=float)}

        # 3. Apply position sizing → equity curve
        logger.info("═══ Simulating equity …")
        self._trades = _simulate_equity(
            trades_raw, self.initial_capital, self.risk_per_trade
        )

        # 4. Compute metrics
        logger.info("═══ Computing metrics …")
        self._metrics = _compute_metrics(
            self._trades, self.initial_capital, self.start, self.end
        )

        # 5. Build equity / drawdown series
        eq = pd.Series(
            [self.initial_capital] + list(self._trades["equity_after"]),
            name="equity"
        )
        dd = (eq - eq.expanding().max()) / eq.expanding().max() * 100
        dd.name = "drawdown_pct"

        return {
            "trades"       : self._trades,
            "metrics"      : self._metrics,
            "equity_curve" : eq,
            "drawdown"     : dd,
            "raw_data"     : self._data,
        }

    # ── Reporting ──────────────────────────────────────────────────────────

    def print_summary(self, results: dict) -> None:
        m = results["metrics"]
        t = results["trades"]

        sep  = "═" * 62
        sep2 = "─" * 62

        print(f"\n{sep}")
        print(f"  NY OPEN FVG BACKTEST RESULTS  ─  {self.symbol}")
        print(f"  {self.start}  →  {self.end}")
        print(sep)
        print(f"  {'Total Trades':<30} {m.total_trades:>10}")
        print(f"  {'Win Rate':<30} {m.win_rate:>9.1f}%")
        print(f"  {'Winners / Losers / BE':<30} {m.winning_trades:>4} / {m.losing_trades:>4} / {m.breakeven_trades:>4}")
        print(sep2)
        print(f"  {'Profit Factor':<30} {m.profit_factor:>10.2f}")
        print(f"  {'Expectancy (per trade)':<30} {m.expectancy_pct:>9.2f}%")
        print(f"  {'Average R':<30} {m.avg_r:>10.2f}")
        print(f"  {'Total R Earned':<30} {m.total_r:>10.2f}")
        print(sep2)
        print(f"  {'Initial Capital':<30} ${self.initial_capital:>10,.2f}")
        print(f"  {'Final Equity':<30} ${m.final_equity:>10,.2f}")
        print(f"  {'Total Return':<30} {m.total_return_pct:>9.1f}%")
        print(f"  {'CAGR':<30} {m.cagr_pct:>9.1f}%")
        print(sep2)
        print(f"  {'Max Drawdown':<30} {m.max_drawdown_pct:>9.1f}%")
        print(f"  {'Sharpe Ratio':<30} {m.sharpe_ratio:>10.2f}")
        print(f"  {'Sortino Ratio':<30} {m.sortino_ratio:>10.2f}")
        print(f"  {'Calmar Ratio':<30} {m.calmar_ratio:>10.2f}")
        print(sep2)
        print(f"  {'Primary FVG entries':<30} {m.primary_fvg_count:>10}")
        print(f"  {'Retest FVG entries':<30} {m.retest_fvg_count:>10}")
        print(f"  {'TP exits / SL exits / EOD':<30} {m.tp_exits:>3} / {m.sl_exits:>3} / {m.eod_exits:>3}")
        print(f"  {'Max Consec Wins':<30} {m.max_consec_wins:>10}")
        print(f"  {'Max Consec Losses':<30} {m.max_consec_losses:>10}")
        print(sep)

    def save_report(self, results: dict, tag: str = "") -> Path:
        """Save trades CSV and summary text to results/ directory."""
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{self.symbol}_{self.start}_{self.end}_{ts}{('_'+tag) if tag else ''}"

        # ── Trades CSV ────────────────────────────────────────────────────
        csv_path = RESULTS_DIR / f"{stem}_trades.csv"
        if not results["trades"].empty:
            results["trades"].to_csv(csv_path, index=False)
            logger.info("Trades saved → %s", csv_path)

        # ── Summary text ──────────────────────────────────────────────────
        m = results["metrics"]
        txt_path = RESULTS_DIR / f"{stem}_summary.txt"
        with open(txt_path, "w") as f:
            f.write(f"NY Open FVG Backtest  |  {self.symbol}  |  {self.start} → {self.end}\n")
            f.write("=" * 60 + "\n")
            for k, v in m.to_dict().items():
                f.write(f"{k:<30}: {v}\n")
        logger.info("Summary saved → %s", txt_path)

        return csv_path


# ═══════════════════════════════════════════════════════════════════════════
#  PARAMETER OPTIMISER  (brute-force grid search)
# ═══════════════════════════════════════════════════════════════════════════

class Optimiser:
    """
    Grid-search over rr_ratio and rt_tolerance.

    Usage
    -----
    opt = Optimiser("SPY", "2024-01-01", "2024-06-30")
    best = opt.run(
        rr_values  = [1.5, 2.0, 2.5, 3.0],
        rt_values  = [5.0, 10.0, 15.0],
    )
    print(best)
    """

    def __init__(self, symbol, start, end, initial_capital=10_000, risk=0.01):
        self.symbol  = symbol
        self.start   = start
        self.end     = end
        self.capital = initial_capital
        self.risk    = risk

        logger.info("Pre-loading data for optimiser …")
        self._data = DataFetcher().get(symbol, start, end)

    def run(
        self,
        rr_values  : list[float] = [1.5, 2.0, 2.5, 3.0],
        rt_values  : list[float] = [5.0, 10.0, 15.0, 20.0],
        metric     : str = "sharpe_ratio",
    ) -> pd.DataFrame:
        rows = []

        for rr in rr_values:
            for rt in rt_values:
                strat  = NYOpenFVGStrategy(rr_ratio=rr, rt_tolerance=rt)
                raw    = strat.run(self._data)
                tdf    = trades_to_df(raw)

                if tdf.empty:
                    continue

                eq_df  = _simulate_equity(tdf, self.capital, self.risk)
                met    = _compute_metrics(eq_df, self.capital, self.start, self.end)

                rows.append({
                    "rr_ratio"       : rr,
                    "rt_tolerance"   : rt,
                    "total_trades"   : met.total_trades,
                    "win_rate"       : round(met.win_rate, 1),
                    "profit_factor"  : round(met.profit_factor, 2),
                    "total_return"   : round(met.total_return_pct, 2),
                    "max_drawdown"   : round(met.max_drawdown_pct, 2),
                    "sharpe_ratio"   : round(met.sharpe_ratio, 2),
                    "avg_r"          : round(met.avg_r, 2),
                    "calmar_ratio"   : round(met.calmar_ratio, 2),
                })

        results = pd.DataFrame(rows).sort_values(metric, ascending=False)
        logger.info("\nOptimiser results (sorted by %s):\n%s", metric, results.to_string(index=False))
        return results


# ── CLI entry ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    bt = Backtest(
        symbol          = "SPY",
        start           = "2024-11-01",
        end             = "2024-11-30",
        initial_capital = 10_000,
        risk_per_trade  = 0.01,
        rr_ratio        = 2.0,
        rt_tolerance    = 10.0,
    )

    results = bt.run()
    bt.print_summary(results)
    bt.save_report(results)
