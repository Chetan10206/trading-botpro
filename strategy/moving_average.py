"""
moving_average.py  (NY Open FVG Breakout Strategy Engine)
─────────────────────────────────────────────────────────────────────────────
Implements the exact rules from the Pine Script strategy:

  1. Opening Range  → first 5-min candle  (9:30–9:34 AM NY)
  2. Breakout       → body-close outside OR on 1-min chart (≥ 9:35)
  3a. Primary Entry → FVG confirmed on the 3rd candle after breakout
  3b. Retest Entry  → no immediate FVG → wait for OR retest → new FVG
  4. Stop Loss      → breakout candle's low (long) or high (short)
  5. Take Profit    → 2 × risk (configurable R:R)
  One trade per session, direction locked by first breakout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  TRADE RECORD
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Single completed trade record."""
    date        : date
    direction   : str          # "long" | "short"
    entry_type  : str          # "primary_fvg" | "retest_fvg"
    entry_time  : pd.Timestamp
    entry_price : float
    stop_loss   : float
    take_profit : float
    exit_time   : Optional[pd.Timestamp] = None
    exit_price  : Optional[float]        = None
    exit_reason : str                    = ""   # "tp" | "sl" | "eod"
    pnl_pts     : float                  = 0.0
    pnl_pct     : float                  = 0.0
    risk_pts    : float                  = 0.0
    r_multiple  : float                  = 0.0  # realised R

    def to_dict(self) -> dict:
        return {
            "date"        : self.date,
            "direction"   : self.direction,
            "entry_type"  : self.entry_type,
            "entry_time"  : self.entry_time,
            "entry_price" : round(self.entry_price, 4),
            "stop_loss"   : round(self.stop_loss,   4),
            "take_profit" : round(self.take_profit,  4),
            "exit_time"   : self.exit_time,
            "exit_price"  : round(self.exit_price,  4) if self.exit_price else None,
            "exit_reason" : self.exit_reason,
            "pnl_pts"     : round(self.pnl_pts, 4),
            "pnl_pct"     : round(self.pnl_pct, 4),
            "risk_pts"    : round(self.risk_pts, 4),
            "r_multiple"  : round(self.r_multiple, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  DAILY STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _DayState:
    """Mutable state that resets each trading day."""
    or_high     : float = np.nan
    or_low      : float = np.nan
    or_set      : bool  = False
    or_hi_run   : float = np.nan   # running max during 9:30–9:34
    or_lo_run   : float = np.nan   # running min during 9:30–9:34

    # Breakout
    bull_break  : bool  = False
    bear_break  : bool  = False
    bk_high     : float = np.nan   # stop-loss reference (short)
    bk_low      : float = np.nan   # stop-loss reference (long)
    bk_bar_idx  : int   = -1       # positional index of breakout candle

    # Phase flags
    fvg_wait    : bool  = False    # watching for 3rd candle (primary)
    rt_wait     : bool  = False    # no FVG → watching for retest
    rt_ready    : bool  = False    # retest touched
    traded      : bool  = False    # one trade per session

    def reset(self):
        for f in self.__dataclass_fields__:
            setattr(self, f, self.__dataclass_fields__[f].default)


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class NYOpenFVGStrategy:
    """
    Vectorised-style row-by-row simulator of the NY Open FVG Breakout.

    Parameters
    ----------
    rr_ratio      : float  – Take-Profit = risk × rr_ratio  (default 2.0)
    rt_tolerance  : float  – Retest band as % of OR height   (default 10 %)
    eod_exit_hhmm : int    – Force-close any open trade at this NY time
                             (default 1555 = 3:55 PM)
    """

    def __init__(
        self,
        rr_ratio      : float = 2.0,
        rt_tolerance  : float = 10.0,
        eod_exit_hhmm : int   = 1555,
    ):
        self.rr          = rr_ratio
        self.rt_tol_pct  = rt_tolerance / 100.0
        self.eod_hhmm    = eod_exit_hhmm

    # ── Main entry point ──────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> list[Trade]:
        """
        Iterate over every 1-minute bar and emit Trade objects.

        Parameters
        ----------
        df : DataFrame with columns [open, high, low, close, volume,
                                     ny_hhmm, date]
               as produced by DataFetcher.
        """
        trades: list[Trade] = []
        state  = _DayState()
        bars   = df.reset_index(drop=False)   # positional indexing

        active_trade: Optional[Trade] = None
        prev_date: Optional[date]     = None

        for i, row in bars.iterrows():
            d     = row["date"]
            hhmm  = row["ny_hhmm"]

            # ── Day reset ─────────────────────────────────────────────────
            if d != prev_date:
                # Close any open trade at EOD of previous day
                if active_trade is not None:
                    active_trade = self._close_trade(
                        active_trade, bars.iloc[i - 1], "eod"
                    )
                    trades.append(active_trade)
                    active_trade = None

                state.reset()
                prev_date = d

            # ── Manage open position (check SL / TP hit intra-bar) ────────
            if active_trade is not None:
                result = self._check_exit(active_trade, row, hhmm)
                if result:
                    trades.append(result)
                    active_trade = None
                continue   # do not look for new entry while in a trade

            # ── PHASE 0 : Build Opening Range (9:30–9:34) ─────────────────
            if 930 <= hhmm < 935:
                if np.isnan(state.or_hi_run):
                    state.or_hi_run = row["high"]
                    state.or_lo_run = row["low"]
                else:
                    state.or_hi_run = max(state.or_hi_run, row["high"])
                    state.or_lo_run = min(state.or_lo_run, row["low"])
                continue

            # ── Lock range on first bar ≥ 9:35 ────────────────────────────
            if not state.or_set:
                if np.isnan(state.or_hi_run):
                    continue   # market not open yet (e.g. holiday stub)
                state.or_high = state.or_hi_run
                state.or_low  = state.or_lo_run
                state.or_set  = True
                logger.debug("%s OR: %.4f / %.4f", d, state.or_high, state.or_low)

            # ── PHASE 1 : Detect breakout candle ─────────────────────────
            if state.or_set and not state.bull_break and not state.bear_break:
                if row["close"] > state.or_high:               # bullish body-close
                    state.bull_break = True
                    state.bk_high    = row["high"]
                    state.bk_low     = row["low"]
                    state.bk_bar_idx = i
                    state.fvg_wait   = True
                    logger.debug("%s BULL breakout @ %.4f", d, row["close"])

                elif row["close"] < state.or_low:              # bearish body-close
                    state.bear_break = True
                    state.bk_high    = row["high"]
                    state.bk_low     = row["low"]
                    state.bk_bar_idx = i
                    state.fvg_wait   = True
                    logger.debug("%s BEAR breakout @ %.4f", d, row["close"])

            # ── PHASE 2 : Check FVG on 3rd candle (bkBar + 2) ─────────────
            if state.fvg_wait and i == state.bk_bar_idx + 2:
                state.fvg_wait = False

                c1_high = bars.iloc[state.bk_bar_idx]["high"]
                c1_low  = bars.iloc[state.bk_bar_idx]["low"]
                c3_high = row["high"]
                c3_low  = row["low"]

                if state.bull_break and c3_low > c1_high:
                    # ✅ Bullish FVG confirmed → primary long entry
                    active_trade = self._open_trade(
                        row, d, "long", "primary_fvg", state
                    )
                    state.traded = True
                    logger.debug(
                        "%s PRIMARY LONG entry: %.4f  SL: %.4f  TP: %.4f",
                        d, active_trade.entry_price,
                        active_trade.stop_loss, active_trade.take_profit
                    )

                elif state.bear_break and c3_high < c1_low:
                    # ✅ Bearish FVG confirmed → primary short entry
                    active_trade = self._open_trade(
                        row, d, "short", "primary_fvg", state
                    )
                    state.traded = True
                    logger.debug(
                        "%s PRIMARY SHORT entry: %.4f  SL: %.4f  TP: %.4f",
                        d, active_trade.entry_price,
                        active_trade.stop_loss, active_trade.take_profit
                    )

                else:
                    state.rt_wait = True   # no FVG → watch for retest

            # ── PHASE 3 : Retest detection ────────────────────────────────
            if state.rt_wait and not state.rt_ready and not state.traded:
                tol = (state.or_high - state.or_low) * self.rt_tol_pct
                if state.bull_break and row["low"] <= state.or_high + tol:
                    state.rt_ready = True
                    logger.debug("%s RETEST of OR high detected", d)
                elif state.bear_break and row["high"] >= state.or_low - tol:
                    state.rt_ready = True
                    logger.debug("%s RETEST of OR low detected", d)

            # ── PHASE 4 : Retest FVG entry ────────────────────────────────
            if state.rt_wait and state.rt_ready and not state.traded and i >= 2:
                c1_high_rt = bars.iloc[i - 2]["high"]
                c1_low_rt  = bars.iloc[i - 2]["low"]
                c3_high_rt = row["high"]
                c3_low_rt  = row["low"]

                if state.bull_break:
                    # Candle-1 of new FVG must be near OR level
                    near_or = c1_high_rt >= state.or_high * 0.9985
                    fvg_ok  = c3_low_rt > c1_high_rt
                    if fvg_ok and near_or:
                        active_trade = self._open_trade(
                            row, d, "long", "retest_fvg", state
                        )
                        state.traded = True
                        logger.debug(
                            "%s RETEST LONG entry: %.4f  SL: %.4f  TP: %.4f",
                            d, active_trade.entry_price,
                            active_trade.stop_loss, active_trade.take_profit
                        )

                elif state.bear_break:
                    near_or = c1_low_rt <= state.or_low * 1.0015
                    fvg_ok  = c3_high_rt < c1_low_rt
                    if fvg_ok and near_or:
                        active_trade = self._open_trade(
                            row, d, "short", "retest_fvg", state
                        )
                        state.traded = True
                        logger.debug(
                            "%s RETEST SHORT entry: %.4f  SL: %.4f  TP: %.4f",
                            d, active_trade.entry_price,
                            active_trade.stop_loss, active_trade.take_profit
                        )

        # Close any trade still open at the very end of the dataset
        if active_trade is not None:
            active_trade = self._close_trade(active_trade, bars.iloc[-1], "eod")
            trades.append(active_trade)

        return trades

    # ── Trade helpers ──────────────────────────────────────────────────────

    def _open_trade(
        self,
        row   : pd.Series,
        d     : date,
        direction : str,
        entry_type: str,
        state : _DayState,
    ) -> Trade:
        ep = row["close"]

        if direction == "long":
            sl = state.bk_low               # breakout candle LOW
            tp = ep + self.rr * (ep - sl)
        else:
            sl = state.bk_high              # breakout candle HIGH
            tp = ep - self.rr * (sl - ep)

        risk = abs(ep - sl)

        return Trade(
            date        = d,
            direction   = direction,
            entry_type  = entry_type,
            entry_time  = row.get("index", row.name),
            entry_price = ep,
            stop_loss   = sl,
            take_profit  = tp,
            risk_pts    = risk,
        )

    def _check_exit(
        self,
        trade : Trade,
        row   : pd.Series,
        hhmm  : int,
    ) -> Optional[Trade]:
        """
        Check whether the current bar hits SL, TP, or EOD.
        Uses intra-bar high/low to detect which level was hit first.
        Returns the completed Trade (or None if still open).
        """
        bar_high = row["high"]
        bar_low  = row["low"]
        ts       = row.get("index", row.name)

        if trade.direction == "long":
            sl_hit = bar_low  <= trade.stop_loss
            tp_hit = bar_high >= trade.take_profit
        else:
            sl_hit = bar_high >= trade.stop_loss
            tp_hit = bar_low  <= trade.take_profit

        # Conservative tie-break: assume SL before TP if both in same bar
        if sl_hit and tp_hit:
            sl_hit = True
            tp_hit = False

        if sl_hit:
            return self._close_trade(trade, row, "sl")
        if tp_hit:
            return self._close_trade(trade, row, "tp")
        if hhmm >= self.eod_hhmm:
            return self._close_trade(trade, row, "eod")

        return None

    @staticmethod
    def _close_trade(trade: Trade, row: pd.Series, reason: str) -> Trade:
        if reason == "tp":
            exit_px = trade.take_profit
        elif reason == "sl":
            exit_px = trade.stop_loss
        else:                                            # eod
            exit_px = row["close"]

        trade.exit_time  = row.get("index", row.name)
        trade.exit_price = exit_px
        trade.exit_reason = reason

        if trade.direction == "long":
            trade.pnl_pts = exit_px - trade.entry_price
        else:
            trade.pnl_pts = trade.entry_price - exit_px

        trade.pnl_pct   = trade.pnl_pts / trade.entry_price * 100
        trade.r_multiple = trade.pnl_pts / trade.risk_pts if trade.risk_pts else 0.0

        return trade


# ── Build trades DataFrame ─────────────────────────────────────────────────

def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.to_dict() for t in trades])
