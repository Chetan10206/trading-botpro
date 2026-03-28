"""
main.py
─────────────────────────────────────────────────────────────────────────────
Unified CLI entry point for the NY Open FVG Backtest suite.

Commands
  backtest   Run a single backtest and print results
  optimise   Grid-search over R:R × retest-tolerance parameters
  dashboard  Launch the interactive Dash web dashboard
  fetch      Download and cache 1-min data (no strategy run)

Examples
  python main.py backtest --symbol SPY  --start 2024-11-01 --end 2024-11-30
  python main.py backtest --symbol QQQ  --start 2024-01-01 --end 2024-06-30 --rr 2.5
  python main.py optimise --symbol SPY  --start 2024-01-01 --end 2024-06-30
  python main.py dashboard
  python main.py fetch    --symbol AAPL --start 2024-01-01 --end 2024-06-30
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Ensure project root is on PYTHONPATH ────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
    )


def cmd_backtest(args: argparse.Namespace) -> None:
    from backtest.backtest import Backtest

    bt = Backtest(
        symbol          = args.symbol,
        start           = args.start,
        end             = args.end,
        initial_capital = args.capital,
        risk_per_trade  = args.risk / 100.0,
        rr_ratio        = args.rr,
        rt_tolerance    = args.rt,
        eod_exit_hhmm   = args.eod,
        data_source     = args.source,
    )

    results = bt.run()
    bt.print_summary(results)

    if args.save:
        csv = bt.save_report(results)
        print(f"\n  Trades → {csv}")


def cmd_optimise(args: argparse.Namespace) -> None:
    from backtest.backtest import Optimiser

    rr_values = [float(x) for x in args.rr_list.split(",")]
    rt_values = [float(x) for x in args.rt_list.split(",")]

    opt = Optimiser(
        symbol          = args.symbol,
        start           = args.start,
        end             = args.end,
        initial_capital = args.capital,
        risk            = args.risk / 100.0,
    )

    df = opt.run(rr_values=rr_values, rt_values=rt_values, metric=args.metric)

    print(f"\n{'═'*70}")
    print(f"  OPTIMISER RESULTS  ·  {args.symbol}  ·  sorted by {args.metric}")
    print(f"{'═'*70}")
    print(df.to_string(index=False))
    print()

    out_path = ROOT / "results" / f"optimise_{args.symbol}_{args.start}_{args.end}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


def cmd_dashboard(_args: argparse.Namespace) -> None:
    print("\n  Starting NY Open FVG Dashboard …")
    print("  Open http://127.0.0.1:8050 in your browser\n")
    from app.app import app
    app.run(debug=True, host="0.0.0.0", port=8050)


def cmd_fetch(args: argparse.Namespace) -> None:
    from data.fetch_data import get_data

    df = get_data(
        symbol     = args.symbol,
        start      = args.start,
        end        = args.end,
        source     = args.source,
        use_cache  = True,
    )

    print(f"\n  {args.symbol}: {len(df):,} rows cached")
    print(f"  Range: {df.index[0]}  →  {df.index[-1]}")
    print(df.tail(5).to_string())


# ═══════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "main.py",
        description = "NY Open FVG Breakout Strategy — backtest suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    sub = p.add_subparsers(dest="command", required=True)

    # ── Shared date/symbol args ────────────────────────────────────────────
    def add_common(sp):
        sp.add_argument("--symbol",  default="SPY",        help="Ticker (default: SPY)")
        sp.add_argument("--start",   default="2024-11-01", help="Start date YYYY-MM-DD")
        sp.add_argument("--end",     default="2024-11-30", help="End date   YYYY-MM-DD")
        sp.add_argument("--capital", default=10_000, type=float, help="Initial capital $")
        sp.add_argument("--risk",    default=1.0,    type=float, help="Risk %% per trade (default 1)")
        sp.add_argument("--source",  default="yfinance",   choices=["yfinance","alpaca"])

    # ── backtest ───────────────────────────────────────────────────────────
    sp_bt = sub.add_parser("backtest", help="Run a single backtest")
    add_common(sp_bt)
    sp_bt.add_argument("--rr",   default=2.0,  type=float, help="R:R ratio (default 2.0)")
    sp_bt.add_argument("--rt",   default=10.0, type=float, help="Retest tolerance %% (default 10)")
    sp_bt.add_argument("--eod",  default=1555, type=int,   help="EOD exit time HHMM (default 1555)")
    sp_bt.add_argument("--save", action="store_true",      help="Save CSV + summary to results/")
    sp_bt.set_defaults(func=cmd_backtest)

    # ── optimise ───────────────────────────────────────────────────────────
    sp_opt = sub.add_parser("optimise", help="Grid-search R:R × retest-tolerance")
    add_common(sp_opt)
    sp_opt.add_argument("--rr-list",  default="1.5,2.0,2.5,3.0",
                        help="Comma-separated R:R values to test")
    sp_opt.add_argument("--rt-list",  default="5.0,10.0,15.0,20.0",
                        help="Comma-separated retest-tolerance %% values to test")
    sp_opt.add_argument("--metric",   default="sharpe_ratio",
                        help="Metric to sort optimiser output by")
    sp_opt.set_defaults(func=cmd_optimise)

    # ── dashboard ──────────────────────────────────────────────────────────
    sp_dash = sub.add_parser("dashboard", help="Launch Dash web dashboard")
    sp_dash.set_defaults(func=cmd_dashboard)

    # ── fetch ──────────────────────────────────────────────────────────────
    sp_fetch = sub.add_parser("fetch", help="Download & cache 1-min data")
    add_common(sp_fetch)
    sp_fetch.set_defaults(func=cmd_fetch)

    return p


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    setup_logging(args.verbose)

    # Dispatch to sub-command handler
    args.func(args)
