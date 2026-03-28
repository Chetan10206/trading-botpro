"""
app.py
─────────────────────────────────────────────────────────────────────────────
Interactive Dash dashboard for the NY Open FVG Backtest.

Features
  • Run new backtests directly from the browser
  • Equity curve with drawdown overlay
  • Trade-by-trade scatter chart
  • Full performance metrics card grid
  • Trades table with sortable columns
  • Parameter optimiser heatmap

Run
  python app/app.py
  → open http://127.0.0.1:8050
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc

# ── Path fix so imports work from any cwd ────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest.backtest import Backtest, Optimiser

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  THEME / COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════

THEME = {
    "bg"      : "#0d1117",
    "surface" : "#161b22",
    "border"  : "#30363d",
    "text"    : "#e6edf3",
    "muted"   : "#8b949e",
    "green"   : "#3fb950",
    "red"     : "#f85149",
    "blue"    : "#58a6ff",
    "yellow"  : "#d29922",
    "purple"  : "#bc8cff",
}

CHART_LAYOUT = dict(
    paper_bgcolor = THEME["bg"],
    plot_bgcolor  = THEME["surface"],
    font          = dict(color=THEME["text"], family="'JetBrains Mono', monospace"),
    margin        = dict(l=60, r=20, t=40, b=40),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=THEME["border"]),
    xaxis         = dict(gridcolor=THEME["border"], zerolinecolor=THEME["border"]),
    yaxis         = dict(gridcolor=THEME["border"], zerolinecolor=THEME["border"]),
)


# ═══════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_equity_chart(equity: pd.Series, drawdown: pd.Series) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
    )

    # Equity curve
    fig.add_trace(go.Scatter(
        y=equity, mode="lines",
        line=dict(color=THEME["blue"], width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
        name="Equity",
    ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        y=drawdown, mode="lines",
        line=dict(color=THEME["red"], width=1.5),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.15)",
        name="Drawdown %",
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text="Equity Curve & Drawdown", font=dict(size=14)),
        **CHART_LAYOUT,
    )
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    return fig


def build_trades_scatter(trades: pd.DataFrame) -> go.Figure:
    if trades.empty:
        return go.Figure().update_layout(**CHART_LAYOUT)

    colors = [THEME["green"] if p > 0 else THEME["red"]
              for p in trades["pnl_pct"]]
    sizes  = [max(6, abs(r) * 10) for r in trades["r_multiple"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades["entry_time"],
        y=trades["pnl_pct"],
        mode="markers",
        marker=dict(color=colors, size=sizes, opacity=0.85,
                    line=dict(width=0.5, color=THEME["border"])),
        customdata=trades[["direction","entry_type","exit_reason","r_multiple","dollar_pnl"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "P&L: %{y:.2f}%<br>"
            "Direction: %{customdata[0]}<br>"
            "Type: %{customdata[1]}<br>"
            "Exit: %{customdata[2]}<br>"
            "R: %{customdata[3]:.2f}<br>"
            "$P&L: $%{customdata[4]:.2f}<extra></extra>"
        ),
        name="Trades",
    ))

    fig.add_hline(y=0, line_color=THEME["muted"], line_dash="dash", line_width=1)
    fig.update_layout(
        title=dict(text="Trade P&L Distribution", font=dict(size=14)),
        yaxis_title="P&L %",
        **CHART_LAYOUT,
    )
    return fig


def build_monthly_heatmap(trades: pd.DataFrame) -> go.Figure:
    if trades.empty:
        return go.Figure().update_layout(**CHART_LAYOUT)

    df = trades.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.strftime("%b %Y")
    monthly = df.groupby("month")["pnl_pct"].sum().reset_index()
    monthly["color"] = monthly["pnl_pct"].apply(
        lambda x: THEME["green"] if x >= 0 else THEME["red"]
    )

    fig = go.Figure(go.Bar(
        x=monthly["month"],
        y=monthly["pnl_pct"],
        marker_color=monthly["color"],
        name="Monthly P&L %",
    ))
    fig.add_hline(y=0, line_color=THEME["muted"], line_dash="dot")
    fig.update_layout(
        title=dict(text="Monthly P&L %", font=dict(size=14)),
        yaxis_ticksuffix="%",
        **CHART_LAYOUT,
    )
    return fig


def build_r_histogram(trades: pd.DataFrame) -> go.Figure:
    if trades.empty:
        return go.Figure().update_layout(**CHART_LAYOUT)

    wins = trades[trades["r_multiple"] > 0]["r_multiple"]
    loss = trades[trades["r_multiple"] <= 0]["r_multiple"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=wins, name="Winners",
                               marker_color=THEME["green"], opacity=0.75))
    fig.add_trace(go.Histogram(x=loss, name="Losers",
                               marker_color=THEME["red"], opacity=0.75))
    fig.update_layout(
        title=dict(text="R-Multiple Distribution", font=dict(size=14)),
        barmode="overlay",
        xaxis_title="R Multiple",
        **CHART_LAYOUT,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC CARD HELPER
# ═══════════════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, color: str = THEME["text"]) -> dbc.Col:
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.P(label, className="text-muted mb-1",
                       style={"fontSize": "0.7rem", "letterSpacing": "0.08em",
                              "textTransform": "uppercase"}),
                html.H5(value, className="mb-0 fw-bold",
                        style={"color": color, "fontSize": "1.1rem",
                               "fontFamily": "'JetBrains Mono', monospace"}),
            ])
        ], style={"background": THEME["surface"],
                  "border": f"1px solid {THEME['border']}",
                  "borderRadius": "8px"}),
        xs=6, sm=4, md=3, lg=2, className="mb-2",
    )


def build_metrics_row(m) -> dbc.Row:
    pf_color = THEME["green"] if m.profit_factor >= 1.5 else (
               THEME["yellow"] if m.profit_factor >= 1.0 else THEME["red"])
    wr_color = THEME["green"] if m.win_rate >= 50 else THEME["red"]

    return dbc.Row([
        metric_card("Total Trades",    str(m.total_trades)),
        metric_card("Win Rate",        f"{m.win_rate:.1f}%",         wr_color),
        metric_card("Profit Factor",   f"{m.profit_factor:.2f}",     pf_color),
        metric_card("Total Return",    f"{m.total_return_pct:.1f}%",
                    THEME["green"] if m.total_return_pct >= 0 else THEME["red"]),
        metric_card("Max Drawdown",    f"{m.max_drawdown_pct:.1f}%", THEME["red"]),
        metric_card("Sharpe",          f"{m.sharpe_ratio:.2f}"),
        metric_card("Sortino",         f"{m.sortino_ratio:.2f}"),
        metric_card("Calmar",          f"{m.calmar_ratio:.2f}"),
        metric_card("Avg R",           f"{m.avg_r:.2f}R"),
        metric_card("CAGR",            f"{m.cagr_pct:.1f}%"),
        metric_card("TP / SL / EOD",   f"{m.tp_exits} / {m.sl_exits} / {m.eod_exits}"),
        metric_card("Max Loss Streak", str(m.max_consec_losses),     THEME["red"]),
    ], className="g-2")


# ═══════════════════════════════════════════════════════════════════════════
#  DASH APP
# ═══════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap",
    ],
    title="NY Open FVG Strategy",
    suppress_callback_exceptions=True,
)

# ── Layout ─────────────────────────────────────────────────────────────────

SIDEBAR = dbc.Card([
    dbc.CardHeader(html.B("⚙  Backtest Config",
                          style={"fontFamily": "'JetBrains Mono', monospace"})),
    dbc.CardBody([
        dbc.Label("Symbol"),
        dbc.Input(id="inp-symbol", value="SPY", type="text", className="mb-2"),

        dbc.Label("Start Date"),
        dbc.Input(id="inp-start",  value="2024-11-01", type="text", className="mb-2"),

        dbc.Label("End Date"),
        dbc.Input(id="inp-end",    value="2024-11-30", type="text", className="mb-2"),

        dbc.Label("Initial Capital ($)"),
        dbc.Input(id="inp-capital", value=10000, type="number", className="mb-2"),

        dbc.Label("Risk per Trade (%)"),
        dbc.Input(id="inp-risk", value=1.0, type="number", step=0.1,
                  min=0.1, max=10, className="mb-2"),

        dbc.Label("R : R Ratio"),
        dbc.Input(id="inp-rr", value=2.0, type="number", step=0.5,
                  min=0.5, max=10, className="mb-2"),

        dbc.Label("Retest Tolerance (%)"),
        dbc.Input(id="inp-rt", value=10.0, type="number", step=1.0,
                  min=0, max=50, className="mb-3"),

        dbc.Button("▶  Run Backtest", id="btn-run", color="primary",
                   className="w-100 mb-2"),
        dbc.Button("⚙  Optimise", id="btn-opt", color="secondary",
                   className="w-100"),

        html.Div(id="div-status", className="mt-2 text-center",
                 style={"fontSize": "0.8rem", "color": THEME["muted"]}),
    ])
], style={"background": THEME["surface"], "border": f"1px solid {THEME['border']}",
          "borderRadius": "10px"})

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("NY OPEN  ·  FVG BREAKOUT", className="mb-0",
                    style={"fontFamily": "'JetBrains Mono', monospace",
                           "letterSpacing": "0.12em", "color": THEME["blue"]}),
            html.Small("9:30 AM EST  ·  1-Min Chart  ·  Fair Value Gap Strategy",
                       style={"color": THEME["muted"]}),
        ]), className="py-3")
    ]),

    dbc.Row([
        # Sidebar
        dbc.Col(SIDEBAR, xs=12, md=3, className="mb-3"),

        # Main content
        dbc.Col([
            # Metrics row
            html.Div(id="div-metrics", className="mb-3"),

            # Charts
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-equity", config={"displayModeBar": False}),
                        xs=12),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-trades", config={"displayModeBar": False}),
                        xs=12, md=6),
                dbc.Col(dcc.Graph(id="chart-monthly", config={"displayModeBar": False}),
                        xs=12, md=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-r-hist", config={"displayModeBar": False}),
                        xs=12, md=6),
                dbc.Col(html.Div(id="div-optim-table"), xs=12, md=6),
            ], className="mb-3"),

            # Trades table
            html.H6("Trade Log", className="mt-2",
                    style={"fontFamily": "'JetBrains Mono', monospace",
                           "color": THEME["muted"]}),
            html.Div(id="div-trades-table"),

        ], xs=12, md=9),
    ]),

    # Hidden data store
    dcc.Store(id="store-results"),

], fluid=True, style={"background": THEME["bg"], "minHeight": "100vh",
                       "color": THEME["text"]})


# ═══════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════

@callback(
    Output("store-results",  "data"),
    Output("div-status",     "children"),
    Input("btn-run",         "n_clicks"),
    State("inp-symbol",  "value"),
    State("inp-start",   "value"),
    State("inp-end",     "value"),
    State("inp-capital", "value"),
    State("inp-risk",    "value"),
    State("inp-rr",      "value"),
    State("inp-rt",      "value"),
    prevent_initial_call=True,
)
def run_backtest(n, symbol, start, end, capital, risk, rr, rt):
    try:
        bt = Backtest(
            symbol          = symbol.upper().strip(),
            start           = start,
            end             = end,
            initial_capital = float(capital),
            risk_per_trade  = float(risk) / 100,
            rr_ratio        = float(rr),
            rt_tolerance    = float(rt),
        )
        results = bt.run()
        trades  = results["trades"]
        eq      = results["equity_curve"]
        dd      = results["drawdown"]

        store = {
            "trades"  : trades.to_json() if not trades.empty else "{}",
            "equity"  : eq.tolist(),
            "drawdown": dd.tolist(),
            "metrics" : results["metrics"].to_dict() if results["metrics"] else {},
            "symbol"  : symbol.upper(),
        }
        return store, f"✅  {len(trades)} trades generated"

    except Exception as e:
        logger.exception("Backtest failed")
        return dash.no_update, f"❌  Error: {str(e)[:80]}"


@callback(
    Output("div-metrics",    "children"),
    Output("chart-equity",   "figure"),
    Output("chart-trades",   "figure"),
    Output("chart-monthly",  "figure"),
    Output("chart-r-hist",   "figure"),
    Output("div-trades-table", "children"),
    Input("store-results",   "data"),
    prevent_initial_call=True,
)
def update_charts(store):
    if not store:
        raise dash.exceptions.PreventUpdate

    # Reconstruct objects from store
    trades_json = store.get("trades", "{}")
    trades = pd.read_json(trades_json) if trades_json != "{}" else pd.DataFrame()

    eq = pd.Series(store.get("equity",   [10000]))
    dd = pd.Series(store.get("drawdown", [0]))

    # Metrics display
    from backtest.backtest import PerformanceMetrics
    m_dict = store.get("metrics", {})
    m = PerformanceMetrics(**{k: v for k, v in m_dict.items()
                               if k in PerformanceMetrics.__dataclass_fields__})
    metrics_row = build_metrics_row(m)

    # Charts
    fig_equity  = build_equity_chart(eq, dd)
    fig_trades  = build_trades_scatter(trades)
    fig_monthly = build_monthly_heatmap(trades)
    fig_r_hist  = build_r_histogram(trades)

    # Trades table
    if trades.empty:
        tbl = html.P("No trades.", style={"color": THEME["muted"]})
    else:
        cols = ["date","direction","entry_type","entry_price","stop_loss",
                "take_profit","exit_price","exit_reason","pnl_pct","r_multiple","dollar_pnl"]
        show = [c for c in cols if c in trades.columns]
        tbl = dash_table.DataTable(
            data=trades[show].round(3).to_dict("records"),
            columns=[{"name": c, "id": c} for c in show],
            page_size=15,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": THEME["surface"],
                          "color": THEME["muted"],
                          "fontFamily": "'JetBrains Mono', monospace",
                          "fontSize": "0.7rem",
                          "border": f"1px solid {THEME['border']}"},
            style_cell={"backgroundColor": THEME["bg"],
                        "color": THEME["text"],
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "0.75rem",
                        "border": f"1px solid {THEME['border']}",
                        "padding": "6px"},
            style_data_conditional=[
                {"if": {"filter_query": "{pnl_pct} > 0",
                        "column_id": "pnl_pct"},
                 "color": THEME["green"]},
                {"if": {"filter_query": "{pnl_pct} < 0",
                        "column_id": "pnl_pct"},
                 "color": THEME["red"]},
            ],
        )

    return metrics_row, fig_equity, fig_trades, fig_monthly, fig_r_hist, tbl


@callback(
    Output("div-optim-table", "children"),
    Input("btn-opt", "n_clicks"),
    State("inp-symbol",  "value"),
    State("inp-start",   "value"),
    State("inp-end",     "value"),
    State("inp-capital", "value"),
    State("inp-risk",    "value"),
    prevent_initial_call=True,
)
def run_optimiser(n, symbol, start, end, capital, risk):
    try:
        opt = Optimiser(
            symbol=symbol.upper().strip(),
            start=start, end=end,
            initial_capital=float(capital),
            risk=float(risk) / 100,
        )
        df = opt.run()

        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": THEME["surface"],
                          "color": THEME["muted"],
                          "fontFamily": "'JetBrains Mono', monospace",
                          "fontSize": "0.68rem",
                          "border": f"1px solid {THEME['border']}"},
            style_cell={"backgroundColor": THEME["bg"],
                        "color": THEME["text"],
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "0.7rem",
                        "border": f"1px solid {THEME['border']}",
                        "padding": "5px"},
            style_data_conditional=[{
                "if": {"row_index": 0},
                "backgroundColor": "rgba(63,185,80,0.12)",
                "border": f"1px solid {THEME['green']}",
            }],
        )
    except Exception as e:
        return html.P(f"Optimiser error: {e}", style={"color": THEME["red"]})


# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
