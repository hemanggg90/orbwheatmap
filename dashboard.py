"""
Streamlit control-room dashboard for orbwheatmap.py.

Run with:
    streamlit run dashboard.py

Everything the CLI menu in orbwheatmap.main() could do (pick paper/live mode,
start/stop the trading loop, view performance) is available here, plus a live
sector heatmap, an open-positions table with manual close, a Config editor,
and a log tail - all without touching a terminal.
"""

import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import orbwheatmap as ow
import backtest_engine as bt

st.set_page_config(page_title="ORB Wheatmap Control Room", layout="wide")


# ============================================
# Singleton trading-system handle
# ============================================
# st.cache_resource keeps ONE TradingSystem object alive across Streamlit
# reruns (and across browser tabs) so the background trading thread and its
# in-memory positions aren't recreated every time the page refreshes.

@st.cache_resource(show_spinner=False)
def _get_system(mode: str) -> ow.TradingSystem:
    return ow.TradingSystem(mode=mode)


@st.cache_resource(show_spinner=False)
def _get_nse_fetcher() -> ow.NSEDataFetcher:
    return ow.NSEDataFetcher()


def _active_system() -> "ow.TradingSystem | None":
    mode = st.session_state.get("active_mode")
    if mode is None:
        return None
    return _get_system(mode)


if "active_mode" not in st.session_state:
    st.session_state.active_mode = None
if "live_confirmed" not in st.session_state:
    st.session_state.live_confirmed = False


# ============================================
# Sidebar - control panel
# ============================================

st.sidebar.title("Control Panel")

mode = st.sidebar.radio("Trading mode", ["PAPER", "LIVE"], horizontal=True)

if mode == "LIVE":
    st.sidebar.error(
        "LIVE mode places REAL orders with REAL money via your Dhan account."
    )
    creds_ok = (
        ow.Config.DHAN_CLIENT_ID not in ("", "your_client_id")
        and ow.Config.DHAN_ACCESS_TOKEN not in ("", "your_access_token")
    )
    if not creds_ok:
        st.sidebar.warning(
            "DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN are not set (env vars or .env). "
            "Set them in the Config tab or your .env file before going live."
        )
    live_checkbox = st.sidebar.checkbox("I understand this trades real money")
    live_text = st.sidebar.text_input('Type CONFIRM to arm live trading')
    armed = live_checkbox and live_text.strip().upper() == "CONFIRM" and creds_ok
    st.session_state.live_confirmed = armed
    if armed:
        ow.Config.LIVE_TRADING_ENABLED = True
else:
    st.session_state.live_confirmed = False

col_start, col_stop = st.sidebar.columns(2)
start_disabled = mode == "LIVE" and not st.session_state.live_confirmed
start_clicked = col_start.button("Start", type="primary", disabled=start_disabled, width="stretch")
stop_clicked = col_stop.button("Stop", width="stretch")

if start_clicked:
    current = _active_system()
    if current is not None and current.mode != mode:
        current.stop()
    system = _get_system(mode)
    system.start()
    st.session_state.active_mode = mode
    st.rerun()

if stop_clicked:
    current = _active_system()
    if current is not None:
        current.stop()
    st.rerun()

st.sidebar.divider()
system = _active_system()
if system is None:
    st.sidebar.info("Not started yet.")
else:
    badge = "🟢 running" if system.running else "🔴 stopped"
    st.sidebar.markdown(f"**{system.mode}** — {badge}")
    st.sidebar.caption(f"Dhan connected: {'yes' if system.client_manager.is_connected else 'no'}")

st.sidebar.divider()
st.sidebar.caption("Auto-refreshing every 15s to track live LTP.")
st.markdown(
    '<meta http-equiv="refresh" content="15">',
    unsafe_allow_html=True,
)


# ============================================
# Header
# ============================================

st.title("ORB Sector-Heatmap Options Trading — Control Room")

status = system.get_status() if system else {
    "mode": mode, "running": False, "market_open": ow.is_market_open(),
    "ist_time": ow.ist_now().strftime('%Y-%m-%d %H:%M:%S'),
    "entry_window": ow.is_entry_window(), "squareoff_window": ow.is_squareoff_window(),
    "open_positions": 0, "trades_today": 0, "daily_pnl": 0.0,
    "trading_disabled": False, "dhan_connected": False, "performance": {},
}

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("IST time", status["ist_time"].split(" ")[1])
m2.metric("Market", "Open" if status["market_open"] else "Closed")
m3.metric("Entry window", "Open" if status["entry_window"] else "Closed")
m4.metric("Open positions", status["open_positions"])
m5.metric("Trades today", status["trades_today"])
m6.metric("Daily P&L", f"Rs.{status['daily_pnl']:,.2f}")

if status["trading_disabled"]:
    st.error("New entries are currently DISABLED by the risk manager (daily loss / consecutive-loss halt).")


# ============================================
# Tabs
# ============================================

tab_overview, tab_heatmap, tab_positions, tab_history, tab_backtest, tab_config, tab_logs = st.tabs(
    ["Overview", "Sector Heatmap", "Positions", "Trade History", "Backtest", "Config", "Logs"]
)

# ---- Overview ----
with tab_overview:
    if status.get("regime_adaptive_enabled"):
        st.subheader("Current market regime")
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Regime", status.get("regime", "UNKNOWN"))
        r2.metric("ADX", f"{status.get('regime_adx', 0):.1f}")
        r3.metric("ATR percentile", f"{status.get('regime_atr_percentile', 50):.0f}")
        r4.metric("India VIX", f"{status.get('regime_vix', 0):.1f}")
        r5.metric("Breadth", f"{status.get('regime_breadth', 0.5):.2f}")
        st.caption(
            "Regime is recomputed every scan cycle and gates new entries, direction, target R-multiple, "
            "ATR-aware stop sizing, and (TRENDING regimes) a trailing stop - see Config's REGIME_* settings."
        )
        st.divider()

    perf = status.get("performance", {})
    if not perf or perf.get("total_trades", 0) == 0:
        st.info("No closed trades yet this session.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total trades", perf["total_trades"])
        c2.metric("Win rate", f"{perf['win_rate']:.1f}%")
        c3.metric("Total P&L", f"Rs.{perf['total_pnl']:,.2f}")
        c4.metric("Profit factor", f"{perf['profit_factor']:.2f}")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Avg win", f"Rs.{perf['avg_win']:,.2f}")
        c6.metric("Avg loss", f"Rs.{perf['avg_loss']:,.2f}")
        c7.metric("Max consec. wins", perf["max_consecutive_wins"])
        c8.metric("Max consec. losses", perf["max_consecutive_losses"])

    st.subheader("Sector universe")
    sec_df = pd.DataFrame(
        [(sector, ", ".join(stocks)) for sector, stocks in ow.Config.SECTOR_STOCKS.items()],
        columns=["Sector", "Stocks tracked"],
    )
    st.dataframe(sec_df, width="stretch", hide_index=True)

# ---- Sector Heatmap ----
with tab_heatmap:
    st.caption("Pulls live sector-index performance directly from the NSE website.")
    if st.button("Fetch heatmap now"):
        # The heatmap only needs NSE data, not Dhan - use the running system's
        # fetcher if there is one, otherwise a plain NSEDataFetcher (no need to
        # spin up a full Dhan login just to show the sector heatmap).
        nse_fetcher = system.data_fetcher.nse if system else _get_nse_fetcher()
        with st.spinner("Fetching sector heatmap from NSE..."):
            try:
                df = nse_fetcher.get_sector_heatmap()
            except Exception as e:
                st.error(f"Could not fetch the sector heatmap: {e}")
                df = pd.DataFrame()
        st.session_state["heatmap_df"] = df

    df = st.session_state.get("heatmap_df")
    if df is None or df.empty:
        st.info("Click 'Fetch heatmap now' to pull the latest sector performance.")
    else:
        def _color_change(val: float) -> str:
            # Manual red/green shading - avoids a hard dependency on matplotlib,
            # which pandas' Styler.background_gradient(cmap=...) requires.
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            intensity = min(abs(v) / 2.0, 1.0)
            alpha = 0.15 + 0.55 * intensity
            if v > 0:
                return f"background-color: rgba(0, 170, 0, {alpha:.2f})"
            if v < 0:
                return f"background-color: rgba(220, 0, 0, {alpha:.2f})"
            return ""

        st.dataframe(
            df.style.map(_color_change, subset=["change_percent"]),
            width="stretch", hide_index=True,
        )
        st.bar_chart(df.set_index("sector")["change_percent"])

# ---- Positions ----
with tab_positions:
    if system is None:
        st.info("Start the system to see live positions.")
    else:
        open_positions = [p for p in system.trading_engine.positions if p.status == ow.TradeStatus.OPEN]
        if not open_positions:
            st.info("No open positions.")
        else:
            sec_ids = [p.security_id for p in open_positions]
            premiums = system.data_fetcher.get_option_premiums_batch(sec_ids)
            rows = []
            for p in open_positions:
                ltp = premiums.get(str(p.security_id), 0.0)
                unreal = (ltp - p.entry_price) * p.remaining_quantity if ltp else 0.0
                rows.append({
                    "symbol": p.stock_symbol, "underlying": p.underlying_symbol, "sector": p.sector,
                    "type": p.option_type, "entry": p.entry_price, "ltp": ltp,
                    "qty": p.remaining_quantity, "unrealized_pnl": unreal,
                    "stop": p.stop_loss, "underlying_stop": p.underlying_stop_loss,
                    "underlying_target": p.underlying_target, "entry_time": p.entry_time,
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            st.subheader("Manual close")
            choice = st.selectbox("Position", [p.stock_symbol for p in open_positions])
            if st.button("Close this position now", type="secondary"):
                pos = next(p for p in open_positions if p.stock_symbol == choice)
                ltp = premiums.get(str(pos.security_id), pos.entry_price)
                system.trading_engine._exit_full(pos, ltp, ow.ExitReason.MANUAL.value)
                st.success(f"Closed {choice} at Rs.{ltp:.2f}")
                st.rerun()

        closed = system.trading_engine.closed_trades
        if closed:
            st.subheader("Closed this session")
            st.dataframe(pd.DataFrame([{
                "symbol": p.stock_symbol, "underlying": p.underlying_symbol, "exit_reason": p.exit_reason,
                "pnl": p.pnl, "exit_time": p.exit_time,
            } for p in closed]), width="stretch", hide_index=True)

# ---- Trade History ----
with tab_history:
    label_paths = [("Paper trades", ow.Config.PAPER_TRADES_FILE), ("Live trades", ow.Config.LIVE_TRADES_FILE),
                    ("Paper trades (full)", "paper_trades_full.csv")]
    chosen = st.selectbox("File", [l for l, _ in label_paths])
    path = dict(label_paths)[chosen]
    if not os.path.exists(path):
        st.info(f"{path} does not exist yet.")
    else:
        df = pd.read_csv(path)
        st.caption(f"{len(df)} row(s) in {path}")
        if not df.empty and "net_pnl" in df.columns:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total P&L", f"Rs.{df['net_pnl'].sum():,.2f}")
            c2.metric("Win rate", f"{(len(df[df['net_pnl'] > 0]) / len(df)) * 100:.1f}%")
            c3.metric("Trades", len(df))
        elif not df.empty and "pnl" in df.columns:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total P&L", f"Rs.{df['pnl'].sum():,.2f}")
            closed_only = df[df["status"] == "CLOSED"] if "status" in df.columns else df
            c2.metric("Win rate", f"{(len(closed_only[closed_only['pnl'] > 0]) / len(closed_only) * 100) if len(closed_only) else 0:.1f}%")
            c3.metric("Trades", len(df))
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), file_name=os.path.basename(path))

# ---- Backtest ----
with tab_backtest:
    st.caption(
        "Replays the exact entry rule (StrategyEngine) and risk gates (RiskManager) over "
        "historical Dhan 5-min candles - including the regime-adaptive layer (market-regime "
        "classifier, displacement->pullback entry, ATR-aware dynamic stop, regime-conditional "
        "target, trending trailing stop) when ENABLE_REGIME_ADAPTIVE is on below. Known "
        "approximations: (1) no historical NSE sector-heatmap feed exists, so this scans the "
        "full configured stock+index universe rather than only the day's leading/lagging "
        "sectors, and breadth compares each stock to its own day's open rather than the prior "
        "trading day's close; (2) no historical option-premium data is fetched - P&L is an "
        "estimated R-multiple of the risk_amount used for position sizing, scaled by the "
        "regime's risk multiplier - not a real option fill-price replay. "
        "See backtest_engine.py's module docstring for full detail."
    )

    bt_regime_enabled = st.checkbox(
        "Use regime-adaptive layer for this backtest", value=bool(ow.Config.ENABLE_REGIME_ADAPTIVE), key="bt_regime_enabled",
        help="Off = original deterministic no-pullback-wait rule with a fixed 1:2R target (faster, no NIFTY/VIX fetch needed).",
    )

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        bt_from = st.date_input("From date", value=datetime.now().date().replace(day=1))
    with bc2:
        bt_to = st.date_input("To date", value=datetime.now().date())
    with bc3:
        bt_capital = st.number_input("Capital (Rs.)", value=float(ow.Config.CAPITAL), step=1000.0, key="bt_capital")
    with bc4:
        bt_risk_pct = st.number_input("Risk per trade (%)", value=float(ow.Config.RISK_PER_TRADE), step=0.1, key="bt_risk_pct")

    universe = bt.default_universe()
    all_symbols = [u["symbol"] for u in universe]
    chosen_symbols = st.multiselect(
        "Universe (default: every configured sector stock + NIFTY/BANKNIFTY)",
        options=all_symbols, default=all_symbols,
    )

    if st.button("Run backtest", type="primary"):
        selected = [u for u in universe if u["symbol"] in chosen_symbols]
        progress_box = st.empty()
        log_lines: list[str] = []

        def _progress(msg: str) -> None:
            log_lines.append(msg)
            progress_box.code("\n".join(log_lines[-15:]))

        with st.spinner("Fetching historical candles and replaying the strategy..."):
            try:
                trades_df, summary_df = bt.run_backtest(
                    bt_from.strftime("%Y-%m-%d"), bt_to.strftime("%Y-%m-%d"),
                    symbols=selected, capital=bt_capital, risk_per_trade=bt_risk_pct,
                    regime_adaptive=bt_regime_enabled,
                    progress_cb=_progress,
                )
                bt.save_backtest_csvs(trades_df, summary_df)
                st.session_state["bt_trades_df"] = trades_df
                st.session_state["bt_summary_df"] = summary_df
                st.success(f"Backtest complete - wrote {bt.TRADES_CSV} and {bt.RESULTS_CSV}.")
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    trades_df = st.session_state.get("bt_trades_df")
    summary_df = st.session_state.get("bt_summary_df")

    if summary_df is not None and not summary_df.empty:
        row = summary_df.iloc[0]
        st.subheader("Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total trades", int(row.get("total_trades", 0)))
        c2.metric("Win rate", f"{row.get('win_rate_pct', 0):.1f}%")
        c3.metric("Estimated total P&L", f"Rs.{row.get('total_estimated_pnl', 0):,.2f}")
        c4.metric("Profit factor", f"{row.get('profit_factor', 0):.2f}")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Avg R-multiple", f"{row.get('avg_r_multiple', 0):.2f}R")
        c6.metric("Max drawdown (est.)", f"Rs.{row.get('max_drawdown_estimate', 0):,.2f}")
        c7.metric("Final equity (est.)", f"Rs.{row.get('final_equity_estimate', 0):,.2f}")
        c8.metric("Signals risk-rejected", int(row.get("signals_risk_rejected", 0)))
        st.dataframe(summary_df, width="stretch", hide_index=True)
        st.download_button("Download results CSV", summary_df.to_csv(index=False), file_name=bt.RESULTS_CSV)

    if trades_df is not None and not trades_df.empty:
        executed = trades_df[trades_df["exit_reason"] != "RISK_REJECTED"]
        if not executed.empty:
            equity = bt_capital + executed.sort_values("exit_time")["estimated_pnl"].cumsum()
            st.subheader("Estimated equity curve")
            st.line_chart(equity.reset_index(drop=True))

        st.subheader("Backtested trades")
        st.dataframe(trades_df, width="stretch", hide_index=True)
        st.download_button("Download trades CSV", trades_df.to_csv(index=False), file_name=bt.TRADES_CSV)
    elif trades_df is not None:
        st.info("No trades were generated for this universe/date range.")

    for label, path in [("Last saved backtest trades", bt.TRADES_CSV), ("Last saved backtest results", bt.RESULTS_CSV)]:
        if os.path.exists(path) and trades_df is None:
            st.caption(f"{label}: {path} (from a previous run)")

# ---- Config ----
with tab_config:
    st.caption("Edits apply immediately to the live Config (in-memory) — most thresholds are read fresh on every scan/check.")
    with st.form("config_form"):
        c1, c2 = st.columns(2)
        with c1:
            capital = st.number_input("CAPITAL", value=float(ow.Config.CAPITAL), step=1000.0)
            risk_pct = st.number_input("RISK_PER_TRADE (%)", value=float(ow.Config.RISK_PER_TRADE), step=0.1)
            max_open = st.number_input("MAX_OPEN_POSITIONS", value=int(ow.Config.MAX_OPEN_POSITIONS), step=1)
            max_daily_trades = st.number_input("MAX_DAILY_TRADES", value=int(ow.Config.MAX_DAILY_TRADES), step=1)
            max_daily_loss = st.number_input("MAX_DAILY_LOSS (Rs.)", value=float(ow.Config.MAX_DAILY_LOSS), step=100.0)
            max_per_sector = st.number_input("MAX_POSITIONS_PER_SECTOR", value=int(ow.Config.MAX_POSITIONS_PER_SECTOR), step=1)
            consec_loss_halt = st.number_input("CONSECUTIVE_LOSS_HALT", value=int(ow.Config.CONSECUTIVE_LOSS_HALT), step=1,
                                                help="Stop opening new trades after this many consecutive losing closes.")
            max_trades_per_symbol = st.number_input("MAX_TRADES_PER_SYMBOL_PER_DAY", value=int(ow.Config.MAX_TRADES_PER_SYMBOL_PER_DAY), step=1,
                                                      help="One-shot design only truly supports 1 - see orbwheatmap.py StrategyEngine.")
        with c2:
            entry_start = st.text_input("ENTRY_START_TIME (HH:MM)", value=ow.Config.ENTRY_START_TIME)
            entry_end = st.text_input("ENTRY_END_TIME (HH:MM)", value=ow.Config.ENTRY_END_TIME)
            squareoff = st.text_input("EOD_SQUAREOFF_TIME (HH:MM)", value=ow.Config.EOD_SQUAREOFF_TIME)
            min_score = st.number_input("MIN_SIGNAL_SCORE", value=float(ow.Config.MIN_SIGNAL_SCORE), step=1.0)
            volume_mult = st.number_input("VOLUME_CONFIRMATION_MULTIPLE", value=float(ow.Config.VOLUME_CONFIRMATION_MULTIPLE), step=0.1,
                                           help="Last candle's volume must be >= this x the prior VOLUME_LOOKBACK_CANDLES average (stocks only - indices skip this check).")
            target_dte_str = st.text_input("TARGET_DTE (days to expiry, blank = nearest)", value=("" if ow.Config.TARGET_DTE is None else str(ow.Config.TARGET_DTE)),
                                            help="0=0DTE, 1=1DTE, 2/3=2-3DTE, etc. Leave blank for the nearest valid expiry (default).")
            client_id = st.text_input("DHAN_CLIENT_ID", value=ow.Config.DHAN_CLIENT_ID)
            access_token = st.text_input("DHAN_ACCESS_TOKEN", value=ow.Config.DHAN_ACCESS_TOKEN, type="password")

        st.divider()
        st.caption(
            "Regime-adaptive layer: market-regime classifier, displacement->pullback entry, "
            "ATR-aware dynamic stop, regime-conditional target, and trending trailing stop."
        )
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            regime_enabled = st.checkbox("ENABLE_REGIME_ADAPTIVE", value=bool(ow.Config.ENABLE_REGIME_ADAPTIVE),
                                          help="Off = original deterministic no-pullback-wait rule with a fixed 1:2R target.")
            atr_stop_mult = st.number_input("ATR_STOP_MULTIPLIER", value=float(ow.Config.ATR_STOP_MULTIPLIER), step=0.1)
        with rc2:
            pullback_max = st.number_input("PULLBACK_MAX_CANDLES", value=int(ow.Config.PULLBACK_MAX_CANDLES), step=1)
            order_flow_ratio = st.number_input("ORDER_FLOW_MIN_UP_RATIO", value=float(ow.Config.ORDER_FLOW_MIN_UP_RATIO), step=0.05)
        with rc3:
            breadth_bull = st.number_input("REGIME_BREADTH_BULLISH_MIN", value=float(ow.Config.REGIME_BREADTH_BULLISH_MIN), step=0.05)
            breadth_bear = st.number_input("REGIME_BREADTH_BEARISH_MAX", value=float(ow.Config.REGIME_BREADTH_BEARISH_MAX), step=0.05)

        if st.form_submit_button("Apply changes"):
            ow.Config.CAPITAL = capital
            ow.Config.RISK_PER_TRADE = risk_pct
            ow.Config.MAX_OPEN_POSITIONS = int(max_open)
            ow.Config.MAX_DAILY_TRADES = int(max_daily_trades)
            ow.Config.MAX_DAILY_LOSS = max_daily_loss
            ow.Config.MAX_POSITIONS_PER_SECTOR = int(max_per_sector)
            ow.Config.CONSECUTIVE_LOSS_HALT = int(consec_loss_halt)
            ow.Config.MAX_TRADES_PER_SYMBOL_PER_DAY = int(max_trades_per_symbol)
            ow.Config.ENTRY_START_TIME = entry_start
            ow.Config.ENTRY_END_TIME = entry_end
            ow.Config.EOD_SQUAREOFF_TIME = squareoff
            ow.Config.MIN_SIGNAL_SCORE = min_score
            ow.Config.VOLUME_CONFIRMATION_MULTIPLE = volume_mult
            ow.Config.TARGET_DTE = int(target_dte_str) if target_dte_str.strip() else None
            ow.Config.DHAN_CLIENT_ID = client_id
            ow.Config.DHAN_ACCESS_TOKEN = access_token
            ow.Config.ENABLE_REGIME_ADAPTIVE = regime_enabled
            ow.Config.ATR_STOP_MULTIPLIER = atr_stop_mult
            ow.Config.PULLBACK_MAX_CANDLES = int(pullback_max)
            ow.Config.ORDER_FLOW_MIN_UP_RATIO = order_flow_ratio
            ow.Config.REGIME_BREADTH_BULLISH_MIN = breadth_bull
            ow.Config.REGIME_BREADTH_BEARISH_MAX = breadth_bear
            st.success("Config updated. Credential changes only take effect for a system started after this point.")

    st.divider()
    if st.button("Reset today's risk state (trades_today / daily P&L / halt)"):
        if os.path.exists(ow.Config.RISK_STATE_FILE):
            os.remove(ow.Config.RISK_STATE_FILE)
        if system:
            system.trading_engine.risk_manager.state = system.trading_engine.risk_manager._load_or_reset()
        st.success("Risk state reset.")
        st.rerun()

# ---- Logs ----
with tab_logs:
    n_lines = st.slider("Lines to show", 20, 500, 100)
    log_path = "trading_system.log"
    if not os.path.exists(log_path):
        st.info("No log file yet.")
    else:
        with open(log_path, "r", errors="ignore") as f:
            lines = f.readlines()[-n_lines:]
        st.code("".join(lines), language="log")
