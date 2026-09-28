"""
Historical backtesting engine for the direct-breakout ORB strategy in orbwheatmap.py.

Design goal: reuse the EXACT production entry logic (StrategyEngine's per-symbol
arm/fire state machine, driven off Stock snapshots built the same way
DataFetcher._build_stock_from_ohlc does) instead of re-implementing the rule from
scratch, so the backtest can't silently drift from what paper/live actually does.

Historical 5-minute OHLCV candles are pulled from Dhan (client.intraday_minute_data),
resolved through the same DhanInstrumentMaster security master the live engine uses.

LIMITATIONS (explicit, not hidden):
  - No historical NSE sector-heatmap feed exists, so the live "scan only the day's
    leading/lagging sectors" universe filter can't be replayed. By default this
    backtest scans every configured sector stock (deduplicated) + NIFTY/BANKNIFTY for
    a direct breakout - it will surface MORE candidate signals than the live system's
    heatmap-gated universe would.
  - No historical option-chain premium data is fetched. The real strategy's stop/
    target are structural levels on the UNDERLYING (see
    TradingEngine._check_and_apply_exits) with the option premium tracked only for
    P&L bookkeeping and a -30% emergency stop - so this engine simulates the
    stop/target exactly on the underlying, then estimates rupee P&L as an R-multiple
    of the same risk_amount the live position sizer uses (Config.CAPITAL x
    Config.RISK_PER_TRADE% x the regime's risk multiplier), i.e. a full stop-loss ~=
    -1R, a fixed 1:2 target ~= +2R. This ignores real option slippage, theta decay,
    and IV changes - it is an estimate, not a real fill-price replay.
  - Position/risk-gate simulation (MAX_OPEN_POSITIONS, MAX_DAILY_TRADES,
    MAX_POSITIONS_PER_SECTOR, MAX_DAILY_LOSS, CONSECUTIVE_LOSS_HALT) mirrors
    RiskManager.check_trade_allowed(), evaluated once per 5-minute candle close
    (the live loop scans every 60s / monitors every 15s - candle-granularity is the
    finest this can reproduce from candle data alone).
  - Entry/exit fills are marked at the exact stop/target price once a candle's
    high/low crosses it (not simulated slippage), and a position opened on a candle
    close is only monitored for exit from the NEXT candle onward.
  - Config.ENABLE_REGIME_ADAPTIVE=True (the default) replays the FULL regime-adaptive
    architecture: a MarketRegime is recomputed every candle from the regime index's
    (default NIFTY) candles-so-far + India VIX (best-effort - skipped if VIX isn't in
    the security master for the historical range) + this-cycle breadth (fraction of the
    scanned universe trading above its own day's open so far - there's no historical
    previous-close feed here, so breadth uses intraday day-open reference instead of
    the live engine's prior-trading-day-close reference); this gates new entries,
    entry direction, the target R-multiple, ATR-aware dynamic stop sizing, and (for
    TRENDING regimes) a trailing stop exactly as orbwheatmap.TradingEngine does. Set it
    to False to backtest the original deterministic no-pullback-wait rule instead.

Usage (CLI):
    python backtest_engine.py --from 2026-06-01 --to 2026-09-01

Usage (library, e.g. from dashboard.py):
    from backtest_engine import run_backtest, save_backtest_csvs
    trades_df, summary_df = run_backtest("2026-06-01", "2026-09-01")
    save_backtest_csvs(trades_df, summary_df)
"""

import argparse
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()  # orbwheatmap.Config reads DHAN_CLIENT_ID/DHAN_ACCESS_TOKEN via os.getenv at import time
except ImportError:
    pass

import orbwheatmap as ow

TRADES_CSV = "backtest_trades.csv"
RESULTS_CSV = "backtest_results.csv"


# ============================================
# Universe
# ============================================

def default_universe() -> List[Dict[str, Any]]:
    """Every configured sector stock (deduped, first sector wins) + the
    independently-scanned indices - see module docstring on the heatmap-filter gap."""
    universe: List[Dict[str, Any]] = []
    seen = set()
    for sector, stocks in ow.Config.SECTOR_STOCKS.items():
        for s in stocks:
            s = str(s).strip().upper()
            if s and s not in seen:
                seen.add(s)
                universe.append({"symbol": s, "sector": sector, "is_index": False})
    for idx in ow.Config.TRADING_INDICES:
        universe.append({"symbol": idx, "sector": "INDEX", "is_index": True})
    return universe


# ============================================
# Historical data
# ============================================

def fetch_historical_candles(
    client: Any,
    security_id: str,
    exchange_segment: str,
    instrument_type: str,
    from_date: datetime,
    to_date: datetime,
    interval: str = "5",
) -> pd.DataFrame:
    """Pull 5-min OHLCV candles for [from_date, to_date] in <=59-day chunks (Dhan's
    intraday endpoint is a narrow-range API) and stitch them into one frame."""
    frames: List[pd.DataFrame] = []
    cur = from_date
    while cur <= to_date:
        chunk_end = min(cur + timedelta(days=59), to_date)
        try:
            response = client.intraday_minute_data(
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument_type=instrument_type,
                from_date=cur.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=chunk_end.strftime("%Y-%m-%d 23:59:59"),
                interval=int(interval),
            )
            df = ow.DhanDataFetcher._normalize_ohlc(response)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            ow.logger.warning(f"BACKTEST_DATA_FETCH_FAILED: {security_id} {cur.date()}->{chunk_end.date()}: {e}")
        cur = chunk_end + timedelta(days=1)
        time.sleep(0.25)  # be gentle with the historical endpoint

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="Timestamp").sort_values("Timestamp").reset_index(drop=True)
    return out


def fetch_historical_daily(
    client: Any, security_id: str, exchange_segment: str, instrument_type: str,
    from_date: datetime, to_date: datetime,
) -> pd.DataFrame:
    """DAY-timeframe candles (one row per trading day) - used only for the regime
    layer's daily-range history (see RegimeClassifier). A wider lookback than the
    backtest's own from_date is fetched so early days in the range still have a
    percentile baseline to compare against."""
    try:
        response = client.historical_daily_data(
            security_id=security_id, exchange_segment=exchange_segment, instrument_type=instrument_type,
            from_date=(from_date - timedelta(days=60)).strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
        )
        return ow.DhanDataFetcher._normalize_ohlc(response)
    except Exception as e:
        ow.logger.warning(f"BACKTEST_DAILY_FETCH_FAILED: {security_id}: {e}")
        return pd.DataFrame()


# ============================================
# Per-candle Stock snapshot (mirrors DataFetcher._build_stock_from_ohlc)
# ============================================

def _build_snapshot(symbol: str, sector: str, day_slice: pd.DataFrame, has_real_volume: bool) -> ow.Stock:
    candle_minutes = int(ow.Config.CANDLE_TIMEFRAME)
    orb_candles = max(1, ow.Config.ORB_MINUTES // candle_minutes)
    orb_period = min(orb_candles, len(day_slice))

    current_price = float(day_slice["Close"].iloc[-1])
    orb_high = float(day_slice["High"].iloc[:orb_period].max())
    orb_low = float(day_slice["Low"].iloc[:orb_period].min())
    orb_range = ((orb_high - orb_low) / orb_low) * 100 if orb_low > 0 else 0.0
    orb_complete = len(day_slice) >= orb_candles

    has_volume = has_real_volume and "Volume" in day_slice and day_slice["Volume"].sum() > 0
    if has_volume:
        typical_price = (day_slice["High"] + day_slice["Low"] + day_slice["Close"]) / 3.0
        vwap = float((typical_price * day_slice["Volume"]).sum() / day_slice["Volume"].sum())
        prior = day_slice["Volume"].iloc[:-1].tail(ow.Config.VOLUME_LOOKBACK_CANDLES)
        avg_volume_20 = float(prior.mean()) if not prior.empty else 0.0
        last_volume = int(day_slice["Volume"].iloc[-1])
    else:
        typical_price = (day_slice["High"] + day_slice["Low"] + day_slice["Close"]) / 3.0
        vwap = float(typical_price.mean())
        avg_volume_20 = 0.0
        last_volume = 0

    atr = ow.compute_atr(day_slice, ow.Config.REGIME_ATR_PERIOD) if ow.Config.ENABLE_REGIME_ADAPTIVE else 0.0

    return ow.Stock(
        symbol=symbol, name=symbol, sector=sector,
        current_price=current_price, change_percent=0.0,  # not used by the entry rule; see module docstring
        volume=last_volume,
        orb_high=orb_high, orb_low=orb_low, orb_range=orb_range,
        orb_complete=orb_complete,
        vwap=vwap, avg_volume_20=avg_volume_20, candle_count=len(day_slice),
        has_real_volume=has_volume,
        current_high=float(day_slice["High"].iloc[-1]), current_low=float(day_slice["Low"].iloc[-1]),
        current_open=float(day_slice["Open"].iloc[-1]), atr=atr,
    )


def _compute_regime_at(
    nifty_day_slice: pd.DataFrame, daily_range_history: Optional[pd.Series], vix_value: float,
    day_frames: Dict[str, pd.DataFrame], universe_by_symbol: Dict[str, Dict[str, Any]], ts: pd.Timestamp,
) -> "ow.RegimeSnapshot":
    """One regime read as of `ts` - mirrors DataFetcher.get_regime_snapshot(), but
    breadth here compares each stock's current price to ITS OWN day's open (no
    historical previous-close feed available in this backtest) rather than the prior
    trading day's close the live engine uses."""
    if nifty_day_slice is None or nifty_day_slice.empty:
        return ow.RegimeSnapshot(regime=ow.MarketRegime.UNKNOWN)

    green, total = 0, 0
    for symbol, df in day_frames.items():
        if universe_by_symbol.get(symbol, {}).get("is_index"):
            continue
        slice_ = df[df["Timestamp"] <= ts]
        if slice_.empty:
            continue
        total += 1
        if float(slice_["Close"].iloc[-1]) > float(slice_["Open"].iloc[0]):
            green += 1
    breadth_ratio = (green / total) if total else 0.5

    atr_history = daily_range_history if daily_range_history is not None else pd.Series(dtype=float)
    return ow.RegimeClassifier().classify(nifty_day_slice, atr_history, vix_value, breadth_ratio, 0.0)


# ============================================
# Lightweight in-memory risk gate (mirrors RiskManager, no disk state)
# ============================================

class _SimRiskState:
    def __init__(self) -> None:
        self.trades_today = 0
        self.realized_pnl = 0.0
        self.trading_disabled = False
        self.consecutive_losses = 0
        self.halt_reason = ""

    def check_trade_allowed(self, sector: str, open_positions: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if self.trading_disabled:
            reasons.append(self.halt_reason)
        if len(open_positions) >= ow.Config.MAX_OPEN_POSITIONS:
            reasons.append(f"already at MAX_OPEN_POSITIONS ({ow.Config.MAX_OPEN_POSITIONS})")
        if self.trades_today >= ow.Config.MAX_DAILY_TRADES:
            reasons.append(f"already at MAX_DAILY_TRADES ({ow.Config.MAX_DAILY_TRADES})")
        sector_count = sum(1 for p in open_positions if p["sector"] == sector)
        if sector_count >= ow.Config.MAX_POSITIONS_PER_SECTOR:
            reasons.append(f"already at MAX_POSITIONS_PER_SECTOR ({ow.Config.MAX_POSITIONS_PER_SECTOR}) for {sector}")
        return len(reasons) == 0, reasons

    def record_open(self) -> None:
        self.trades_today += 1

    def record_close(self, pnl: float) -> None:
        self.realized_pnl += pnl
        self.consecutive_losses = self.consecutive_losses + 1 if pnl <= 0 else 0
        if self.realized_pnl <= -ow.Config.MAX_DAILY_LOSS:
            self.trading_disabled = True
            self.halt_reason = f"MAX_DAILY_LOSS of Rs.{ow.Config.MAX_DAILY_LOSS:,.0f} reached (realized P&L Rs.{self.realized_pnl:,.2f})"
        elif self.consecutive_losses >= ow.Config.CONSECUTIVE_LOSS_HALT:
            self.trading_disabled = True
            self.halt_reason = f"{self.consecutive_losses} consecutive losing trades (>= CONSECUTIVE_LOSS_HALT of {ow.Config.CONSECUTIVE_LOSS_HALT})"


# ============================================
# Core simulation
# ============================================

def _settle_exit(pos: Dict[str, Any], ts: pd.Timestamp, exit_price: float, exit_reason: str, risk_amount: float) -> Dict[str, Any]:
    risk_per_share = abs(pos["entry_price"] - pos["initial_stop_loss"])
    move = (exit_price - pos["entry_price"]) if pos["direction"] == "UP" else (pos["entry_price"] - exit_price)
    r_multiple = (move / risk_per_share) if risk_per_share > 0 else 0.0
    estimated_pnl = r_multiple * risk_amount * pos.get("risk_multiplier", 1.0)
    return {
        **pos,
        "exit_time": ts, "exit_price": exit_price, "exit_reason": exit_reason,
        "underlying_points": round(move, 4), "r_multiple": round(r_multiple, 3),
        "estimated_pnl": round(estimated_pnl, 2),
    }


def _update_trailing_stop(pos: Dict[str, Any], price: float) -> None:
    """Mirrors TradingEngine._update_trailing_stop: once price has moved
    REGIME_TRAIL_TRIGGER_R in favor, ratchet pos['stop_loss'] behind price by
    REGIME_TRAIL_ATR_MULTIPLE x ATR-at-entry - never loosens, only tightens."""
    if not pos.get("trailing_enabled"):
        return
    initial_risk = abs(pos["entry_price"] - pos["initial_stop_loss"])
    if initial_risk <= 0:
        return
    trigger_distance = ow.Config.REGIME_TRAIL_TRIGGER_R * initial_risk
    trail_distance = (ow.Config.REGIME_TRAIL_ATR_MULTIPLE * pos.get("atr_at_entry", 0.0)) or initial_risk

    if pos["direction"] == "UP":
        if (price - pos["entry_price"]) >= trigger_distance:
            candidate = price - trail_distance
            if candidate > pos["stop_loss"]:
                pos["stop_loss"] = candidate
    else:
        if (pos["entry_price"] - price) >= trigger_distance:
            candidate = price + trail_distance
            if candidate < pos["stop_loss"]:
                pos["stop_loss"] = candidate


def _simulate_day(
    symbol_frames: Dict[str, pd.DataFrame],
    universe_by_symbol: Dict[str, Dict[str, Any]],
    risk_amount: float,
    nifty_day_df: Optional[pd.DataFrame] = None,
    daily_range_history: Optional[pd.Series] = None,
    vix_value: float = 0.0,
) -> List[Dict[str, Any]]:
    """Replay one trading day across every symbol that has candles that day."""
    entry_start_h, entry_start_m = (int(x) for x in ow.Config.ENTRY_START_TIME.split(":"))
    entry_end_h, entry_end_m = (int(x) for x in ow.Config.ENTRY_END_TIME.split(":"))
    squareoff_h, squareoff_m = (int(x) for x in ow.Config.EOD_SQUAREOFF_TIME.split(":"))

    # Union of every candle timestamp seen today, across symbols, in order.
    all_ts = sorted(set(ts for df in symbol_frames.values() for ts in df["Timestamp"]))
    if not all_ts:
        return []

    engine = ow.StrategyEngine()
    risk = _SimRiskState()
    open_positions: List[Dict[str, Any]] = []
    closed_trades: List[Dict[str, Any]] = []

    for ts in all_ts:
        t = ts.time()
        within_entry_window = (
            (t.hour, t.minute) >= (entry_start_h, entry_start_m)
            and (t.hour, t.minute) <= (entry_end_h, entry_end_m)
        )
        past_squareoff = (t.hour, t.minute) >= (squareoff_h, squareoff_m)

        # --- 0) one regime read for this timestamp, shared by every symbol ---
        if ow.Config.ENABLE_REGIME_ADAPTIVE and nifty_day_df is not None:
            nifty_slice = nifty_day_df[nifty_day_df["Timestamp"] <= ts]
            regime = _compute_regime_at(nifty_slice, daily_range_history, vix_value, symbol_frames, universe_by_symbol, ts)
        else:
            regime = ow.RegimeSnapshot()

        # --- 1) exits first, using this candle's high/low for every symbol with a position ---
        still_open: List[Dict[str, Any]] = []
        for pos in open_positions:
            df = symbol_frames.get(pos["symbol"])
            row = df[df["Timestamp"] == ts] if df is not None else pd.DataFrame()
            if row.empty:
                still_open.append(pos)
                continue
            high, low, close = float(row["High"].iloc[0]), float(row["Low"].iloc[0]), float(row["Close"].iloc[0])
            _update_trailing_stop(pos, close)

            exit_price = None
            exit_reason = None
            # Trending + trailing positions still carry a fixed target as an upside cap -
            # the trailing stop lets a winner run beyond it once armed, but a position that
            # reaches target before the trailing stop has even armed isn't left riding on
            # nothing but EOD square-off.
            if past_squareoff:
                exit_price, exit_reason = close, "EOD_SQUAREOFF"
            elif pos["direction"] == "UP":
                if low <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "STOP_LOSS" if pos["stop_loss"] <= pos["initial_stop_loss"] else "TRAILING_STOP"
                elif high >= pos["target"]:
                    exit_price, exit_reason = pos["target"], "TARGET_1"
            else:
                if high >= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "STOP_LOSS" if pos["stop_loss"] >= pos["initial_stop_loss"] else "TRAILING_STOP"
                elif low <= pos["target"]:
                    exit_price, exit_reason = pos["target"], "TARGET_1"

            if exit_price is None or exit_reason is None:
                still_open.append(pos)
                continue

            trade = _settle_exit(pos, ts, exit_price, exit_reason, risk_amount)
            closed_trades.append(trade)
            risk.record_close(trade["estimated_pnl"])

        open_positions = still_open
        if past_squareoff:
            continue  # no new entries once square-off time has passed

        if ow.Config.ENABLE_REGIME_ADAPTIVE and not regime.allows_new_entries():
            continue  # e.g. LOW_VOLATILITY_CHOP this cycle - manage existing positions only

        # --- 2) signals: feed today's candle into every symbol's state machine ---
        fresh_signals: List[Dict[str, Any]] = []
        for symbol, df in symbol_frames.items():
            day_slice = df[df["Timestamp"] <= ts]
            if day_slice.empty or day_slice["Timestamp"].iloc[-1] != ts:
                continue
            meta = universe_by_symbol[symbol]
            stock = _build_snapshot(symbol, meta["sector"], day_slice, has_real_volume=not meta["is_index"])
            if not stock.orb_complete:
                continue
            if not within_entry_window:
                continue  # live loop only scans for signals inside the entry window
            signal = engine.evaluate_direct_breakout(stock, regime)
            if signal is None:
                continue
            direction = "UP" if signal.trade_type == ow.TradeType.CALL else "DOWN"
            fresh_signals.append({
                "symbol": symbol, "sector": meta["sector"], "direction": direction,
                "entry_time": ts, "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss, "initial_stop_loss": signal.stop_loss, "target": signal.target_1,
                "score": signal.score, "regime": signal.regime.value, "atr_at_entry": signal.atr_at_signal,
                "risk_multiplier": signal.risk_multiplier, "trailing_enabled": signal.trailing_enabled,
            })

        fresh_signals.sort(key=lambda s: s["score"], reverse=True)
        for sig in fresh_signals:
            allowed, reasons = risk.check_trade_allowed(sig["sector"], open_positions)
            if not allowed:
                sig["rejected_reason"] = "; ".join(reasons)
                closed_trades.append({**sig, "exit_time": ts, "exit_price": None, "exit_reason": "RISK_REJECTED",
                                       "underlying_points": 0.0, "r_multiple": 0.0, "estimated_pnl": 0.0})
                continue
            risk.record_open()
            open_positions.append(sig)

    # Anything still open at the very last candle of the day gets closed there.
    if open_positions:
        last_ts = all_ts[-1]
        for pos in open_positions:
            df = symbol_frames.get(pos["symbol"])
            close = float(df[df["Timestamp"] == df["Timestamp"].max()]["Close"].iloc[0]) if df is not None else pos["entry_price"]
            closed_trades.append(_settle_exit(pos, last_ts, close, "EOD_SQUAREOFF", risk_amount))

    return closed_trades


def run_backtest(
    from_date: str,
    to_date: str,
    symbols: Optional[List[Dict[str, Any]]] = None,
    capital: Optional[float] = None,
    risk_per_trade: Optional[float] = None,
    regime_adaptive: Optional[bool] = None,
    client_id: Optional[str] = None,
    access_token: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the backtest over [from_date, to_date] (inclusive, 'YYYY-MM-DD') and return
    (trades_df, summary_df). Does not touch any live paper/risk-state files.

    `regime_adaptive` overrides Config.ENABLE_REGIME_ADAPTIVE for just this run (None =
    use whatever Config is currently set to) - see the module docstring for what that
    layer adds and what it costs in extra data fetches (NIFTY + India VIX candles)."""

    def log(msg: str) -> None:
        ow.logger.info(f"BACKTEST: {msg}")
        if progress_cb:
            progress_cb(msg)

    universe = symbols or default_universe()
    universe_by_symbol = {u["symbol"]: u for u in universe}

    # CAPITAL/RISK_PER_TRADE are resolved to LOCAL values, never written to ow.Config -
    # this backtest computes its own risk_amount/summary directly and never calls
    # TradingEngine.calculate_position_size/execute_signal (the only places that read
    # those two Config fields), so mutating them would serve no purpose here while still
    # being visible to a live/paper TradingSystem running in the same process (e.g. via
    # dashboard.py) - which WOULD read the clobbered values for real position sizing for
    # the full duration of this (potentially long) backtest run.
    resolved_capital = ow.Config.CAPITAL if capital is None else capital
    resolved_risk_pct = ow.Config.RISK_PER_TRADE if risk_per_trade is None else risk_per_trade
    risk_amount = resolved_capital * (resolved_risk_pct / 100)

    old_regime_adaptive = ow.Config.ENABLE_REGIME_ADAPTIVE
    if regime_adaptive is not None:
        ow.Config.ENABLE_REGIME_ADAPTIVE = regime_adaptive

    client_manager = ow.DhanClientManager(
        client_id or ow.Config.DHAN_CLIENT_ID, access_token or ow.Config.DHAN_ACCESS_TOKEN
    )
    client = client_manager.get_client()
    if client is None:
        ow.Config.ENABLE_REGIME_ADAPTIVE = old_regime_adaptive
        raise RuntimeError("Could not connect to Dhan - check DHAN_CLIENT_ID/DHAN_ACCESS_TOKEN before backtesting.")

    instruments = ow.DhanInstrumentMaster(cache_file="security_id_list.csv")

    start_dt = datetime.strptime(from_date, "%Y-%m-%d")
    end_dt = datetime.strptime(to_date, "%Y-%m-%d")

    try:
        # --- fetch every symbol's candles for the whole range up front ---
        all_candles: Dict[str, pd.DataFrame] = {}
        for meta in universe:
            symbol = meta["symbol"]
            resolver = instruments.resolve_index if meta["is_index"] else instruments.resolve_equity
            resolved = resolver(symbol)
            if not resolved:
                log(f"{symbol}: not found in the security master - skipping.")
                continue
            exchange_segment = ow.Config.INDEX_EXCHANGE_SEGMENT if meta["is_index"] else ow.Config.EQUITY_EXCHANGE_SEGMENT
            instrument_type = "INDEX" if meta["is_index"] else "EQUITY"
            log(f"Fetching historical 5-min candles for {symbol} ({from_date} -> {to_date})...")
            df = fetch_historical_candles(
                client, resolved["security_id"], exchange_segment, instrument_type, start_dt, end_dt
            )
            if df.empty:
                log(f"{symbol}: no historical candles returned - skipping.")
                continue
            all_candles[symbol] = df

        if not all_candles:
            raise RuntimeError("No historical candle data could be fetched for any symbol in the universe.")

        # --- regime layer inputs: NIFTY (or Config.REGIME_INDEX_SYMBOL) intraday
        # candles + daily-range history, and India VIX (best-effort) ---
        nifty_intraday, nifty_daily, vix_daily = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if ow.Config.ENABLE_REGIME_ADAPTIVE:
            regime_symbol = ow.Config.REGIME_INDEX_SYMBOL
            if regime_symbol in all_candles:
                nifty_intraday = all_candles[regime_symbol]
            else:
                resolved = instruments.resolve_index(regime_symbol)
                if resolved:
                    log(f"Fetching regime-index ({regime_symbol}) candles for the regime layer...")
                    nifty_intraday = fetch_historical_candles(
                        client, resolved["security_id"], ow.Config.INDEX_EXCHANGE_SEGMENT, "INDEX", start_dt, end_dt
                    )
            resolved = instruments.resolve_index(regime_symbol)
            if resolved:
                nifty_daily = fetch_historical_daily(
                    client, resolved["security_id"], ow.Config.INDEX_EXCHANGE_SEGMENT, "INDEX", start_dt, end_dt
                )
            vix_resolved = instruments.resolve_index(ow.Config.REGIME_VIX_SYMBOL)
            if vix_resolved:
                vix_daily = fetch_historical_daily(
                    client, vix_resolved["security_id"], ow.Config.INDEX_EXCHANGE_SEGMENT, "INDEX", start_dt, end_dt
                )
            else:
                log(f"{ow.Config.REGIME_VIX_SYMBOL} not found in the security master - regime reads will skip the VIX kicker.")

            if nifty_intraday.empty:
                log(
                    f"Could not fetch {regime_symbol} candles - regime falls back to UNKNOWN with neutral "
                    "breadth (0.5) every cycle, which the breadth-confirmation filter rejects in BOTH "
                    "directions (0.5 is below the bullish floor and above the bearish ceiling) - this run "
                    "will likely produce zero trades. Fix the regime-index fetch, or set "
                    "Config.ENABLE_REGIME_ADAPTIVE=False, to get real signals."
                )

        # --- group by calendar day and replay day by day ---
        all_days = sorted(set(ts.date() for df in all_candles.values() for ts in df["Timestamp"]))
        log(f"Replaying {len(all_days)} trading day(s) across {len(all_candles)} symbol(s)...")

        all_trades: List[Dict[str, Any]] = []
        for day in all_days:
            day_frames = {}
            for symbol, df in all_candles.items():
                day_df = df[df["Timestamp"].dt.date == day].reset_index(drop=True)
                if not day_df.empty:
                    day_frames[symbol] = day_df
            if not day_frames:
                continue

            nifty_day_df = None
            daily_range_history = None
            vix_value = 0.0
            if ow.Config.ENABLE_REGIME_ADAPTIVE and not nifty_intraday.empty:
                nifty_day_df = nifty_intraday[nifty_intraday["Timestamp"].dt.date == day].reset_index(drop=True)
                if not nifty_day_df.empty and not nifty_daily.empty:
                    prior_days = nifty_daily[nifty_daily["Timestamp"].dt.date < day]
                    if not prior_days.empty:
                        daily_range_history = ow.compute_true_range(prior_days).tail(ow.Config.REGIME_ATR_PERCENTILE_LOOKBACK_DAYS)
                if not vix_daily.empty:
                    vix_row = vix_daily[vix_daily["Timestamp"].dt.date < day]
                    if not vix_row.empty:
                        vix_value = float(vix_row["Close"].iloc[-1])

            day_trades = _simulate_day(day_frames, universe_by_symbol, risk_amount, nifty_day_df, daily_range_history, vix_value)
            all_trades.extend(day_trades)

        trades_df = pd.DataFrame(all_trades)
        if not trades_df.empty:
            trades_df["trade_date"] = pd.to_datetime(trades_df["entry_time"]).dt.date
            trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
            wanted_cols = [
                "trade_date", "symbol", "sector", "direction", "regime", "entry_time", "entry_price",
                "initial_stop_loss", "stop_loss", "target", "trailing_enabled", "risk_multiplier",
                "exit_time", "exit_price", "exit_reason",
                "underlying_points", "r_multiple", "estimated_pnl", "score", "rejected_reason",
            ]
            trades_df = trades_df[[c for c in wanted_cols if c in trades_df.columns]]
            trades_df = trades_df.rename(columns={"stop_loss": "final_stop_loss"})

        summary_df = _build_summary(trades_df, from_date, to_date, resolved_capital, resolved_risk_pct)
        log(f"Done - {len(trades_df)} trade record(s) (including risk-rejected signals).")
        return trades_df, summary_df
    finally:
        ow.Config.ENABLE_REGIME_ADAPTIVE = old_regime_adaptive


def _build_summary(trades_df: pd.DataFrame, from_date: str, to_date: str, capital: float, risk_pct: float) -> pd.DataFrame:
    executed = trades_df[trades_df["exit_reason"] != "RISK_REJECTED"] if not trades_df.empty else trades_df
    rejected_count = int((trades_df["exit_reason"] == "RISK_REJECTED").sum()) if not trades_df.empty else 0

    if executed.empty:
        return pd.DataFrame([{
            "from_date": from_date, "to_date": to_date, "capital": capital, "risk_per_trade_pct": risk_pct,
            "total_trades": 0, "signals_risk_rejected": rejected_count,
        }])

    wins = executed[executed["estimated_pnl"] > 0]
    losses = executed[executed["estimated_pnl"] <= 0]
    total_pnl = executed["estimated_pnl"].sum()
    gross_win = wins["estimated_pnl"].sum()
    gross_loss = -losses["estimated_pnl"].sum()

    equity = capital + executed.sort_values("exit_time")["estimated_pnl"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    by_exit_reason = executed["exit_reason"].value_counts().to_dict()

    return pd.DataFrame([{
        "from_date": from_date, "to_date": to_date, "capital": capital, "risk_per_trade_pct": risk_pct,
        "total_trades": len(executed), "signals_risk_rejected": rejected_count,
        "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(executed) * 100, 2),
        "total_estimated_pnl": round(total_pnl, 2),
        "final_equity_estimate": round(capital + total_pnl, 2),
        "avg_win": round(wins["estimated_pnl"].mean(), 2) if not wins.empty else 0.0,
        "avg_loss": round(losses["estimated_pnl"].mean(), 2) if not losses.empty else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "avg_r_multiple": round(executed["r_multiple"].mean(), 3),
        "max_drawdown_estimate": round(max_drawdown, 2),
        "stop_loss_exits": by_exit_reason.get("STOP_LOSS", 0),
        "trailing_stop_exits": by_exit_reason.get("TRAILING_STOP", 0),
        "target_exits": by_exit_reason.get("TARGET_1", 0),
        "eod_squareoff_exits": by_exit_reason.get("EOD_SQUAREOFF", 0),
        "regime_adaptive_enabled": ow.Config.ENABLE_REGIME_ADAPTIVE,
    }])


def save_backtest_csvs(
    trades_df: pd.DataFrame, summary_df: pd.DataFrame,
    trades_path: str = TRADES_CSV, results_path: str = RESULTS_CSV,
) -> None:
    trades_df.to_csv(trades_path, index=False)
    summary_df.to_csv(results_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the ORB direct-breakout strategy over historical Dhan data.")
    parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--risk-pct", type=float, default=None)
    parser.add_argument("--no-regime-adaptive", action="store_true",
                         help="Disable the regime-adaptive layer for this run (original deterministic rule instead).")
    parser.add_argument("--trades-csv", default=TRADES_CSV)
    parser.add_argument("--results-csv", default=RESULTS_CSV)
    args = parser.parse_args()

    trades_df, summary_df = run_backtest(
        args.from_date, args.to_date, capital=args.capital, risk_per_trade=args.risk_pct,
        regime_adaptive=(False if args.no_regime_adaptive else None),
        progress_cb=print,
    )
    save_backtest_csvs(trades_df, summary_df, args.trades_csv, args.results_csv)
    print(f"\nWrote {len(trades_df)} trade row(s) to {args.trades_csv}")
    print(f"Wrote summary to {args.results_csv}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
