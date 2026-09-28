"""
NSE Sector Heatmap Options Trading System - Hardened Engine (Phase 1)
======================================================================

This file is organized into clearly separated conceptual modules within a
single file (see section banners below). A physical multi-file package
split is a safe, mechanical follow-up once this logic has been validated
against live data - deliberately NOT done in this pass, to avoid adding
import-wiring risk on top of the logic changes.

WHAT THIS PHASE FIXES (see PHASE1_CHANGES.md-equivalent summary at the
bottom of this docstring, and the chat response, for the full audit):

  1. Sector classification now distinguishes POSITIVE / NEGATIVE / NEUTRAL
     by actual sign + threshold, not "top 3 / bottom 3 regardless of sign".
  2. Daily stock return now uses the previous TRADING DAY's close, not the
     previous 5-minute candle's close.
  3. Breakout and pullback are now a real, persisted state machine per
     symbol: a breakout requires a confirmed candle close beyond the ORB
     level (with a small noise buffer), the price must then actually move
     away, and a pullback requires it to return toward that level and hold
     - "price is currently beyond ORB_HIGH" is no longer treated as a
     pullback by itself.
  4. A confirmed breakout+pullback sequence is consumed exactly once - the
     same setup does not regenerate a fresh signal on every 60-second scan.
  5. T1/T2/T3 now close partial quantities (configurable split, lot-size
     aware, with an explicit single-lot fallback) instead of exiting the
     whole position at T1.
  6. Option selection now evaluates ATM / ATM+1 / ATM-1 using Dhan's real
     option chain quote data (bid/ask/volume/OI), rejects illiquid/wide-
     spread/mispriced contracts with a logged reason, and only falls back
     off ATM when ATM itself fails those checks.
  7. LIVE orders are only marked OPEN after polling Dhan's actual order
     status and seeing a filled state - an order ID alone is no longer
     treated as a filled position.
  8. A RiskManager gate (max open positions, max trades/day, max loss/day,
     max positions per sector) runs before every trade, with logged
     rejection reasons.
  9. EOD square-off closes all open intraday positions at a configurable
     time instead of leaving them to be discovered on the next restart.
 10. All "today" / "market hours" logic uses IST computed from UTC, never
     the host machine's local clock.

WHAT IS DEFERRED (needs a live, funded Dhan account to build/verify safely
- see the chat response for the full list): true broker-position
reconciliation on restart, an NSE trading-holiday calendar (currently
weekday-only), a fast/medium/slow multi-threaded loop split (currently a
single loop with different scan cadences), websocket feeds, the 10-module
package split, and a monitoring dashboard.

Install:
    pip install dhanhq requests pandas
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timezone, timedelta, date as date_cls
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# MODULE: market / IST time helpers
# ============================================
# NSE trades 09:15-15:30 IST regardless of what timezone the host machine
# is set to. Every time comparison here is computed explicitly from UTC.

IST_OFFSET = timedelta(hours=5, minutes=30)


def ist_now() -> datetime:
    """Current wall-clock time in India (IST), as a naive datetime."""
    return (datetime.now(timezone.utc) + IST_OFFSET).replace(tzinfo=None)


def is_trading_day(now: Optional[datetime] = None) -> bool:
    """
    True if `now` (IST) is a weekday. LIMITATION: this does not know about
    NSE trading holidays (Diwali, Republic Day, etc.) - a real holiday
    calendar needs an external data source (e.g. NSE's holiday API or a
    maintained holiday list) and is deliberately deferred; see module
    docstring. Treat 'is_trading_day() is True' as necessary but not
    sufficient.
    """
    now = now or ist_now()
    return now.weekday() < 5  # Monday=0 ... Friday=4


def is_market_open(now: Optional[datetime] = None) -> bool:
    """True if `now` (IST) falls within NSE's 09:15-15:30 trading window on a trading day."""
    now = now or ist_now()
    if not is_trading_day(now):
        return False
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t


def is_pre_market(now: Optional[datetime] = None) -> bool:
    """True if `now` (IST) is before market open on a trading day."""
    now = now or ist_now()
    if not is_trading_day(now):
        return False
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    return now < open_t


def is_orb_complete(now: Optional[datetime] = None) -> bool:
    """True once the opening-range window (Config.ORB_START-Config.ORB_END) has fully elapsed."""
    now = now or ist_now()
    end_h, end_m = (int(x) for x in Config.ORB_END_TIME.split(':'))
    orb_end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    return now >= orb_end


def is_entry_window(now: Optional[datetime] = None) -> bool:
    """True if `now` (IST) is within the configured window for opening NEW positions."""
    now = now or ist_now()
    if not is_market_open(now):
        return False
    start_h, start_m = (int(x) for x in Config.ENTRY_START_TIME.split(':'))
    end_h, end_m = (int(x) for x in Config.ENTRY_END_TIME.split(':'))
    start_t = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end_t = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    return start_t <= now <= end_t


def is_squareoff_window(now: Optional[datetime] = None) -> bool:
    """True once it's time to force-close all open intraday positions for the day."""
    now = now or ist_now()
    sq_h, sq_m = (int(x) for x in Config.EOD_SQUAREOFF_TIME.split(':'))
    squareoff_t = now.replace(hour=sq_h, minute=sq_m, second=0, microsecond=0)
    return now >= squareoff_t


# ============================================
# MODULE: config
# ============================================

@dataclass
class Config:
    """All strategy/risk/execution parameters live here - no magic numbers scattered in logic."""

    TRADING_MODE: str = "PAPER"  # PAPER or LIVE

    # --- Safety gate: even mode="LIVE" refuses to place real orders unless
    # this is explicitly set true (env var LIVE_TRADING_ENABLED=true) ---
    LIVE_TRADING_ENABLED: bool = os.getenv('LIVE_TRADING_ENABLED', 'false').strip().lower() == 'true'

    # --- Sector universe ---
    SECTOR_INDICES: ClassVar[List[str]] = [
        'NIFTY BANK', 'NIFTY IT', 'NIFTY PHARMA', 'NIFTY AUTO', 'NIFTY FMCG',
        'NIFTY METAL', 'NIFTY REALTY', 'NIFTY ENERGY', 'NIFTY MEDIA',
        'NIFTY FIN SERVICE', 'NIFTY COMMODITIES', 'NIFTY PSU BANK'
    ]
    SECTOR_STOCKS: ClassVar[Dict[str, List[str]]] = {
        'NIFTY BANK': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN'],
        'NIFTY IT': ['TCS', 'PERSISTENT', 'OFSS', 'COFORGE', 'LTM', 'MPHASIS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM'],
        'NIFTY PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'TORNTPHARM'],
        'NIFTY AUTO': ['TMPV', 'TMCV', 'MARUTI', 'M&M', 'HEROMOTOCO', 'EICHERMOT', 'BAJAJ-AUTO', 'ASHOKLEY'],
        'NIFTY FMCG': ['ITC', 'HINDUNILVR', 'NESTLEIND', 'BRITANNIA', 'TATACONSUM'],
        'NIFTY METAL': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL', 'COALINDIA', 'SAIL', 'NMDC', 'HINDCOPPER', 'JINDALSTEL', 'WELCORP'],
        'NIFTY REALTY': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PHOENIXLTD', 'BRIGADE', 'PRESTIGE', 'LODHA'],
        'NIFTY ENERGY': ['RELIANCE', 'ONGC', 'POWERGRID', 'NTPC', 'ADANIGREEN'],
        'NIFTY MEDIA': ['SUNTV', 'ZEEL', 'PVRINOX', 'SAREGAMA', 'DBCORP', 'HATHWAY', 'NETWORK18'],
        'NIFTY FIN SERVICE': ['BAJFINANCE', 'BAJAJFINSV', 'SBIN', 'ICICIPRULI'],
        'NIFTY COMMODITIES': ['HINDALCO', 'VEDL', 'TATASTEEL', 'JSWSTEEL'],
        'NIFTY PSU BANK': ['SBIN', 'BANKBARODA', 'PNB', 'CANBK', 'UNIONBANK', 'INDIANB', 'IOB', 'PSB', 'BANKINDIA', 'UCOBANK', 'CENTRALBK']
    }
    TOP_SECTORS: int = 3
    STOCKS_PER_SECTOR: int = 3

    # --- Indices traded independently of the sector-heatmap funnel, each running its
    # own direct-breakout ORB sequence (security master trading symbols) ---
    TRADING_INDICES: ClassVar[List[str]] = ["NIFTY", "BANKNIFTY"]

    # --- Sector classification thresholds (item A) ---
    # A sector below MIN_POSITIVE and above -MIN_NEGATIVE is NEUTRAL and ignored.
    SECTOR_MIN_POSITIVE_RETURN: float = 0.05   # % - must be at least this positive
    SECTOR_MIN_NEGATIVE_RETURN: float = 0.05   # % - must be at least this negative (magnitude)

    # --- ORB window ---
    ORB_START_TIME: str = "09:15"
    ORB_END_TIME: str = "09:30"   # was "09:20" - didn't match ORB_MINUTES=15 (9:15+15min=9:30); is_orb_complete() gated on this
    ORB_MINUTES: int = 15
    CANDLE_TIMEFRAME: str = "5"
    MIN_ORB_RANGE: float = 0.1      # % - reject ORBs narrower than this as unreliable
    MAX_ORB_RANGE: float = 5.0      # % - reject ORBs wider than this as abnormal/bad-data

    # --- Entry / square-off windows ---
    ENTRY_START_TIME: str = "09:30"   # no entries before ORB is complete
    ENTRY_END_TIME: str = "14:30"     # no NEW entries after this - hard cutoff per spec
    EOD_SQUAREOFF_TIME: str = "15:00" # force-close all open intraday positions at/after this time

    # --- Direct-breakout entry rule (close beyond OR + beyond VWAP + volume confirmation;
    # enters on the NEXT completed candle, no pullback wait) ---
    VOLUME_CONFIRMATION_MULTIPLE: float = 1.3   # last candle's volume must be >= this x the prior VOLUME_LOOKBACK_CANDLES average
    VOLUME_LOOKBACK_CANDLES: int = 20           # rolling window for the volume-confirmation average
    MAX_TRADES_PER_SYMBOL_PER_DAY: int = 1      # one shot per symbol per day; no re-entry after that symbol's SL or TP

    # --- Signal scoring ---
    # No score-based gating in the direct-breakout rule (it's a deterministic pass/fail) -
    # MIN_SIGNAL_SCORE is kept at 0 so score is used only to SORT same-cycle signals, never to reject one.
    MIN_SIGNAL_SCORE: float = 0.0

    # --- Feature flags: initially OFF so paper/live/backtest measure the clean baseline
    # edge with no confounding filters. None of these filters exist in the engine yet -
    # these are documented placeholders for future extension. ---
    ENABLE_GAP_FILTER: bool = False
    ENABLE_ATR_FILTER: bool = False
    ENABLE_VIX_FILTER: bool = False
    ENABLE_TRAILING_STOP: bool = False
    ENABLE_BREAKEVEN_STOP: bool = False

    # --- Options selection (item G) ---
    EQUITY_EXCHANGE_SEGMENT: str = "NSE_EQ"       # underlying equity's dhanhq exchange segment
    INDEX_EXCHANGE_SEGMENT: str = "IDX_I"         # underlying index's dhanhq exchange segment (NIFTY/BANKNIFTY spot)
    DERIVATIVE_EXCHANGE_SEGMENT: str = "NSE_FNO"  # segment options actually trade/settle on
    OPTION_PRODUCT_TYPE: str = "INTRADAY"         # dhanhq product_type for option orders
    TARGET_DTE: Optional[int] = None  # None = nearest expiry >= MIN_DAYS_TO_EXPIRY (default); else the valid expiry closest to this many days out (0=0DTE, 1=1DTE, 2/3=2-3DTE, etc.) - makes expiry configurable for testing
    MIN_DAYS_TO_EXPIRY: int = 1  # nearest expiry must be at least this many days out - never trade today's expiry
    OPTION_CANDIDATE_OFFSETS: ClassVar[List[str]] = ["ATM", "ATM+1", "ATM-1"]
    MIN_OPTION_VOLUME: int = 0        # set >0 once real volume field is confirmed against a live account (see report)
    MIN_OPTION_OI: int = 0            # set >0 once real OI field is confirmed against a live account
    MAX_OPTION_SPREAD_PERCENT: float = 15.0
    MIN_OPTION_PREMIUM: float = 1.0
    MAX_OPTION_PREMIUM: float = 1000.0
    FALLBACK_LOT_SIZE: int = 50

    # --- Instrument master (authoritative security_id / lot_size / strikes) ---
    INSTRUMENT_MASTER_CACHE_FILE: str = "Dependencies/dhan_instrument_master_cache.csv"
    INSTRUMENT_MASTER_MODE: str = "compact"

    # --- Risk management (item L) ---
    CAPITAL: float = 100000.0
    RISK_PER_TRADE: float = 3.0
    MAX_OPEN_POSITIONS: int = 3
    MAX_POSITIONS_PER_SECTOR: int = 2
    MAX_DAILY_TRADES: int = 5
    MAX_DAILY_LOSS: float = 2000.0      # rupees - 2% of CAPITAL; stop opening new trades once realized loss reaches this
    CONSECUTIVE_LOSS_HALT: int = 10      # stop opening new trades after this many consecutive losing closes
    OPTION_EMERGENCY_STOP_PERCENT: float = 30.0  # secondary/safety-net stop: exit if premium drops this % from entry regardless of underlying
    PREMIUM_NOTIONAL_CAP_MULTIPLE: float = 3.0   # total premium outlay (capital_allocated) for one position may not exceed this multiple of that trade's risk budget

    # --- Target (item M) ---
    TAKE_PROFIT_1: float = 2.0   # 1:2 reward:risk on the UNDERLYING - the only target; full exit, no partials
    TAKE_PROFIT_2: float = 2.0   # unused by the new underlying-target exit path; kept only for legacy positions still on the old premium ladder
    TAKE_PROFIT_3: float = 2.0
    T1_EXIT_PERCENT: float = 33.0
    T2_EXIT_PERCENT: float = 33.0
    T3_EXIT_PERCENT: float = 34.0
    # If the position is only 1 lot, a partial exit would create an invalid
    # fractional lot - fall back to this policy instead:
    SINGLE_LOT_FALLBACK: str = "FULL_AT_T1"  # "FULL_AT_T1" | "HOLD_TO_T2" | "HOLD_TO_T3"

    # --- Order execution ---
    ORDER_FILL_TIMEOUT_SECONDS: int = 20
    ORDER_STATUS_POLL_INTERVAL_SECONDS: float = 2.0
    # Dhan order status strings observed as "filled" - see report for the
    # caveat that this needs verifying against a real filled order.
    FILLED_ORDER_STATUSES: ClassVar[List[str]] = ["TRADED", "FILLED", "EXECUTED", "COMPLETE"]
    REJECTED_ORDER_STATUSES: ClassVar[List[str]] = ["REJECTED", "CANCELLED", "EXPIRED"]

    # --- Loop cadence ---
    POSITION_MONITOR_INTERVAL_SECONDS: int = 15
    SIGNAL_SCAN_INTERVAL_SECONDS: int = 60

    # --- Dhan credentials (dhanhq only supports access-token auth) ---
    DHAN_CLIENT_ID: str = os.getenv('DHAN_CLIENT_ID', 'your_client_id')
    DHAN_ACCESS_TOKEN: str = os.getenv('DHAN_ACCESS_TOKEN', 'your_access_token')

    # --- NSE website ---
    NSE_BASE_URL: str = "https://www.nseindia.com"
    NSE_API_URL: str = "https://www.nseindia.com/api"

    # --- Persistence ---
    POSITIONS_FILE: str = "positions.json"
    RISK_STATE_FILE: str = "daily_risk_state.json"
    PAPER_TRADES_FILE: str = "paper_trades.csv"
    LIVE_TRADES_FILE: str = "live_trades.csv"

    # ============================================
    # Regime-adaptive momentum/breakout layer
    # ============================================
    # Feature flag - False reproduces the exact pre-existing direct-breakout
    # behaviour (single displacement candle fires the very next candle, fixed
    # 1:2R target, structural-only stop). True enables the regime filter,
    # displacement->pullback entry, dynamic ATR stop, and regime-conditional
    # target described in the architecture doc. Kept togglable so a bad regime
    # read never silently changes behaviour without an explicit opt-in.
    ENABLE_REGIME_ADAPTIVE: bool = True

    # --- Regime classification inputs ---
    REGIME_INDEX_SYMBOL: str = "NIFTY"          # the index whose intraday candles drive the regime read
    REGIME_VIX_SYMBOL: str = "INDIA VIX"
    REGIME_ADX_PERIOD: int = 14
    REGIME_ATR_PERIOD: int = 14
    REGIME_ATR_PERCENTILE_LOOKBACK_DAYS: int = 20   # trailing daily-ATR history used to rank today's ATR
    REGIME_TRENDING_ADX_MIN: float = 20.0
    REGIME_CHOP_ADX_MAX: float = 15.0
    REGIME_HIGH_VOL_ATR_PERCENTILE: float = 80.0    # today's ATR at/above this percentile of the trailing history -> HIGH_VOLATILITY
    REGIME_LOW_VOL_ATR_PERCENTILE: float = 20.0     # combined with low ADX -> LOW_VOLATILITY_CHOP
    REGIME_VIX_HIGH: float = 20.0                   # India VIX at/above this reinforces a HIGH_VOLATILITY read
    REGIME_BREADTH_BULLISH_MIN: float = 0.55        # fraction of the scanned universe green -> bullish breadth
    REGIME_BREADTH_BEARISH_MAX: float = 0.45        # fraction green at/below this -> bearish breadth

    # --- What each regime is allowed to trade (item: "don't trade the same strategy every day") ---
    REGIME_ALLOWS_ENTRIES: ClassVar[Dict[str, bool]] = {
        "TRENDING_BULL": True, "TRENDING_BEAR": True, "RANGE_BOUND": True,
        "HIGH_VOLATILITY": True, "LOW_VOLATILITY_CHOP": False,
    }
    # Directional gate: which TradeType(s) a regime accepts. Empty/omitted = both directions.
    REGIME_ALLOWED_DIRECTIONS: ClassVar[Dict[str, List[str]]] = {
        "TRENDING_BULL": ["CALL"], "TRENDING_BEAR": ["PUT"],
    }
    REGIME_TARGET_R: ClassVar[Dict[str, float]] = {
        "TRENDING_BULL": 2.0, "TRENDING_BEAR": 2.0, "RANGE_BOUND": 1.5, "HIGH_VOLATILITY": 2.5,
    }
    REGIME_RISK_MULTIPLIER: ClassVar[Dict[str, float]] = {
        "HIGH_VOLATILITY": 0.5,  # smaller size in a high-vol regime, per the architecture doc
    }
    REGIME_TRAIL_ENABLED: ClassVar[Dict[str, bool]] = {"TRENDING_BULL": True, "TRENDING_BEAR": True}
    REGIME_TRAIL_TRIGGER_R: float = 1.0     # start trailing once price has moved this many R in favor
    REGIME_TRAIL_ATR_MULTIPLE: float = 1.0  # trail distance behind price, in units of ATR

    # --- Dynamic (ATR-aware) structural stop ---
    ATR_STOP_MULTIPLIER: float = 0.8   # SL = entry -/+ max(structural distance, this x ATR)

    # --- Displacement -> pullback entry (replaces "fires on the very next candle") ---
    PULLBACK_MAX_CANDLES: int = 6              # candles allowed between displacement and a confirmed pullback before the setup expires
    PULLBACK_ZONE_BUFFER_PERCENT: float = 0.15  # how far price may probe past the OR/VWAP zone and still count as "holding"

    # --- Order-flow confirmation proxy (no L2/tick feed available - approximated
    # from where the confirmation candle closed within its own range) ---
    ENABLE_ORDER_FLOW_FILTER: bool = True
    ORDER_FLOW_MIN_UP_RATIO: float = 0.55   # (close-low)/(high-low) must be >= this for a CALL confirmation candle

    # --- Market-breadth confirmation (fraction of the scanned universe trading green) ---
    ENABLE_BREADTH_FILTER: bool = True


# ============================================
# MODULE: enums
# ============================================

class TradeStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class TradeType(Enum):
    """Which option is bought: CALL means a CE was bought, PUT means a PE was bought."""
    CALL = "CALL"
    PUT = "PUT"


class ExitReason(Enum):
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    TARGET_1 = "TARGET_1"
    TARGET_2 = "TARGET_2"
    TARGET_3 = "TARGET_3"
    EOD_SQUAREOFF = "EOD_SQUAREOFF"
    EXPIRED = "EXPIRED"
    MANUAL = "MANUAL"
    NONE = ""


class SectorDirection(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class BreakoutPhase(Enum):
    """
    Per-symbol breakout state.

    With Config.ENABLE_REGIME_ADAPTIVE=False this is the original deterministic
    no-pullback-wait sequence: WAITING_FOR_BREAKOUT -> ARMED (fires next candle) -> FIRED.

    With it True, ARMED means "displaced" (a candle closed beyond OR+VWAP+volume) and
    the sequence now waits for price to pull back toward that zone and hold before a
    confirmation candle fires the signal - see StrategyEngine._check_pullback.
    """
    WAITING_FOR_BREAKOUT = "WAITING_FOR_BREAKOUT"
    ARMED = "ARMED"           # displaced: a completed candle satisfied the entry rule
    PULLBACK_TOUCHED = "PULLBACK_TOUCHED"  # price has pulled back into the breakout zone - waiting for a confirmation candle
    FIRED = "FIRED"           # signal already generated today - terminal, no re-entry
    EXPIRED = "EXPIRED"       # displaced but the pullback never confirmed in time, or structure was invalidated - terminal, no re-entry


class OrderState(Enum):
    """Order lifecycle - a position is only OPEN after FILLED is confirmed."""
    SIGNAL = "SIGNAL"
    ORDER_INTENT = "ORDER_INTENT"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_PENDING = "ORDER_PENDING"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_TIMEOUT = "ORDER_TIMEOUT"


class MarketRegime(Enum):
    """Whole-session regime read (see RegimeClassifier) - "don't trade the same
    strategy every day": this gates whether new entries are allowed at all, which
    direction(s) are accepted, the target R-multiple, trailing behaviour, and the
    position-size multiplier for that cycle."""
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGE_BOUND = "RANGE_BOUND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY_CHOP = "LOW_VOLATILITY_CHOP"
    UNKNOWN = "UNKNOWN"  # not enough data yet to classify - treated as conservatively as RANGE_BOUND


# ============================================
# MODULE: data models
# ============================================

@dataclass
class Stock:
    symbol: str
    name: str
    sector: str
    current_price: float
    change_percent: float   # vs previous TRADING DAY close - informational only, not used by the direct-breakout rule
    volume: int              # last completed candle's volume
    orb_high: float = 0.0
    orb_low: float = 0.0
    orb_range: float = 0.0
    orb_complete: bool = False

    # --- direct-breakout entry rule inputs ---
    vwap: float = 0.0            # today's volume-weighted average price so far
    avg_volume_20: float = 0.0   # rolling mean of the PRIOR Config.VOLUME_LOOKBACK_CANDLES candles (excludes the current one)
    candle_count: int = 0        # number of completed candles seen today so far - lets the engine detect a new candle has arrived
    has_real_volume: bool = True  # False for spot indices (no real traded volume) - the volume-confirmation check is skipped when False

    # --- regime-adaptive inputs (Config.ENABLE_REGIME_ADAPTIVE) ---
    current_high: float = 0.0    # last completed candle's high - needed for the order-flow proxy and pullback checks
    current_low: float = 0.0     # last completed candle's low
    current_open: float = 0.0    # last completed candle's open - confirmation-candle direction check
    atr: float = 0.0             # intraday ATR (Wilder TR mean) over candles seen so far today - dynamic stop sizing


@dataclass
class SectorRanking:
    sector: str
    change_percent: float
    direction: SectorDirection
    rank: int
    timestamp: datetime


@dataclass
class RegimeSnapshot:
    """One whole-session regime read - see RegimeClassifier. Recomputed each scan
    cycle from the regime index's intraday candles + India VIX + scanned-universe breadth."""
    regime: MarketRegime = MarketRegime.UNKNOWN
    adx: float = 0.0
    atr: float = 0.0
    atr_percentile: float = 50.0
    vwap_distance_percent: float = 0.0
    india_vix: float = 0.0
    opening_gap_percent: float = 0.0
    breadth_ratio: float = 0.5   # fraction of the scanned universe trading green
    timestamp: Optional[datetime] = None

    def allows_new_entries(self) -> bool:
        return Config.REGIME_ALLOWS_ENTRIES.get(self.regime.value, True)

    def allows_direction(self, trade_type: "TradeType") -> bool:
        allowed = Config.REGIME_ALLOWED_DIRECTIONS.get(self.regime.value)
        return allowed is None or trade_type.value in allowed

    def target_r(self) -> float:
        return Config.REGIME_TARGET_R.get(self.regime.value, Config.TAKE_PROFIT_1)

    def risk_multiplier(self) -> float:
        return Config.REGIME_RISK_MULTIPLIER.get(self.regime.value, 1.0)

    def trailing_enabled(self) -> bool:
        return Config.REGIME_TRAIL_ENABLED.get(self.regime.value, False)


@dataclass
class BreakoutState:
    """Persisted per symbol for the trading day - see BreakoutPhase. One shot per
    symbol per day: once FIRED or EXPIRED, no further signals for that symbol today."""
    symbol: str
    trade_date: date_cls
    orb_high: float
    orb_low: float
    phase: BreakoutPhase = BreakoutPhase.WAITING_FOR_BREAKOUT
    direction: Optional[str] = None  # 'UP' or 'DOWN', set once armed/displaced
    armed_candle_count: int = 0      # stock.candle_count at the moment the rule was satisfied - fires the very next candle (regime-adaptive OFF)
    breakout_price: float = 0.0      # the displacement candle's close
    breakout_time: Optional[datetime] = None
    signal_id: str = ""

    # --- displacement -> pullback tracking (Config.ENABLE_REGIME_ADAPTIVE) ---
    displacement_extreme: float = 0.0   # furthest price reached since displacement, in the breakout direction
    pullback_zone: float = 0.0          # the level price must pull back toward (max(orb_high, vwap) for UP / min(orb_low, vwap) for DOWN)
    candles_since_displacement: int = 0


@dataclass
class OptionCandidate:
    """One evaluated strike/type option, with the liquidity data used to accept/reject it."""
    option_symbol: str
    option_type: str
    strike_label: str  # 'ATM', 'ATM+1', 'ATM-1'
    strike_price: float = 0.0
    ltp: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread_percent: Optional[float] = None
    volume: int = 0
    oi: int = 0
    lot_size: int = 0
    security_id: str = ""
    passes_filters: bool = False
    rejection_reason: str = ""


@dataclass
class OptionContract:
    """The final selected, tradable option contract."""
    option_symbol: str
    underlying_symbol: str
    option_type: str
    strike_price: float
    expiry_date: str
    lot_size: int
    premium: float
    security_id: str = ""
    selection_reason: str = ""
    distance_from_atm: str = "ATM"
    bid: float = 0.0
    ask: float = 0.0
    spread_percent: Optional[float] = None
    volume: int = 0
    oi: int = 0


@dataclass
class TradeSignal:
    """
    A directional signal generated from a CONFIRMED breakout + pullback on
    the underlying stock. entry_price/stop_loss/target_1-3 here describe
    the STOCK's price levels at signal time (context for why the trade was
    taken, and used as the primary underlying-structure invalidation level)
    - NOT what the option position is managed against; see Position for the
    option-premium-based levels used for the emergency stop.
    """
    signal_id: str
    stock: Stock
    trade_type: TradeType
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    score: float
    score_breakdown: Dict[str, float]
    breakout_price: float = 0.0
    breakout_time: Optional[datetime] = None
    pullback_price: float = 0.0
    vwap_at_signal: float = 0.0
    relative_volume: float = 0.0  # last candle's volume / avg_volume_20 at signal time (0.0 for indices - no real volume)

    # --- regime-adaptive context (Config.ENABLE_REGIME_ADAPTIVE) ---
    regime: MarketRegime = MarketRegime.UNKNOWN
    atr_at_signal: float = 0.0
    risk_multiplier: float = 1.0   # from RegimeSnapshot.risk_multiplier() - scales position size for this trade
    trailing_enabled: bool = False


@dataclass
class PartialExit:
    quantity: int
    exit_price: float
    exit_time: datetime
    exit_reason: str
    pnl: float


@dataclass
class Position:
    stock_symbol: str  # holds the OPTION's trading symbol (what's actually bought/sold/monitored)
    trade_type: TradeType
    entry_price: float  # option premium at entry (requested)
    quantity: int        # original total quantity
    stop_loss: float     # option premium emergency-stop level
    target_1: float
    target_2: float
    target_3: float
    entry_time: datetime
    status: TradeStatus = TradeStatus.PENDING
    order_state: str = OrderState.SIGNAL.value

    # Underlying-structure primary stop (item: primary stop vs emergency stop)
    underlying_stop_loss: float = 0.0
    underlying_target: float = 0.0  # 1:2 RR target measured on the UNDERLYING - 0.0 for legacy positions predating this field, which stay on the old premium T1/T2/T3 ladder

    remaining_quantity: int = 0
    partial_exits: List[PartialExit] = field(default_factory=list)
    exit_price: float = 0.0     # last/final exit price (for CSV convenience)
    exit_time: Optional[datetime] = None
    exit_reason: str = ExitReason.NONE.value
    pnl: float = 0.0            # total realized P&L across all partial exits
    order_id: Optional[str] = None
    filled_price: float = 0.0
    filled_quantity: int = 0

    underlying_symbol: str = ""
    option_type: str = ""
    strike_price: float = 0.0
    expiry_date: str = ""
    lot_size: int = 0
    security_id: str = ""  # the option contract's real Dhan security_id - required for live order placement
    underlying_price_at_entry: float = 0.0

    signal_id: str = ""
    sector: str = ""
    orb_high: float = 0.0
    orb_low: float = 0.0
    orb_range_percent: float = 0.0
    breakout_price: float = 0.0
    pullback_level: float = 0.0
    vwap_at_signal: float = 0.0
    relative_volume: float = 0.0
    score: float = 0.0
    capital_allocated: float = 0.0
    risk_amount: float = 0.0
    risk_per_share: float = 0.0

    t1_hit: bool = False
    t2_hit: bool = False
    t3_hit: bool = False

    # --- regime-adaptive fields (Config.ENABLE_REGIME_ADAPTIVE) ---
    regime_at_entry: str = MarketRegime.UNKNOWN.value
    atr_at_entry: float = 0.0
    trailing_enabled: bool = False
    trailing_stop_underlying: float = 0.0  # 0.0 = trailing not yet armed; once armed, tracks underlying_stop_loss upward/downward


# ============================================
# MODULE: dhanhq (official SDK) import
# ============================================

DHANHQ_AVAILABLE: bool = False
DhanContext_class: Optional[Any] = None
dhanhq_class: Optional[Any] = None

try:
    from dhanhq import DhanContext as _DhanContext, dhanhq as _dhanhq

    DhanContext_class = _DhanContext
    dhanhq_class = _dhanhq
    DHANHQ_AVAILABLE = True
    logger.info("dhanhq SDK found and imported - live data and orders are available.")
except ImportError:
    logger.warning(
        "dhanhq is not installed, so no chart data or live orders can be fetched/placed. "
        "Install it with: pip install dhanhq"
    )


# ============================================
# MODULE: execution / Dhan client manager
# ============================================

class DhanClientManager:
    """Owns a single official dhanhq SDK client connection, shared across data reads and order placement."""

    def __init__(self, client_id: str, access_token: str) -> None:
        self.client_id = client_id
        self.access_token = access_token
        self.client: Optional[Any] = None
        self._connect()

    def _connect(self) -> None:
        if not DHANHQ_AVAILABLE or dhanhq_class is None or DhanContext_class is None:
            logger.error("Cannot connect to Dhan - the dhanhq SDK isn't installed.")
            return

        logger.info("Logging in to Dhan via the dhanhq SDK...")
        try:
            context = DhanContext_class(self.client_id, self.access_token)
            candidate = dhanhq_class(context)

            # DhanContext/dhanhq's own __init__ doesn't validate the token - a bad
            # client_id/access_token still returns a usable-looking object. Confirm
            # the login actually works with a real authenticated call.
            check = candidate.get_fund_limits()
            if isinstance(check, dict) and check.get('status') == 'success':
                self.client = candidate
                logger.info("Connected to Dhan successfully - ready to fetch data and place orders.")
            else:
                remarks = check.get('remarks') if isinstance(check, dict) else check
                logger.error(
                    f"Dhan login did not succeed ({remarks}) - chart data and live orders will not work "
                    "until this is fixed (check DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN)."
                )
                self.client = None
        except Exception as e:
            logger.error(f"Could not log in to Dhan: {e}")
            self.client = None

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    def get_client(self) -> Optional[Any]:
        """Returns the connected dhanhq client, or None - bind to a local var and check for
        None (as every caller does) so type checkers can narrow it properly."""
        return self.client


# ============================================
# MODULE: market data - NSE heatmap
# ============================================

class NSEDataFetcher:
    """Fetches sector index heatmap data directly from the NSE India website."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                            '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/market-data/live-equity-market',
            'Connection': 'keep-alive',
        })
        self._warmed_up = False

    def _warm_up(self) -> bool:
        try:
            logger.info("Connecting to the NSE website to fetch fresh session cookies...")
            self.session.get(f"{Config.NSE_BASE_URL}/", timeout=8)
            self.session.get(f"{Config.NSE_BASE_URL}/market-data/live-equity-market", timeout=8)
            self._warmed_up = True
            logger.info("Connected to NSE - ready to pull index data.")
            return True
        except Exception as e:
            logger.error(f"Couldn't connect to the NSE website: {e}")
            self._warmed_up = False
            return False

    def get_sector_heatmap(self) -> pd.DataFrame:
        try:
            if not self._warmed_up and not self._warm_up():
                return pd.DataFrame()

            logger.info("Asking NSE for today's sector index performance...")
            url = f"{Config.NSE_API_URL}/allIndices"
            resp = self.session.get(url, timeout=10)

            if resp.status_code != 200:
                logger.info("NSE session looked stale, refreshing cookies and retrying...")
                if self._warm_up():
                    resp = self.session.get(url, timeout=10)

            resp.raise_for_status()
            payload = resp.json()
            entries = payload.get('data', [])

            rows: List[Dict[str, Any]] = []
            for entry in entries:
                name = str(entry.get('index') or entry.get('indexName') or '').strip()
                if name not in Config.SECTOR_INDICES:
                    continue
                rows.append({
                    'sector': name,
                    'price': float(entry.get('last', 0) or 0),
                    'change_percent': float(entry.get('percentChange', 0) or 0),
                    'volatility': abs(float(entry.get('variation', 0) or 0)),
                    'volume': 0
                })

            df = pd.DataFrame(rows)
            if not df.empty:
                df['heatmap_score'] = df['change_percent'] * df['volatility']
                df = df.sort_values('change_percent', ascending=False)
                best, worst = df.iloc[0], df.iloc[-1]
                logger.info(
                    f"Got heatmap data for {len(df)} sectors. Best: {best['sector']} "
                    f"({best['change_percent']:+.2f}%). Worst: {worst['sector']} ({worst['change_percent']:+.2f}%)."
                )
            else:
                logger.warning("NSE returned no matching sector data - heatmap is empty this cycle.")

            return df

        except Exception as e:
            logger.error(f"Something went wrong fetching the NSE sector heatmap: {e}")
            return pd.DataFrame()


# ============================================
# MODULE: Dhan security master (authoritative security_id / lot_size / strikes)
# ============================================

class DhanInstrumentMaster:
    """
    Wraps the official Dhan security master CSV (via `dhanhq.fetch_security_list`)
    - the single authoritative source for security_id, lot_size, tick_size, expiry,
    and strike. Never treat a hardcoded derivative ID or a hand-assumed strike
    interval as stable; always resolve fresh against this master.

    Loaded once per trading day and cached in memory; also mirrored to a local CSV
    so a transient fetch failure doesn't block the whole session if the CSV was
    already loaded earlier that day.
    """

    def __init__(self, cache_file: str = "") -> None:
        self.cache_file = cache_file or Config.INSTRUMENT_MASTER_CACHE_FILE
        self.df: pd.DataFrame = pd.DataFrame()
        self._loaded_date: Optional[date_cls] = None

    def ensure_loaded(self) -> bool:
        today = ist_now().date()
        if not self.df.empty and self._loaded_date == today:
            return True

        if dhanhq_class is None:
            logger.error("Cannot load the Dhan security master - the dhanhq SDK isn't installed.")
            return False

        try:
            df = dhanhq_class.fetch_security_list(Config.INSTRUMENT_MASTER_MODE)
            if df is None or df.empty:
                raise ValueError("empty security master response")
            self.df = df
            self._loaded_date = today
            try:
                cache_dir = os.path.dirname(self.cache_file)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                self.df.to_csv(self.cache_file, index=False)
            except Exception as e:
                logger.warning(f"Could not cache the instrument master to disk: {e}")
            logger.info(f"Loaded the Dhan security master fresh ({len(self.df)} rows).")
            return True
        except Exception as e:
            logger.error(f"Could not fetch a fresh Dhan security master: {e}")
            if os.path.exists(self.cache_file):
                try:
                    self.df = pd.read_csv(self.cache_file)
                    logger.warning(
                        f"Falling back to the locally cached instrument master ({self.cache_file}) - "
                        "strikes/lot sizes may be stale until a fresh fetch succeeds."
                    )
                    return not self.df.empty
                except Exception as e2:
                    logger.error(f"Could not load the cached instrument master either: {e2}")
            return False

    def resolve_equity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Resolve a cash-market NSE equity symbol to its security_id."""
        if not self.ensure_loaded():
            return None
        symbol = str(symbol).strip().upper()
        df = self.df
        mask = (
            (df['SEM_EXM_EXCH_ID'].astype(str).str.upper() == 'NSE')
            & (df['SEM_TRADING_SYMBOL'].astype(str).str.upper() == symbol)
        )
        if 'SEM_INSTRUMENT_NAME' in df.columns:
            mask &= df['SEM_INSTRUMENT_NAME'].astype(str).str.upper() == 'EQUITY'
        elif 'SEM_OPTION_TYPE' in df.columns:
            # No instrument-type column to filter on - at least exclude derivative
            # rows (which always carry an option type) so we don't match an option leg.
            mask &= df['SEM_OPTION_TYPE'].isna() | (df['SEM_OPTION_TYPE'].astype(str).str.strip() == '')
        match = df[mask]
        if match.empty:
            return None
        row = match.iloc[0]
        return {
            'security_id': str(row['SEM_SMST_SECURITY_ID']),
            'trading_symbol': str(row['SEM_TRADING_SYMBOL']),
        }

    def resolve_index(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Resolve an NSE index (e.g. NIFTY/BANKNIFTY) to its security_id for spot data."""
        if not self.ensure_loaded():
            return None
        alias = {"NIFTY 50": "NIFTY", "NIFTY BANK": "BANKNIFTY"}
        symbol = alias.get(str(symbol).strip().upper(), str(symbol).strip().upper())
        df = self.df
        mask = (
            (df['SEM_EXM_EXCH_ID'].astype(str).str.upper() == 'NSE')
            & (df['SEM_INSTRUMENT_NAME'].astype(str).str.upper() == 'INDEX')
            & (df['SEM_TRADING_SYMBOL'].astype(str).str.upper() == symbol)
        )
        match = df[mask]
        if match.empty:
            return None
        row = match.iloc[0]
        return {
            'security_id': str(row['SEM_SMST_SECURITY_ID']),
            'trading_symbol': str(row['SEM_TRADING_SYMBOL']),
        }

    def resolve_option_contract(
        self, underlying: str, strike: float, option_type: str, expiry: str
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve one real, currently-listed option contract from the security master
        by underlying + strike + option type + expiry - never construct a trading
        symbol string by hand, and never treat the option chain's own strike list as
        proof a contract with that exact strike is actually tradable.
        """
        if not self.ensure_loaded():
            return None
        underlying = str(underlying).strip().upper()
        df = self.df
        # Exact first-token match, NOT startswith: "NIFTY" as a prefix also matches
        # "NIFTYNXT50-..."/"NIFTYFPI-..." contracts in the security master, which are
        # different underlyings entirely - a bare .startswith(underlying) silently
        # mis-resolves those for underlying="NIFTY".
        symbol_root = df['SEM_TRADING_SYMBOL'].astype(str).str.split('-').str[0].str.upper()
        mask = (
            (df['SEM_OPTION_TYPE'].astype(str).str.upper() == option_type.upper())
            & (symbol_root == underlying)
        )
        if expiry:
            target_date = pd.to_datetime(expiry).date()
            mask &= pd.to_datetime(df['SEM_EXPIRY_DATE'], errors='coerce').dt.date == target_date
        if strike is not None:
            mask &= pd.to_numeric(df['SEM_STRIKE_PRICE'], errors='coerce') == float(strike)

        matches = df[mask]
        if matches.empty:
            return None
        row = matches.iloc[0]
        lot_units = row.get('SEM_LOT_UNITS')
        return {
            'security_id': str(row['SEM_SMST_SECURITY_ID']),
            'trading_symbol': str(row.get('SEM_TRADING_SYMBOL', '')),
            'lot_size': int(lot_units) if lot_units not in (None, '') and not pd.isna(lot_units) else 0,
        }


# ============================================
# MODULE: market data - Dhan chart/quote/options
# ============================================

class DhanDataFetcher:
    """Fetches per-stock intraday chart data, previous-day close, quotes, and option
    chains via the official dhanhq SDK, resolving every instrument through
    DhanInstrumentMaster rather than any hand-built symbol string."""

    def __init__(self, client_manager: DhanClientManager, instruments: DhanInstrumentMaster) -> None:
        self.client_manager = client_manager
        self.instruments = instruments
        # Symbols that failed to resolve this session - don't hammer the API
        # with a symbol we already know Dhan can't find (spec: "do not
        # permanently patch individual symbols manually" + "do not
        # repeatedly hammer the API every scan").
        self._unavailable_symbols: Dict[str, str] = {}  # symbol -> reason

    def is_symbol_unavailable(self, symbol: str) -> Optional[str]:
        return self._unavailable_symbols.get(symbol)

    def _mark_unavailable(self, symbol: str, reason: str) -> None:
        if symbol not in self._unavailable_symbols:
            logger.error(f"Marking {symbol} unavailable for the rest of this session: {reason}")
        self._unavailable_symbols[symbol] = reason

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize a broker trading symbol before every data request."""
        if symbol is None:
            return ""
        return str(symbol).strip().upper()

    def get_intraday_ohlc(
        self,
        symbol: str,
        timeframe: str = Config.CANDLE_TIMEFRAME,
    ) -> pd.DataFrame:
        """Fetch intraday/daily OHLC for a cash-market NSE equity symbol."""
        return self._fetch_ohlc(
            symbol, timeframe,
            resolver=self.instruments.resolve_equity,
            instrument_type="EQUITY",
            exchange_segment=Config.EQUITY_EXCHANGE_SEGMENT,
        )

    def get_index_intraday_ohlc(
        self,
        symbol: str,
        timeframe: str = Config.CANDLE_TIMEFRAME,
    ) -> pd.DataFrame:
        """Fetch intraday/daily OHLC for an NSE index (e.g. NIFTY/BANKNIFTY spot)."""
        return self._fetch_ohlc(
            symbol, timeframe,
            resolver=self.instruments.resolve_index,
            instrument_type="INDEX",
            exchange_segment=Config.INDEX_EXCHANGE_SEGMENT,
        )

    def _fetch_ohlc(
        self,
        symbol: str,
        timeframe: str,
        resolver: Callable[[str], Optional[Dict[str, Any]]],
        instrument_type: str,
        exchange_segment: str,
    ) -> pd.DataFrame:
        """
        Shared fetch/retry/validation logic behind get_intraday_ohlc()/get_index_intraday_ohlc() -
        only the resolver + instrument_type/exchange_segment differ between equity and index.

        IMPORTANT:
        - A temporary API failure must NOT permanently blacklist a symbol.
        - Only this scan is skipped after retries are exhausted.
        """
        symbol = self._normalize_symbol(symbol)
        timeframe = str(timeframe).strip().upper()
        unavailable_key = f"{instrument_type}:{symbol}"

        if not symbol:
            logger.error("DATA_CONFIG_ERROR: Empty trading symbol.")
            return pd.DataFrame()

        if self.is_symbol_unavailable(unavailable_key):
            logger.warning(
                f"{symbol}: previously marked unavailable; skipping this request."
            )
            return pd.DataFrame()

        resolved = resolver(symbol)
        if not resolved:
            self._mark_unavailable(unavailable_key, "not found in the Dhan security master")
            return pd.DataFrame()
        security_id = resolved['security_id']

        client = self.client_manager.get_client()
        if client is None:
            logger.warning(f"Skipping {symbol} - not connected to Dhan.")
            return pd.DataFrame()

        supported_timeframes = {"1", "5", "15", "25", "60", "DAY"}
        if timeframe not in supported_timeframes:
            logger.error(
                f"DATA_CONFIG_ERROR: Unsupported timeframe '{timeframe}' "
                f"for {symbol}. Supported: {sorted(supported_timeframes)}"
            )
            return pd.DataFrame()

        now = ist_now()
        last_error = ""

        for attempt in range(1, 4):
            try:
                logger.debug(
                    f"Dhan OHLC request: {symbol} (security_id {security_id}) | "
                    f"timeframe={timeframe} | attempt={attempt}/3"
                )

                if timeframe == "DAY":
                    response = client.historical_daily_data(
                        security_id=security_id,
                        exchange_segment=exchange_segment,
                        instrument_type=instrument_type,
                        from_date=(now - timedelta(days=30)).strftime('%Y-%m-%d'),
                        to_date=now.strftime('%Y-%m-%d'),
                    )
                else:
                    response = client.intraday_minute_data(
                        security_id=security_id,
                        exchange_segment=exchange_segment,
                        instrument_type=instrument_type,
                        from_date=now.replace(hour=0, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S'),
                        to_date=now.strftime('%Y-%m-%d %H:%M:%S'),
                        interval=int(timeframe),
                    )

                df = self._normalize_ohlc(response)

                if not df.empty:
                    return df

                last_error = self._response_error(response) or "empty response"
                logger.warning(
                    f"EMPTY_OHLC: {symbol} | timeframe={timeframe} | attempt={attempt}/3 | {last_error}"
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"OHLC_REQUEST_FAILED: {symbol} | timeframe={timeframe} | attempt={attempt}/3 | {e}"
                )

            if attempt < 3:
                time.sleep(1.0)

        # Do NOT blacklist based merely on two/three failed calls.
        # A symbol can be valid while the broker/API is temporarily unavailable.
        logger.error(
            f"OHLC_UNAVAILABLE: {symbol} failed after 3 attempts. Last error: {last_error}"
        )

        return pd.DataFrame()

    @staticmethod
    def _response_error(response: Any) -> str:
        if isinstance(response, dict) and response.get('status') != 'success':
            return str(response.get('remarks') or 'unknown error')
        return ""

    @staticmethod
    def _normalize_ohlc(response: Any) -> pd.DataFrame:
        """Parse a dhanhq chart-data envelope: {"status": "success", "data": {"open": [...], ...}}."""
        if not isinstance(response, dict) or response.get('status') != 'success':
            return pd.DataFrame()

        payload = response.get('data', {})
        if not isinstance(payload, dict):
            return pd.DataFrame()

        for required in ('open', 'high', 'low', 'close'):
            if required not in payload:
                logger.error(f"Dhan returned chart data missing '{required}' - skipping.")
                return pd.DataFrame()

        n = len(payload['open'])
        df = pd.DataFrame({
            'Open': payload['open'],
            'High': payload['high'],
            'Low': payload['low'],
            'Close': payload['close'],
            'Volume': payload.get('volume', [0] * n),
        })

        if 'timestamp' in payload and payload['timestamp']:
            df['Timestamp'] = (
                pd.to_datetime(payload['timestamp'], unit='s', utc=True)
                .tz_convert('Asia/Kolkata')
                .tz_localize(None)
            )
            df = df.sort_values('Timestamp').reset_index(drop=True)

        # --- Data-quality validation (spec: reject invalid market data) ---
        bad_rows = (
            (df['High'] < df['Low']) |
            (df['Close'] <= 0) | (df['Open'] <= 0) |
            (df['High'] <= 0) | (df['Low'] <= 0)
        )
        if bad_rows.any():
            logger.warning(f"DATA_QUALITY_REJECTED: dropping {int(bad_rows.sum())} candle(s) with impossible OHLC values.")
            df = df[~bad_rows].reset_index(drop=True)
        if 'Timestamp' in df.columns and df['Timestamp'].duplicated().any():
            dupes = int(df['Timestamp'].duplicated().sum())
            logger.warning(f"DATA_QUALITY_REJECTED: dropping {dupes} duplicate-timestamp candle(s).")
            df = df.drop_duplicates(subset='Timestamp', keep='last').reset_index(drop=True)

        return df

    def get_previous_trading_day_close(self, symbol: str) -> float:
        """
        Fetch the previous TRADING DAY's close using daily candles.

        Retries transient failures and never silently converts a missing
        previous close into a 0% stock return.
        """
        symbol = self._normalize_symbol(symbol)

        if not symbol:
            return 0.0

        resolved = self.instruments.resolve_equity(symbol)
        if not resolved:
            return 0.0
        security_id = resolved['security_id']

        client = self.client_manager.get_client()
        if client is None:
            return 0.0

        now = ist_now()
        last_error = ""

        for attempt in range(1, 4):
            try:
                response = client.historical_daily_data(
                    security_id=security_id,
                    exchange_segment=Config.EQUITY_EXCHANGE_SEGMENT,
                    instrument_type="EQUITY",
                    from_date=(now - timedelta(days=15)).strftime('%Y-%m-%d'),
                    to_date=now.strftime('%Y-%m-%d'),
                )

                df = self._normalize_ohlc(response)

                if not df.empty:
                    today = now.date()

                    if 'Timestamp' in df.columns:
                        df = df[df['Timestamp'].dt.date < today]

                    if not df.empty:
                        previous_close = float(df['Close'].iloc[-1])

                        if previous_close > 0:
                            return previous_close

                        last_error = (
                            f"invalid previous close {previous_close}"
                        )
                    else:
                        last_error = "no historical trading-day candle before today"
                else:
                    last_error = self._response_error(response) or "empty daily response"

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"DAILY_CLOSE_FAILED: {symbol} | "
                    f"attempt={attempt}/3 | {e}"
                )

            if attempt < 3:
                time.sleep(1.0)

        logger.error(
            f"PREVIOUS_CLOSE_UNAVAILABLE: {symbol} after 3 attempts. "
            f"Last error: {last_error}"
        )
        return 0.0

    @staticmethod
    def _unwrap_maybe_double_nested(response: Any) -> Optional[Dict[str, Any]]:
        """
        Some dhanhq v2 endpoints (option_chain, expiry_list, ticker/quote data)
        return an SDK envelope {"status", "data": ...} where the raw Dhan HTTP
        body already carries its own nested "data" key - i.e. the real payload
        can be response["data"] OR response["data"]["data"] depending on SDK
        version. Handle both rather than assuming one fixed shape.
        """
        if not isinstance(response, dict) or response.get('status') != 'success':
            return None
        payload = response.get('data', {})
        if isinstance(payload, dict) and 'data' in payload:
            return payload['data']
        return payload

    def get_ltp(self, symbol: str) -> float:
        """Fetch the latest traded price for a cash-market equity symbol, or an index (NIFTY/BANKNIFTY) spot."""
        symbol = self._normalize_symbol(symbol)
        if symbol in Config.TRADING_INDICES:
            resolved = self.instruments.resolve_index(symbol)
            exchange_segment = Config.INDEX_EXCHANGE_SEGMENT
        else:
            resolved = self.instruments.resolve_equity(symbol)
            exchange_segment = Config.EQUITY_EXCHANGE_SEGMENT
        if not resolved:
            return 0.0
        return self.get_ltp_by_security_id(resolved['security_id'], exchange_segment)

    def get_ltp_by_security_id(self, security_id: str, exchange_segment: str) -> float:
        """Fetch the latest traded price for any already-resolved security_id (equity or option)."""
        client = self.client_manager.get_client()
        if client is None:
            return 0.0
        try:
            response = client.ticker_data({exchange_segment: [int(security_id)]})
            inner = self._unwrap_maybe_double_nested(response)
            if not isinstance(inner, dict):
                return 0.0
            segment_data = inner.get(exchange_segment, {})
            leg = segment_data.get(str(security_id)) if isinstance(segment_data, dict) else None
            if leg is None and isinstance(segment_data, dict):
                leg = segment_data.get(int(security_id))
            price = leg.get('last_price') if isinstance(leg, dict) else None
            return float(price) if price else 0.0
        except Exception as e:
            logger.error(f"Couldn't fetch the latest price for security_id {security_id}: {e}")
            return 0.0

    def get_ltp_batch(self, security_ids: List[str], exchange_segment: str) -> Dict[str, float]:
        """
        Fetch last-traded prices for MULTIPLE security_ids in a single call.

        Dhan's quote-snapshot APIs (ticker_data/ohlc_data/quote_data) are all rate-limited
        to 1 request/second. Calling get_ltp_by_security_id() once per open position in a
        loop burns through that limit in milliseconds, so every position after the first
        in a monitoring cycle gets throttled and comes back empty. Batching every id into
        one request (the API already accepts a list per segment) keeps this to 1 call/cycle
        regardless of how many positions are open.
        """
        result: Dict[str, float] = {str(sid): 0.0 for sid in security_ids}
        if not security_ids:
            return result
        client = self.client_manager.get_client()
        if client is None:
            return result
        try:
            response = client.ticker_data({exchange_segment: [int(sid) for sid in security_ids]})
            inner = self._unwrap_maybe_double_nested(response)
            if not isinstance(inner, dict):
                return result
            segment_data = inner.get(exchange_segment, {})
            if not isinstance(segment_data, dict):
                return result
            for sid in security_ids:
                leg = segment_data.get(str(sid))
                if leg is None:
                    leg = segment_data.get(int(sid))
                price = leg.get('last_price') if isinstance(leg, dict) else None
                result[str(sid)] = float(price) if price else 0.0
            return result
        except Exception as e:
            logger.error(f"Couldn't fetch a batch of latest prices for security_ids {security_ids}: {e}")
            return result

    def get_option_chain(self, underlying: str) -> Optional[Tuple[float, pd.DataFrame, str]]:
        """
        Fetch the REAL option chain for an underlying and return (spot, chain_df, expiry).
        chain_df has one row per actually-listed strike with real ce_/pe_ liquidity
        fields - the caller must never assume a fixed strike interval and must only
        pick from strikes that appear in this frame.
        """
        underlying = self._normalize_symbol(underlying)
        is_index = underlying in Config.TRADING_INDICES
        resolver = self.instruments.resolve_index if is_index else self.instruments.resolve_equity
        exchange_segment = Config.INDEX_EXCHANGE_SEGMENT if is_index else Config.EQUITY_EXCHANGE_SEGMENT
        resolved = resolver(underlying)
        if not resolved:
            logger.error(f"OPTION_CHAIN_UNAVAILABLE: {underlying} not found in the security master.")
            return None
        security_id = resolved['security_id']

        client = self.client_manager.get_client()
        if client is None:
            return None

        try:
            expiries = self._get_expiry_list_raw(client, security_id, exchange_segment)
            if not expiries:
                logger.error(f"OPTION_CHAIN_UNAVAILABLE: no expiries found for {underlying}.")
                return None
            # Valid expiries are at least Config.MIN_DAYS_TO_EXPIRY out - never today's expiry
            # (0 days left), which is illiquid/pinned and about to be worthless theta-wise.
            today = ist_now().date()
            valid_expiries = sorted(
                e for e in expiries
                if (pd.to_datetime(e).date() - today).days >= Config.MIN_DAYS_TO_EXPIRY
            )
            if not valid_expiries:
                logger.error(f"OPTION_CHAIN_UNAVAILABLE: no expiry for {underlying} is >= {Config.MIN_DAYS_TO_EXPIRY} day(s) out.")
                return None
            if Config.TARGET_DTE is None:
                expiry = str(valid_expiries[0])  # nearest valid expiry (default)
            else:
                target_dte = int(Config.TARGET_DTE)
                expiry = str(min(
                    valid_expiries,
                    key=lambda e: abs((pd.to_datetime(e).date() - today).days - target_dte)
                ))

            time.sleep(3.0)  # option-chain rate limit: one unique request per 3 seconds

            response = client.option_chain(
                under_security_id=int(security_id),
                under_exchange_segment=exchange_segment,
                expiry=expiry,
            )
            inner = self._unwrap_maybe_double_nested(response)
            if not isinstance(inner, dict):
                logger.error(f"OPTION_CHAIN_FAILED: {underlying} - {self._response_error(response) or 'bad response shape'}")
                return None

            spot = float(inner.get('last_price') or 0.0)
            chain_df = self._normalize_option_chain(inner)
            if chain_df.empty or spot <= 0:
                logger.error(f"OPTION_CHAIN_EMPTY: {underlying} returned no usable strikes/spot.")
                return None
            return spot, chain_df, expiry
        except Exception as e:
            logger.error(f"Error fetching option chain for {underlying}: {e}")
            return None

    def _get_expiry_list_raw(self, client: Any, security_id: str, exchange_segment: str) -> List[str]:
        try:
            response = client.expiry_list(under_security_id=int(security_id), under_exchange_segment=exchange_segment)
            inner = self._unwrap_maybe_double_nested(response)
            if inner is None:
                logger.error(f"EXPIRY_LIST_FAILED: {security_id} - {self._response_error(response)}")
                return []
            return list(inner or [])
        except Exception as e:
            logger.error(f"Couldn't fetch expiry list for security_id {security_id}: {e}")
            return []

    @staticmethod
    def _normalize_option_chain(raw: Dict[str, Any]) -> pd.DataFrame:
        """Normalize the raw {"last_price", "oc": {strike: {"ce": {...}, "pe": {...}}}}
        payload into one row per real listed strike."""
        oc = raw.get('oc', {}) or {}
        rows: List[Dict[str, Any]] = []
        for strike_key, leg_payload in oc.items():
            try:
                strike = float(strike_key)
            except (TypeError, ValueError):
                continue
            row: Dict[str, Any] = {'strike': strike}
            for side in ('ce', 'pe'):
                leg = (leg_payload or {}).get(side) or {}
                row[f'{side}_ltp'] = leg.get('last_price')
                row[f'{side}_oi'] = leg.get('oi')
                row[f'{side}_volume'] = leg.get('volume')
                row[f'{side}_bid_price'] = leg.get('top_bid_price')
                row[f'{side}_ask_price'] = leg.get('top_ask_price')
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values('strike').reset_index(drop=True)

    def resolve_option_contract(
        self, underlying: str, strike: float, option_type: str, expiry: str
    ) -> Optional[Dict[str, Any]]:
        return self.instruments.resolve_option_contract(underlying, strike, option_type, expiry)


# ============================================
# MODULE: options - liquidity-aware selector (item G/H/J/K)
# ============================================

class OptionSelector:
    """
    Resolves a tradable option contract for a directional signal from Dhan's REAL
    option-chain data. ATM is the nearest strike that is ACTUALLY LISTED in the
    chain (never an assumed round-number interval), and ATM+1/ATM-1 are the real
    neighboring listed strikes either side - so an irregular strike gap (common on
    stock options) is handled correctly instead of silently picking a strike that
    doesn't exist or isn't the true next one. Each candidate is evaluated against
    real liquidity/spread/premium data and resolved to a real security_id via the
    security master; only a contract that passes every filter is ever returned.
    """

    def __init__(self, dhan: DhanDataFetcher) -> None:
        self.dhan = dhan

    def select(self, underlying: str, direction: str) -> Optional[OptionContract]:
        option_type = 'CE' if direction == 'UP' else 'PE'

        chain_result = self.dhan.get_option_chain(underlying)
        if chain_result is None:
            logger.warning(f"OPTION_REJECTED: no option chain available for {underlying}.")
            return None
        spot, chain_df, expiry = chain_result
        if chain_df.empty or spot <= 0:
            logger.warning(f"OPTION_REJECTED: empty/invalid option chain for {underlying}.")
            return None

        strikes = sorted(chain_df['strike'].unique().tolist())
        atm_strike = min(strikes, key=lambda s: abs(s - spot))
        atm_idx = strikes.index(atm_strike)
        higher_strike = strikes[atm_idx + 1] if atm_idx + 1 < len(strikes) else None
        lower_strike = strikes[atm_idx - 1] if atm_idx - 1 >= 0 else None

        if option_type == 'CE':
            # OTM for a call is the next higher REAL strike; ITM for a call is the next lower REAL strike.
            candidates_raw = [("ATM", atm_strike), ("ATM+1", higher_strike), ("ATM-1", lower_strike)]
        else:
            # OTM for a put is the next lower REAL strike; ITM for a put is the next higher REAL strike.
            candidates_raw = [("ATM", atm_strike), ("ATM+1", lower_strike), ("ATM-1", higher_strike)]

        candidates: List[OptionCandidate] = []
        for label, strike in candidates_raw:
            if strike is None:
                logger.info(f"OPTION_REJECTED: {underlying} {option_type} {label} - no listed strike exists at this position in the chain.")
                continue
            row_match = chain_df[chain_df['strike'] == strike]
            if row_match.empty:
                continue
            candidate = self._evaluate(underlying, float(strike), option_type, label, row_match.iloc[0], expiry)
            candidates.append(candidate)
            if candidate.passes_filters:
                logger.info(
                    f"{underlying} {option_type} {label} (strike {strike}, {candidate.option_symbol}): "
                    f"premium Rs.{candidate.ltp:.2f}, spread {candidate.spread_percent}%, "
                    f"volume {candidate.volume}, OI {candidate.oi} - OK."
                )
            else:
                logger.info(f"OPTION_REJECTED: {underlying} {option_type} {label} (strike {strike}) - {candidate.rejection_reason}")

        # Default preference: ATM, unless it fails the filters, per spec.
        for label in ("ATM", "ATM+1", "ATM-1"):
            match = next((c for c in candidates if c.strike_label == label and c.passes_filters), None)
            if match:
                reason = "default ATM preference" if label == "ATM" else f"ATM rejected, fell back to {label}"
                logger.info(f"OPTION_SELECTED: {underlying} {option_type} {label} ({match.option_symbol}) - {reason}.")
                return OptionContract(
                    option_symbol=match.option_symbol,
                    underlying_symbol=underlying,
                    option_type=option_type,
                    strike_price=match.strike_price,
                    expiry_date=expiry,
                    lot_size=match.lot_size or Config.FALLBACK_LOT_SIZE,
                    premium=match.ltp,
                    selection_reason=reason,
                    distance_from_atm=label,
                    bid=match.bid,
                    ask=match.ask,
                    spread_percent=match.spread_percent,
                    volume=match.volume,
                    oi=match.oi,
                    security_id=match.security_id,
                )

        logger.warning(f"OPTION_REJECTED: no {option_type} candidate for {underlying} passed the liquidity/spread/premium filters.")
        return None

    def _evaluate(
        self, underlying: str, strike: float, option_type: str, label: str, row: "pd.Series", expiry: str
    ) -> OptionCandidate:
        candidate = OptionCandidate(option_symbol="", option_type=option_type, strike_label=label, strike_price=strike)

        side = 'ce' if option_type == 'CE' else 'pe'
        ltp = row.get(f'{side}_ltp')
        bid = row.get(f'{side}_bid_price')
        ask = row.get(f'{side}_ask_price')
        volume = row.get(f'{side}_volume')
        oi = row.get(f'{side}_oi')

        candidate.ltp = float(ltp) if ltp else 0.0
        candidate.bid = float(bid) if bid else 0.0
        candidate.ask = float(ask) if ask else 0.0
        candidate.volume = int(volume) if volume else 0
        candidate.oi = int(oi) if oi else 0

        # The chain gives liquidity data keyed by strike; the tradable security_id and
        # lot_size still have to come from the security master - a listed strike in
        # the chain is not, by itself, proof of a resolvable order-eligible contract.
        resolved = self.dhan.resolve_option_contract(underlying, strike, option_type, expiry)
        if resolved:
            candidate.option_symbol = resolved['trading_symbol']
            candidate.security_id = resolved['security_id']
            candidate.lot_size = resolved['lot_size']

        if candidate.bid > 0 and candidate.ask > 0 and candidate.ltp > 0:
            candidate.spread_percent = round(((candidate.ask - candidate.bid) / candidate.ltp) * 100, 2)

        # --- Filters, in priority order, first failure wins the reason ---
        if not resolved or not candidate.security_id:
            candidate.rejection_reason = "couldn't resolve a real, order-eligible contract from the security master"
        elif candidate.ltp <= 0:
            candidate.rejection_reason = "no live premium available"
        elif candidate.lot_size <= 0:
            candidate.rejection_reason = "couldn't resolve a valid lot size"
        elif candidate.ltp < Config.MIN_OPTION_PREMIUM:
            candidate.rejection_reason = f"premium Rs.{candidate.ltp:.2f} below minimum Rs.{Config.MIN_OPTION_PREMIUM}"
        elif candidate.ltp > Config.MAX_OPTION_PREMIUM:
            candidate.rejection_reason = f"premium Rs.{candidate.ltp:.2f} above maximum Rs.{Config.MAX_OPTION_PREMIUM}"
        elif Config.MIN_OPTION_VOLUME > 0 and candidate.volume < Config.MIN_OPTION_VOLUME:
            candidate.rejection_reason = f"volume {candidate.volume} below minimum {Config.MIN_OPTION_VOLUME}"
        elif Config.MIN_OPTION_OI > 0 and candidate.oi < Config.MIN_OPTION_OI:
            candidate.rejection_reason = f"OI {candidate.oi} below minimum {Config.MIN_OPTION_OI}"
        elif candidate.spread_percent is not None and candidate.spread_percent > Config.MAX_OPTION_SPREAD_PERCENT:
            candidate.rejection_reason = f"spread {candidate.spread_percent}% above maximum {Config.MAX_OPTION_SPREAD_PERCENT}%"
        else:
            candidate.passes_filters = True

        return candidate


# ============================================
# MODULE: data facade
# ============================================

class DataFetcher:
    """Combines NSE (heatmap) and dhanhq (chart/quote/options) data sources."""

    def __init__(self, client_manager: DhanClientManager) -> None:
        self.nse = NSEDataFetcher()
        self.instruments = DhanInstrumentMaster()
        self.dhan = DhanDataFetcher(client_manager, self.instruments)
        self.option_selector = OptionSelector(self.dhan)
        self.regime_classifier = RegimeClassifier()
        self.last_update: Optional[datetime] = None

    def get_sector_heatmap(self) -> pd.DataFrame:
        df = self.nse.get_sector_heatmap()
        self.last_update = datetime.now()
        return df

    def get_sector_stocks(self, sector_name: str) -> List[Stock]:
        try:
            # Normalize the configured universe once per scan.
            symbols = [
                str(s).strip().upper()
                for s in Config.SECTOR_STOCKS.get(sector_name, [])
                if str(s).strip()
            ]

            logger.info(
                f"Checking {len(symbols)} stocks in {sector_name} "
                f"for trade setups..."
            )

            stocks: List[Stock] = []
            for symbol in symbols:
                stock = self.get_stock(symbol, sector_name)
                if stock:
                    stocks.append(stock)

                # Avoid hammering the historical-data endpoint.
                time.sleep(1.0)

            stocks.sort(key=lambda x: x.change_percent, reverse=True)
            top_stocks = stocks[:Config.STOCKS_PER_SECTOR]
            if top_stocks:
                names = ', '.join(f"{s.symbol} ({s.change_percent:+.2f}%)" for s in top_stocks)
                logger.info(f"Top movers in {sector_name} (vs previous day close): {names}")
            else:
                logger.warning(f"Couldn't get usable chart data for any stock in {sector_name} this cycle.")
            return top_stocks
        except Exception as e:
            logger.error(f"Error fetching stocks for sector {sector_name}: {e}")
            return []

    def get_stock(self, symbol: str, sector_name: str = "") -> Optional[Stock]:
        """Build a Stock with ORB levels (from today's candles) and daily return (vs previous trading day's close)."""
        ohlc = self.dhan.get_intraday_ohlc(symbol)
        if ohlc.empty:
            return None
        prev_close = self.dhan.get_previous_trading_day_close(symbol)
        # Missing previous close is a data-quality failure, not a 0% return -
        # never let bad/missing broker data distort sector stock ranking.
        if prev_close <= 0:
            logger.warning(f"{symbol}: previous trading-day close unavailable - skipping stock for this scan.")
            return None
        return self._build_stock_from_ohlc(symbol, sector_name, ohlc, prev_close=prev_close, has_real_volume=True)

    def get_index_stock(self, symbol: str) -> Optional[Stock]:
        """Build a Stock for an index (NIFTY/BANKNIFTY spot) - no previous-close/change_percent
        (not used by the direct-breakout rule) and no real traded volume (the volume-confirmation
        check is skipped for these symbols - see StrategyEngine.evaluate_direct_breakout)."""
        ohlc = self.dhan.get_index_intraday_ohlc(symbol)
        if ohlc.empty:
            return None
        return self._build_stock_from_ohlc(symbol, sector_name="INDEX", ohlc=ohlc, prev_close=0.0, has_real_volume=False)

    def _build_stock_from_ohlc(
        self, symbol: str, sector_name: str, ohlc: pd.DataFrame, prev_close: float, has_real_volume: bool
    ) -> Optional[Stock]:
        try:
            if 'Timestamp' not in ohlc.columns:
                logger.error(f"{symbol}: chart data has no timestamp column, can't isolate today's session - skipping.")
                return None

            today = ist_now().date()
            todays = ohlc[ohlc['Timestamp'].dt.date == today].reset_index(drop=True)
            if todays.empty:
                logger.warning(f"{symbol}: no candles for today yet (market may not be open, or no trades so far) - skipping.")
                return None

            current_price = float(todays['Close'].iloc[-1])
            change_percent = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0

            orb_complete = is_orb_complete()
            candle_minutes = int(Config.CANDLE_TIMEFRAME)
            orb_candles = max(1, Config.ORB_MINUTES // candle_minutes)
            orb_period = min(orb_candles, len(todays))
            orb_high = float(todays['High'].iloc[:orb_period].max())
            orb_low = float(todays['Low'].iloc[:orb_period].min())
            orb_range = ((orb_high - orb_low) / orb_low) * 100 if orb_low > 0 else 0.0

            status = "final" if (orb_complete and len(todays) >= orb_candles) else "still forming"
            logger.info(
                f"{symbol}: opening range ({status}, {len(todays)} candles seen today) is "
                f"High=Rs.{orb_high:.2f} / Low=Rs.{orb_low:.2f} (range {orb_range:.2f}%). "
                f"Current price Rs.{current_price:.2f}."
            )

            has_volume = has_real_volume and 'Volume' in todays and todays['Volume'].sum() > 0
            if has_volume:
                typical_price = (todays['High'] + todays['Low'] + todays['Close']) / 3.0
                vwap = float((typical_price * todays['Volume']).sum() / todays['Volume'].sum())
                # Rolling mean of the PRIOR VOLUME_LOOKBACK_CANDLES candles, excluding the current one -
                # the spec's "average volume of the previous 20 five-minute candles", not a cumulative
                # today-so-far average.
                prior = todays['Volume'].iloc[:-1].tail(Config.VOLUME_LOOKBACK_CANDLES)
                avg_volume_20 = float(prior.mean()) if not prior.empty else 0.0
                last_volume = int(todays['Volume'].iloc[-1])
            else:
                typical_price = (todays['High'] + todays['Low'] + todays['Close']) / 3.0
                vwap = float(typical_price.mean())
                avg_volume_20 = 0.0
                last_volume = 0

            atr = compute_atr(todays, Config.REGIME_ATR_PERIOD) if Config.ENABLE_REGIME_ADAPTIVE else 0.0

            return Stock(
                symbol=symbol, name=symbol, sector=sector_name,
                current_price=current_price, change_percent=change_percent,
                volume=last_volume,
                orb_high=orb_high, orb_low=orb_low, orb_range=orb_range,
                orb_complete=orb_complete and len(todays) >= orb_candles,
                vwap=vwap, avg_volume_20=avg_volume_20, candle_count=len(todays),
                has_real_volume=has_volume,
                current_high=float(todays['High'].iloc[-1]), current_low=float(todays['Low'].iloc[-1]),
                current_open=float(todays['Open'].iloc[-1]), atr=atr,
            )
        except Exception as e:
            logger.error(f"Error building stock data for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> float:
        return self.dhan.get_ltp(symbol)

    def get_option_premium(self, security_id: str) -> float:
        if not security_id:
            return 0.0
        return self.dhan.get_ltp_by_security_id(security_id, Config.DERIVATIVE_EXCHANGE_SEGMENT)

    def get_option_premiums_batch(self, security_ids: List[str]) -> Dict[str, float]:
        """One rate-limit-friendly call for every open position's premium - see get_ltp_batch()."""
        return self.dhan.get_ltp_batch(security_ids, Config.DERIVATIVE_EXCHANGE_SEGMENT)

    def select_option(self, underlying: str, direction: str) -> Optional[OptionContract]:
        return self.option_selector.select(underlying, direction)

    def get_regime_snapshot(self, scanned_stocks: List[Stock]) -> RegimeSnapshot:
        """One whole-session regime read for this cycle - see RegimeClassifier.
        `scanned_stocks` is whatever the current cycle already fetched (used only for
        the breadth ratio - no extra API calls needed for that part)."""
        try:
            index_intraday = self.dhan.get_index_intraday_ohlc(Config.REGIME_INDEX_SYMBOL)
            daily = self.dhan.get_index_intraday_ohlc(Config.REGIME_INDEX_SYMBOL, timeframe="DAY")
            daily_atr_history = pd.Series(dtype=float)
            if not daily.empty and len(daily) > 1:
                daily_atr_history = compute_true_range(daily).tail(Config.REGIME_ATR_PERCENTILE_LOOKBACK_DAYS)

            vix_ohlc = self.dhan.get_index_intraday_ohlc(Config.REGIME_VIX_SYMBOL)
            vix_value = float(vix_ohlc['Close'].iloc[-1]) if not vix_ohlc.empty else 0.0

            opening_gap_percent = 0.0
            if not daily.empty and len(daily) >= 2 and not index_intraday.empty:
                prev_close = float(daily['Close'].iloc[-2])
                today_open = float(index_intraday['Open'].iloc[0])
                opening_gap_percent = ((today_open - prev_close) / prev_close * 100) if prev_close else 0.0

            green = sum(1 for s in scanned_stocks if s.change_percent > 0)
            breadth_ratio = (green / len(scanned_stocks)) if scanned_stocks else 0.5

            return self.regime_classifier.classify(
                index_intraday, daily_atr_history, vix_value, breadth_ratio, opening_gap_percent
            )
        except Exception as e:
            logger.error(f"Error computing market regime: {e}")
            return RegimeSnapshot(regime=MarketRegime.UNKNOWN)


# ============================================
# MODULE: regime classification (ADX/ATR/VIX/breadth -> MarketRegime)
# ============================================

def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's True Range per candle: max(H-L, |H-prevC|, |L-prevC|)."""
    high, low = df['High'], df['Low']
    prev_close = df['Close'].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    if len(tr) > 0:
        tr.iloc[0] = high.iloc[0] - low.iloc[0]  # no previous close for the first candle
    return tr


def compute_atr(df: pd.DataFrame, period: int) -> float:
    if df is None or df.empty or len(df) < 2:
        return 0.0
    tr = compute_true_range(df)
    window = min(period, len(tr))
    return float(tr.tail(window).mean())


def compute_adx(df: pd.DataFrame, period: int) -> Optional[float]:
    """Wilder's ADX(period). Returns None if there isn't enough history yet -
    callers should treat that as "not trending" rather than erroring."""
    if df is None or len(df) < period + 1:
        return None
    high, low = df['High'], df['Low']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = compute_true_range(df)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, float('nan'))
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    di_sum = (plus_di + minus_di).replace(0, float('nan'))
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    val = adx.iloc[-1]
    return float(val) if pd.notna(val) else None


def percentile_rank(value: float, series: pd.Series) -> float:
    """% of `series` that is <= value - "where does today's ATR sit vs its recent history"."""
    series = series.dropna() if series is not None else pd.Series(dtype=float)
    if series.empty:
        return 50.0
    return float((series <= value).mean() * 100)


class RegimeClassifier:
    """
    Classifies the whole trading session into a MarketRegime from the regime
    index's (default NIFTY) intraday candles, India VIX, and scanned-universe
    breadth - "don't trade the same strategy every day". See Config's REGIME_*
    thresholds and MarketRegime/RegimeSnapshot for what each regime then allows.
    """

    def classify(
        self,
        index_intraday: pd.DataFrame,
        daily_atr_history: pd.Series,
        vix_value: float,
        breadth_ratio: float,
        opening_gap_percent: float,
    ) -> RegimeSnapshot:
        now = ist_now()
        if index_intraday is None or index_intraday.empty:
            return RegimeSnapshot(regime=MarketRegime.UNKNOWN, breadth_ratio=breadth_ratio, timestamp=now)

        adx = compute_adx(index_intraday, Config.REGIME_ADX_PERIOD)
        atr = compute_atr(index_intraday, Config.REGIME_ATR_PERIOD)  # intraday (5-min-candle) ATR - informational only, NOT compared against daily_atr_history (different scale)
        # `daily_atr_history` is a distribution of FULL trading-day true ranges (see
        # DataFetcher.get_regime_snapshot). Comparing a 5-min-candle ATR against that would
        # be an apples-to-oranges scale mismatch (it would almost always rank near the 0th
        # percentile). Compare like-for-like instead: today's cumulative high-low range so
        # far vs that same daily-range distribution - naturally understated early in the
        # session and firms up as the day progresses, same as a live regime read would.
        day_range_so_far = float(index_intraday['High'].max() - index_intraday['Low'].min())
        atr_percentile = percentile_rank(day_range_so_far, daily_atr_history)

        close = float(index_intraday['Close'].iloc[-1])
        typical = (index_intraday['High'] + index_intraday['Low'] + index_intraday['Close']) / 3.0
        vwap = float(typical.mean())  # index has no real traded volume - unweighted average, same fallback used for Stock.vwap
        vwap_distance_pct = ((close - vwap) / vwap * 100) if vwap else 0.0

        regime = self._decide(adx, atr_percentile, vix_value, vwap_distance_pct, breadth_ratio)

        logger.info(
            f"REGIME={regime.value}: ADX={adx if adx is not None else float('nan'):.1f}, "
            f"ATR={atr:.2f} (pctile {atr_percentile:.0f}), VIX={vix_value:.1f}, "
            f"VWAP_dist={vwap_distance_pct:+.2f}%, breadth={breadth_ratio:.2f}."
        )

        return RegimeSnapshot(
            regime=regime, adx=adx or 0.0, atr=atr, atr_percentile=atr_percentile,
            vwap_distance_percent=vwap_distance_pct, india_vix=vix_value,
            opening_gap_percent=opening_gap_percent, breadth_ratio=breadth_ratio, timestamp=now,
        )

    @staticmethod
    def _decide(
        adx: Optional[float], atr_percentile: float, vix_value: float,
        vwap_distance_pct: float, breadth_ratio: float,
    ) -> MarketRegime:
        high_vol = atr_percentile >= Config.REGIME_HIGH_VOL_ATR_PERCENTILE or (
            vix_value and vix_value >= Config.REGIME_VIX_HIGH
        )
        if high_vol:
            return MarketRegime.HIGH_VOLATILITY

        low_adx = adx is None or adx < Config.REGIME_CHOP_ADX_MAX
        low_vol = atr_percentile <= Config.REGIME_LOW_VOL_ATR_PERCENTILE
        if low_adx and low_vol:
            return MarketRegime.LOW_VOLATILITY_CHOP

        if adx is not None and adx >= Config.REGIME_TRENDING_ADX_MIN:
            if vwap_distance_pct > 0 and breadth_ratio >= Config.REGIME_BREADTH_BULLISH_MIN:
                return MarketRegime.TRENDING_BULL
            if vwap_distance_pct < 0 and breadth_ratio <= Config.REGIME_BREADTH_BEARISH_MAX:
                return MarketRegime.TRENDING_BEAR

        return MarketRegime.RANGE_BOUND


# ============================================
# MODULE: strategy - sector/stock ranking + breakout state machine
# ============================================

class StrategyEngine:
    """
    Direct-breakout ORB rule, shared by sector-heatmap stocks and independently-scanned
    indices (NIFTY/BANKNIFTY): a completed 5-min candle closing beyond the ORB level AND
    beyond VWAP AND (for symbols with real volume) with volume >= VOLUME_CONFIRMATION_MULTIPLE
    x the prior VOLUME_LOOKBACK_CANDLES average ARMS the sequence; the signal fires once a
    later candle exists ("enter on the next candle"). One shot per symbol per day - no
    pullback wait, no re-signal after a symbol has fired (win, loss, or otherwise) today.
    """

    def __init__(self) -> None:
        self._breakout_states: Dict[str, BreakoutState] = {}
        self._signal_seq = 0

    # ---- Sector classification (still used to pick WHICH stocks to scan) ----

    def classify_sectors(self, heatmap_df: pd.DataFrame) -> List[SectorRanking]:
        if heatmap_df.empty:
            return []
        now = datetime.now()
        rankings: List[SectorRanking] = []
        sorted_df = heatmap_df.sort_values('change_percent', ascending=False).reset_index(drop=True)
        for rank, (_, row) in enumerate(sorted_df.iterrows(), start=1):
            pct = float(row['change_percent'])
            if pct >= Config.SECTOR_MIN_POSITIVE_RETURN:
                direction = SectorDirection.POSITIVE
            elif pct <= -Config.SECTOR_MIN_NEGATIVE_RETURN:
                direction = SectorDirection.NEGATIVE
            else:
                direction = SectorDirection.NEUTRAL
            rankings.append(SectorRanking(sector=row['sector'], change_percent=pct, direction=direction,
                                           rank=rank, timestamp=now))
        return rankings

    def analyze_sector_heatmap(self, heatmap_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Positive sectors (CALL candidates) and negative sectors (PUT candidates) - never
        the 'least negative' sector padded into the positive bucket, or vice versa."""
        rankings = self.classify_sectors(heatmap_df)
        if not rankings:
            logger.warning("No heatmap data to analyze this cycle - skipping sector selection.")
            return [], []

        positive = [r.sector for r in rankings if r.direction == SectorDirection.POSITIVE][:Config.TOP_SECTORS]
        negative = [r.sector for r in rankings if r.direction == SectorDirection.NEGATIVE][:Config.TOP_SECTORS]
        neutral = [r.sector for r in rankings if r.direction == SectorDirection.NEUTRAL]

        logger.info(f"Positive sectors (CALL candidates, >= +{Config.SECTOR_MIN_POSITIVE_RETURN}%): {positive}")
        logger.info(f"Negative sectors (PUT candidates, <= -{Config.SECTOR_MIN_NEGATIVE_RETURN}%): {negative}")
        if neutral:
            logger.info(f"Neutral sectors (ignored, within +/-{Config.SECTOR_MIN_POSITIVE_RETURN}%): {neutral}")

        return positive, negative

    # ---- Direct-breakout entry rule ----

    def _get_state(self, stock: Stock) -> BreakoutState:
        """
        Look up (or create) the persisted per-symbol state for today.

        IMPORTANT: state must survive across scan cycles. It is only reset on a new
        trading day, or if the ORB levels themselves genuinely change by more than
        floating-point noise - an exact `!=` comparison on recomputed floats would
        spuriously wipe an in-progress ARMED sequence on almost every scan (each fetch
        can reconstruct the same ORB with a tiny last-bit float difference).
        """
        today = ist_now().date()
        state = self._breakout_states.get(stock.symbol)
        orb_changed = state is not None and (
            abs(state.orb_high - stock.orb_high) > 1e-6 or abs(state.orb_low - stock.orb_low) > 1e-6
        )
        if state is None or state.trade_date != today or orb_changed:
            if state is not None and orb_changed and state.trade_date == today:
                logger.warning(
                    f"[{stock.symbol}] ORB_LEVEL_CHANGED: orb_high Rs.{state.orb_high:.2f}->Rs.{stock.orb_high:.2f}, "
                    f"orb_low Rs.{state.orb_low:.2f}->Rs.{stock.orb_low:.2f} - resetting the breakout sequence."
                )
            state = BreakoutState(symbol=stock.symbol, trade_date=today, orb_high=stock.orb_high, orb_low=stock.orb_low)
            self._breakout_states[stock.symbol] = state
        return state

    def evaluate_direct_breakout(self, stock: Stock, regime: Optional[RegimeSnapshot] = None) -> Optional[TradeSignal]:
        """
        Config.ENABLE_REGIME_ADAPTIVE=False: original deterministic rule - a candle
        closing beyond OR+VWAP+volume ARMS the sequence, the very next candle fires.

        Config.ENABLE_REGIME_ADAPTIVE=True: the same displacement candle only ARMS the
        sequence; the signal now fires only after price pulls back toward the breakout
        zone and a confirmation candle holds it (see _check_pullback), and only if the
        current `regime` allows new entries in this direction.
        """
        regime = regime or RegimeSnapshot()
        if not stock.orb_complete:
            return None  # never generate signals from an incomplete ORB
        if stock.orb_range < Config.MIN_ORB_RANGE:
            logger.info(f"{stock.symbol}: opening range only {stock.orb_range:.2f}% wide (below {Config.MIN_ORB_RANGE}% minimum) - too tight, skipping.")
            return None
        if stock.orb_range > Config.MAX_ORB_RANGE:
            logger.warning(f"{stock.symbol}: opening range {stock.orb_range:.2f}% wide (above {Config.MAX_ORB_RANGE}% max) - looks abnormal, rejecting as bad data.")
            return None

        state = self._get_state(stock)
        if state.phase in (BreakoutPhase.FIRED, BreakoutPhase.EXPIRED):
            return None  # one shot per symbol per day - no re-entry after that symbol's SL/TP/expiry

        if not Config.ENABLE_REGIME_ADAPTIVE:
            if state.phase == BreakoutPhase.WAITING_FOR_BREAKOUT:
                self._check_displacement(stock, state)
                return None  # even a just-armed candle only fires on a LATER candle
            if state.phase == BreakoutPhase.ARMED:
                if stock.candle_count <= state.armed_candle_count:
                    return None  # still the same candle that armed it
                return self._fire_signal(stock, state, regime)
            return None

        # --- regime-adaptive path ---
        if state.phase == BreakoutPhase.WAITING_FOR_BREAKOUT:
            if not regime.allows_new_entries():
                return None  # e.g. LOW_VOLATILITY_CHOP - don't even start arming new setups
            self._check_displacement(stock, state)
            return None

        if state.phase in (BreakoutPhase.ARMED, BreakoutPhase.PULLBACK_TOUCHED):
            return self._check_pullback(stock, state, regime)

        return None

    def _volume_confirmed(self, stock: Stock) -> bool:
        """True if the volume-confirmation check passes. Indices (has_real_volume=False)
        have no genuine traded volume on the spot instrument, so the check is a no-op there."""
        if not stock.has_real_volume:
            return True
        return stock.avg_volume_20 > 0 and stock.volume >= Config.VOLUME_CONFIRMATION_MULTIPLE * stock.avg_volume_20

    def _check_displacement(self, stock: Stock, state: BreakoutState) -> None:
        """A completed candle closing beyond OR+VWAP+volume displaces the sequence.
        With ENABLE_REGIME_ADAPTIVE=False this ARMS a next-candle fire (unchanged
        legacy behaviour); with it True this only starts the pullback wait."""
        price = stock.current_price  # last completed candle's close
        volume_ok = self._volume_confirmed(stock)
        volume_note = (
            f", volume {stock.volume} >= {Config.VOLUME_CONFIRMATION_MULTIPLE}x avg {stock.avg_volume_20:.0f}"
            if stock.has_real_volume else " (no real volume on this symbol - check skipped)"
        )
        pullback_wait_note = " - waiting for a pullback." if Config.ENABLE_REGIME_ADAPTIVE else " - entering on the next candle."

        if price > stock.orb_high and price > stock.vwap and volume_ok:
            state.phase = BreakoutPhase.ARMED
            state.direction = 'UP'
            state.breakout_price = price
            state.breakout_time = ist_now()
            state.armed_candle_count = stock.candle_count
            state.displacement_extreme = price
            state.pullback_zone = max(stock.orb_high, stock.vwap)
            state.candles_since_displacement = 0
            logger.info(
                f"[{stock.symbol}] STATE=ARMED/DISPLACED (UP): candle closed Rs.{price:.2f}, above ORB high Rs.{stock.orb_high:.2f} "
                f"and VWAP Rs.{stock.vwap:.2f}{volume_note}{pullback_wait_note}"
            )
        elif price < stock.orb_low and price < stock.vwap and volume_ok:
            state.phase = BreakoutPhase.ARMED
            state.direction = 'DOWN'
            state.breakout_price = price
            state.breakout_time = ist_now()
            state.armed_candle_count = stock.candle_count
            state.displacement_extreme = price
            state.pullback_zone = min(stock.orb_low, stock.vwap)
            state.candles_since_displacement = 0
            logger.info(
                f"[{stock.symbol}] STATE=ARMED/DISPLACED (DOWN): candle closed Rs.{price:.2f}, below ORB low Rs.{stock.orb_low:.2f} "
                f"and VWAP Rs.{stock.vwap:.2f}{volume_note}{pullback_wait_note}"
            )

    def _check_pullback(self, stock: Stock, state: BreakoutState, regime: RegimeSnapshot) -> Optional[TradeSignal]:
        """
        Displacement -> pullback -> confirmation:
          - structure invalidated (candle closes back beyond the ORB itself) -> EXPIRED
          - too many candles since displacement with no confirmed pullback -> EXPIRED
          - price pulls back into the breakout zone (OR level / VWAP, +/- a small buffer) -> PULLBACK_TOUCHED
          - once touched, a confirmation candle (closes in the breakout direction, back
            beyond the zone) fires the signal
        """
        state.candles_since_displacement += 1
        if state.candles_since_displacement > Config.PULLBACK_MAX_CANDLES:
            logger.info(
                f"[{stock.symbol}] PULLBACK_EXPIRED: no confirmed pullback within "
                f"{Config.PULLBACK_MAX_CANDLES} candles of displacement - setup expired for today."
            )
            state.phase = BreakoutPhase.EXPIRED
            return None

        price = stock.current_price
        low = stock.current_low or price
        high = stock.current_high or price
        buffer = Config.PULLBACK_ZONE_BUFFER_PERCENT / 100.0

        if state.direction == 'UP':
            state.displacement_extreme = max(state.displacement_extreme, high)
            if price < state.orb_low:
                logger.info(
                    f"[{stock.symbol}] STRUCTURE_INVALIDATED: closed Rs.{price:.2f} back below "
                    f"ORB low Rs.{state.orb_low:.2f} - setup expired."
                )
                state.phase = BreakoutPhase.EXPIRED
                return None
            if low <= state.pullback_zone * (1 + buffer):
                state.phase = BreakoutPhase.PULLBACK_TOUCHED
            if state.phase == BreakoutPhase.PULLBACK_TOUCHED and price > stock.current_open and price > state.pullback_zone:
                logger.info(
                    f"[{stock.symbol}] PULLBACK_CONFIRMED (UP): held near Rs.{state.pullback_zone:.2f}, "
                    f"confirmation candle closed Rs.{price:.2f} - firing."
                )
                return self._fire_signal(stock, state, regime)
            return None
        else:
            state.displacement_extreme = min(state.displacement_extreme, low)
            if price > state.orb_high:
                logger.info(
                    f"[{stock.symbol}] STRUCTURE_INVALIDATED: closed Rs.{price:.2f} back above "
                    f"ORB high Rs.{state.orb_high:.2f} - setup expired."
                )
                state.phase = BreakoutPhase.EXPIRED
                return None
            if high >= state.pullback_zone * (1 - buffer):
                state.phase = BreakoutPhase.PULLBACK_TOUCHED
            if state.phase == BreakoutPhase.PULLBACK_TOUCHED and price < stock.current_open and price < state.pullback_zone:
                logger.info(
                    f"[{stock.symbol}] PULLBACK_CONFIRMED (DOWN): held near Rs.{state.pullback_zone:.2f}, "
                    f"confirmation candle closed Rs.{price:.2f} - firing."
                )
                return self._fire_signal(stock, state, regime)
            return None

    def _fire_signal(self, stock: Stock, state: BreakoutState, regime: RegimeSnapshot) -> Optional[TradeSignal]:
        trade_type = TradeType.CALL if state.direction == 'UP' else TradeType.PUT

        if Config.ENABLE_REGIME_ADAPTIVE:
            if not regime.allows_direction(trade_type):
                logger.info(
                    f"[{stock.symbol}] REGIME_REJECTED: regime {regime.regime.value} does not allow "
                    f"{trade_type.value} setups - setup expired."
                )
                state.phase = BreakoutPhase.EXPIRED
                return None

            if Config.ENABLE_ORDER_FLOW_FILTER:
                candle_range = stock.current_high - stock.current_low
                up_ratio = ((stock.current_price - stock.current_low) / candle_range) if candle_range > 0 else 0.5
                if trade_type == TradeType.CALL and up_ratio < Config.ORDER_FLOW_MIN_UP_RATIO:
                    logger.info(
                        f"[{stock.symbol}] ORDER_FLOW_REJECTED: confirmation candle closed in the bottom "
                        f"{up_ratio * 100:.0f}% of its range (need >= {Config.ORDER_FLOW_MIN_UP_RATIO * 100:.0f}%) - setup expired."
                    )
                    state.phase = BreakoutPhase.EXPIRED
                    return None
                if trade_type == TradeType.PUT and up_ratio > (1 - Config.ORDER_FLOW_MIN_UP_RATIO):
                    logger.info(
                        f"[{stock.symbol}] ORDER_FLOW_REJECTED: confirmation candle closed in the top "
                        f"{(1 - up_ratio) * 100:.0f}% of its range (need >= {Config.ORDER_FLOW_MIN_UP_RATIO * 100:.0f}% downside) - setup expired."
                    )
                    state.phase = BreakoutPhase.EXPIRED
                    return None

            if Config.ENABLE_BREADTH_FILTER:
                if trade_type == TradeType.CALL and regime.breadth_ratio < Config.REGIME_BREADTH_BULLISH_MIN:
                    logger.info(
                        f"[{stock.symbol}] BREADTH_REJECTED: universe breadth {regime.breadth_ratio:.2f} below the "
                        f"{Config.REGIME_BREADTH_BULLISH_MIN} bullish-confirmation floor - setup expired."
                    )
                    state.phase = BreakoutPhase.EXPIRED
                    return None
                if trade_type == TradeType.PUT and regime.breadth_ratio > Config.REGIME_BREADTH_BEARISH_MAX:
                    logger.info(
                        f"[{stock.symbol}] BREADTH_REJECTED: universe breadth {regime.breadth_ratio:.2f} above the "
                        f"{Config.REGIME_BREADTH_BEARISH_MAX} bearish-confirmation ceiling - setup expired."
                    )
                    state.phase = BreakoutPhase.EXPIRED
                    return None

        self._signal_seq += 1
        signal_id = f"{ist_now().strftime('%Y%m%d')}-{stock.symbol}-{state.breakout_time.strftime('%H%M%S') if state.breakout_time else ''}-{self._signal_seq}"
        state.phase = BreakoutPhase.FIRED  # terminal for the day regardless of outcome
        state.signal_id = signal_id

        price = stock.current_price  # entry at the confirmation candle's close
        atr = stock.atr if Config.ENABLE_REGIME_ADAPTIVE else 0.0
        target_r = regime.target_r() if Config.ENABLE_REGIME_ADAPTIVE else Config.TAKE_PROFIT_1

        if state.direction == 'UP':
            structural_distance = price - stock.orb_low
            risk = max(structural_distance, Config.ATR_STOP_MULTIPLIER * atr) if Config.ENABLE_REGIME_ADAPTIVE else structural_distance
            risk = max(risk, 0.01)
            stop_loss = price - risk
            target_1 = price + risk * target_r
        else:
            structural_distance = stock.orb_high - price
            risk = max(structural_distance, Config.ATR_STOP_MULTIPLIER * atr) if Config.ENABLE_REGIME_ADAPTIVE else structural_distance
            risk = max(risk, 0.01)
            stop_loss = price + risk
            target_1 = price - risk * target_r

        relative_volume = (stock.volume / stock.avg_volume_20) if (stock.has_real_volume and stock.avg_volume_20 > 0) else 0.0

        logger.info(
            f"SIGNAL_FIRED: {stock.symbol} {trade_type.value} at Rs.{price:.2f} (displaced at Rs.{state.breakout_price:.2f}) - "
            f"SL Rs.{stop_loss:.2f} (risk Rs.{risk:.2f}, structural Rs.{structural_distance:.2f}, ATRx{Config.ATR_STOP_MULTIPLIER} Rs.{Config.ATR_STOP_MULTIPLIER * atr:.2f}), "
            f"target Rs.{target_1:.2f} ({target_r:.1f}R), regime={regime.regime.value}."
        )

        score, breakdown = self.score_signal(stock, relative_volume)
        logger.info(f"SIGNAL SCORE = {score:.1f} " + " | ".join(f"{k}={v:.1f}" for k, v in breakdown.items()))

        return TradeSignal(
            signal_id=signal_id, stock=stock, trade_type=trade_type,
            entry_price=price, stop_loss=stop_loss,
            target_1=target_1, target_2=target_1, target_3=target_1,
            score=score, score_breakdown=breakdown,
            breakout_price=state.breakout_price, breakout_time=state.breakout_time,
            pullback_price=price,
            vwap_at_signal=stock.vwap, relative_volume=relative_volume,
            regime=regime.regime, atr_at_signal=atr,
            risk_multiplier=regime.risk_multiplier() if Config.ENABLE_REGIME_ADAPTIVE else 1.0,
            trailing_enabled=regime.trailing_enabled() if Config.ENABLE_REGIME_ADAPTIVE else False,
        )

    # ---- Sorting-only score: the entry rule itself is a deterministic pass/fail
    # (MIN_SIGNAL_SCORE=0.0), so this only orders same-cycle signals for MAX_OPEN_POSITIONS ----

    def score_signal(self, stock: Stock, relative_volume: float) -> Tuple[float, Dict[str, float]]:
        orb_quality = min(10.0, (stock.orb_range / Config.MIN_ORB_RANGE) * 5) if Config.MIN_ORB_RANGE else 0.0
        # Indices have no relative_volume (no real volume) - score them neutrally on this
        # component rather than penalizing them against volume-scored stocks.
        volume_score = min(10.0, relative_volume * 5) if relative_volume else 5.0
        breakdown = {'orb_quality': round(orb_quality, 1), 'relative_volume': round(volume_score, 1)}
        return round(sum(breakdown.values()), 1), breakdown

    def gather_sector_universe(self, heatmap_df: pd.DataFrame, data_fetcher: DataFetcher,
                                already_traded_underlyings: Optional[set] = None) -> List[Stock]:
        """Fetch-only half of the old select_trades(): which stocks are even in play
        this cycle (sector heatmap ranking), with no breakout evaluation yet - callers
        need the full scanned universe assembled BEFORE computing the market regime
        (breadth needs it), and evaluate afterwards via evaluate_universe()."""
        already_traded_underlyings = already_traded_underlyings or set()
        stocks: List[Stock] = []
        try:
            positive_sectors, negative_sectors = self.analyze_sector_heatmap(heatmap_df)
            for sector in positive_sectors + negative_sectors:
                for stock in data_fetcher.get_sector_stocks(sector):
                    if stock.symbol not in already_traded_underlyings:
                        stocks.append(stock)
        except Exception as e:
            logger.error(f"Error gathering the sector universe: {e}")
        return stocks

    def evaluate_universe(self, stocks: List[Stock], regime: Optional[RegimeSnapshot] = None) -> List[TradeSignal]:
        """Runs evaluate_direct_breakout() over an already-fetched list of stocks
        (sector universe + independently-scanned indices) under one shared regime read,
        sorted best-score-first for the risk gate's MAX_OPEN_POSITIONS cutoff."""
        signals: List[TradeSignal] = []
        try:
            for stock in stocks:
                signal = self.evaluate_direct_breakout(stock, regime)
                if signal:
                    signals.append(signal)

            signals.sort(key=lambda s: s.score, reverse=True)
            if signals:
                logger.info(f"Found {len(signals)} qualifying signal(s) this cycle: {[s.stock.symbol for s in signals]}")
            else:
                logger.info("No qualifying trade signals found this cycle.")
            return signals
        except Exception as e:
            logger.error(f"Error evaluating the scanned universe: {e}")
            return []

    def select_trades(self, heatmap_df: pd.DataFrame, data_fetcher: DataFetcher,
                       already_traded_underlyings: Optional[set] = None,
                       regime: Optional[RegimeSnapshot] = None) -> List[TradeSignal]:
        """Convenience wrapper kept for any external/legacy caller: gather + evaluate
        in one call. TradingSystem._scan_for_signals uses the two-step gather/evaluate
        API directly so the regime can be computed from the FULL scanned universe
        (indices + sector stocks) before any breakout is evaluated."""
        stocks = self.gather_sector_universe(heatmap_df, data_fetcher, already_traded_underlyings)
        return self.evaluate_universe(stocks, regime)


# ============================================
# MODULE: risk management (item L)
# ============================================

@dataclass
class DailyRiskState:
    trade_date: str = ""
    trades_today: int = 0
    realized_pnl: float = 0.0
    trading_disabled: bool = False
    consecutive_losses: int = 0
    halt_reason: str = ""


class RiskManager:
    """Gate that runs before every trade. Never bypassed by rounding up to one lot."""

    def __init__(self, state_file: str = Config.RISK_STATE_FILE) -> None:
        self.state_file = state_file
        self.state = self._load_or_reset()

    def _load_or_reset(self) -> DailyRiskState:
        today = ist_now().strftime('%Y-%m-%d')
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                if data.get('trade_date') == today:
                    return DailyRiskState(**data)
            except Exception as e:
                logger.error(f"Couldn't load daily risk state, starting fresh: {e}")
        return DailyRiskState(trade_date=today)

    def _save(self) -> None:
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state.__dict__, f)
        except Exception as e:
            logger.error(f"Couldn't save daily risk state: {e}")

    def record_trade_opened(self) -> None:
        self.state.trades_today += 1
        self._save()

    def record_realized_pnl(self, pnl: float) -> None:
        self.state.realized_pnl += pnl
        self.state.consecutive_losses = self.state.consecutive_losses + 1 if pnl <= 0 else 0

        if self.state.realized_pnl <= -Config.MAX_DAILY_LOSS:
            self.state.trading_disabled = True
            self.state.halt_reason = f"MAX_DAILY_LOSS of Rs.{Config.MAX_DAILY_LOSS:,.0f} reached (realized P&L Rs.{self.state.realized_pnl:,.2f})"
            logger.error(f"{self.state.halt_reason} - disabling new entries for the rest of the session. Existing positions can still be managed/exited.")
        elif self.state.consecutive_losses >= Config.CONSECUTIVE_LOSS_HALT:
            self.state.trading_disabled = True
            self.state.halt_reason = f"{self.state.consecutive_losses} consecutive losing trades (>= CONSECUTIVE_LOSS_HALT of {Config.CONSECUTIVE_LOSS_HALT})"
            logger.error(f"{self.state.halt_reason} - disabling new entries for the rest of the session. Existing positions can still be managed/exited.")
        self._save()

    def check_trade_allowed(self, signal: TradeSignal, open_positions: List[Position]) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        if self.state.trading_disabled:
            reasons.append(self.state.halt_reason or f"daily loss limit reached (realized P&L Rs.{self.state.realized_pnl:,.2f})")

        if len(open_positions) >= Config.MAX_OPEN_POSITIONS:
            reasons.append(f"already at MAX_OPEN_POSITIONS ({Config.MAX_OPEN_POSITIONS})")

        if self.state.trades_today >= Config.MAX_DAILY_TRADES:
            reasons.append(f"already at MAX_DAILY_TRADES ({Config.MAX_DAILY_TRADES})")

        sector_count = sum(1 for p in open_positions if p.sector == signal.stock.sector)
        if sector_count >= Config.MAX_POSITIONS_PER_SECTOR:
            reasons.append(f"already at MAX_POSITIONS_PER_SECTOR ({Config.MAX_POSITIONS_PER_SECTOR}) for {signal.stock.sector}")

        allowed = len(reasons) == 0
        if not allowed:
            logger.warning(f"RISK_REJECTED: {signal.stock.symbol} - " + "; ".join(reasons))
        return allowed, reasons


# ============================================
# MODULE: execution - trading engine
# ============================================

class TradingEngine:
    """Handles option trade execution for both paper and live trading, with confirmed
    fills, partial T1/T2/T3 exits, and EOD square-off."""

    def __init__(self, mode: str = "PAPER", client_manager: Optional[DhanClientManager] = None) -> None:
        self.mode = mode.upper()
        self.client_manager = client_manager
        self.positions: List[Position] = []
        self.closed_trades: List[Position] = []
        self.positions_file = Config.POSITIONS_FILE
        self.trades_file = Config.PAPER_TRADES_FILE if self.mode == "PAPER" else Config.LIVE_TRADES_FILE
        self.risk_manager = RiskManager()
        self._eod_done_today: Optional[date_cls] = None

        if self.mode == "LIVE":
            if not Config.LIVE_TRADING_ENABLED:
                logger.error(
                    "TRADING_MODE is LIVE but LIVE_TRADING_ENABLED is not set to true - refusing to "
                    "place any real orders. Set the LIVE_TRADING_ENABLED environment variable to "
                    "'true' explicitly if you really intend to trade with real money."
                )
            elif not self.client_manager or not self.client_manager.is_connected:
                logger.error("Live trading was selected but Dhan is not connected - live orders will be cancelled.")

        logger.info(f"Trade records for this session will be written to {self.trades_file}")
        self._load_positions()

    @property
    def live_trading_actually_enabled(self) -> bool:
        return self.mode == "LIVE" and Config.LIVE_TRADING_ENABLED

    # ---- Persistence ----

    def _load_positions(self) -> None:
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    data = json.load(f)
                self.positions = [self._dict_to_position(p) for p in data]
                if self.positions:
                    logger.info(f"Loaded {len(self.positions)} open position(s) from a previous session.")
                    logger.warning(
                        "RECOVERY NOTE: this restores local state only. Full broker-position "
                        "reconciliation (comparing against Dhan's actual open positions/orders) is "
                        "not implemented in this phase - verify manually before resuming live trading."
                    )
        except Exception as e:
            logger.error(f"Error loading saved positions: {e}")

    def _save_positions(self) -> None:
        try:
            tmp_path = self.positions_file + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump([self._position_to_dict(p) for p in self.positions], f, default=str)
            os.replace(tmp_path, self.positions_file)  # atomic on POSIX and Windows
        except Exception as e:
            logger.error(f"Error saving positions to disk: {e}")

    def _position_to_dict(self, position: Position) -> Dict[str, Any]:
        d = dict(position.__dict__)
        d['trade_type'] = position.trade_type.value
        d['status'] = position.status.value
        d['entry_time'] = position.entry_time.isoformat()
        d['exit_time'] = position.exit_time.isoformat() if position.exit_time else None
        d['partial_exits'] = [
            {**pe.__dict__, 'exit_time': pe.exit_time.isoformat()} for pe in position.partial_exits
        ]
        return d

    def _dict_to_position(self, data: Dict[str, Any]) -> Position:
        data = dict(data)
        data['trade_type'] = TradeType(data['trade_type'])
        data['status'] = TradeStatus(data['status'])
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        data['exit_time'] = datetime.fromisoformat(data['exit_time']) if data.get('exit_time') else None
        partials = []
        for pe in data.get('partial_exits', []):
            pe = dict(pe)
            pe['exit_time'] = datetime.fromisoformat(pe['exit_time'])
            partials.append(PartialExit(**pe))
        data['partial_exits'] = partials
        known_fields = {f for f in Position.__dataclass_fields__}
        data = {k: v for k, v in data.items() if k in known_fields}
        return Position(**data)

    # ---- Position sizing (item L) ----

    def calculate_position_size(self, entry_price: float, stop_loss: float, lot_size: int,
                                 risk_percent: float = Config.RISK_PER_TRADE) -> Dict[str, float]:
        try:
            risk_amount = Config.CAPITAL * (risk_percent / 100)
            risk_per_share = abs(entry_price - stop_loss)
            logger.info(f"Position sizing: risking {risk_percent:.1f}% of Rs.{Config.CAPITAL:,.0f} = Rs.{risk_amount:,.2f}. Premium risk/unit Rs.{risk_per_share:.2f}.")

            if risk_per_share <= 0 or lot_size <= 0:
                logger.warning("Risk per unit or lot size is zero - can't size this position, skipping.")
                return {'quantity': 0, 'capital_allocated': 0.0, 'risk_amount': risk_amount, 'risk_per_share': risk_per_share}

            lots = int(risk_amount / risk_per_share) // lot_size
            if lots < 1:
                # Take the minimum 1 lot even if that risks more than the budget,
                # rather than skipping the trade - the stop-loss price itself is untouched.
                one_lot_risk = lot_size * risk_per_share
                logger.warning(
                    f"One lot ({lot_size} units) risks Rs.{one_lot_risk:,.2f}, exceeding the "
                    f"Rs.{risk_amount:,.2f} budget for this trade - taking 1 lot anyway with stop-loss intact."
                )
                lots = 1

            # Premium cap: total premium outlay for this position must not exceed
            # PREMIUM_NOTIONAL_CAP_MULTIPLE x the risk budget - risk-based sizing alone can size
            # a huge notional position when risk_per_share (premium distance to stop) is small
            # relative to the premium itself, which is exactly the kind of oversized bet that
            # blows up drawdown even though each SHARE's risk was correctly bounded.
            max_notional = Config.PREMIUM_NOTIONAL_CAP_MULTIPLE * risk_amount
            while lots >= 1 and (lots * lot_size * entry_price) > max_notional:
                lots -= 1
            if lots < 1:
                logger.warning(
                    f"Even 1 lot ({lot_size} units) costs Rs.{lot_size * entry_price:,.2f}, exceeding the "
                    f"premium cap of {Config.PREMIUM_NOTIONAL_CAP_MULTIPLE:.0f}x the risk budget (Rs.{max_notional:,.2f}) - SKIPPING."
                )
                return {'quantity': 0, 'capital_allocated': 0.0, 'risk_amount': risk_amount, 'risk_per_share': risk_per_share}

            quantity = lots * lot_size
            capital_allocated = quantity * entry_price
            logger.info(f"Position size: {quantity} units ({lots} lot(s) of {lot_size}). Capital required: Rs.{capital_allocated:,.2f} (premium cap Rs.{max_notional:,.2f}).")
            return {'quantity': quantity, 'capital_allocated': capital_allocated, 'risk_amount': risk_amount, 'risk_per_share': risk_per_share}
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {'quantity': 0, 'capital_allocated': 0.0, 'risk_amount': 0.0, 'risk_per_share': 0.0}

    # ---- Signal -> order ----

    def execute_signal(self, signal: TradeSignal, data_fetcher: DataFetcher) -> Optional[Position]:
        try:
            option_type = 'CE' if signal.trade_type == TradeType.CALL else 'PE'

            if any(p.underlying_symbol == signal.stock.symbol and p.option_type == option_type
                   and p.status in (TradeStatus.OPEN, TradeStatus.PENDING) for p in self.positions):
                logger.info(f"Already holding/opening a {option_type} position on {signal.stock.symbol} - skipping duplicate.")
                return None

            allowed, reasons = self.risk_manager.check_trade_allowed(signal, [p for p in self.positions if p.status == TradeStatus.OPEN])
            if not allowed:
                return None

            contract = data_fetcher.select_option(signal.stock.symbol, 'UP' if signal.trade_type == TradeType.CALL else 'DOWN')
            if contract is None:
                logger.warning(f"Could not resolve a tradable {option_type} option for {signal.stock.symbol} - skipping this signal.")
                return None

            entry_price = contract.premium
            emergency_stop = entry_price * (1 - Config.OPTION_EMERGENCY_STOP_PERCENT / 100)
            risk = entry_price - emergency_stop
            target_1 = entry_price + risk * Config.TAKE_PROFIT_1
            target_2 = entry_price + risk * Config.TAKE_PROFIT_2
            target_3 = entry_price + risk * Config.TAKE_PROFIT_3

            effective_risk_percent = Config.RISK_PER_TRADE * (signal.risk_multiplier or 1.0)
            sizing = self.calculate_position_size(entry_price, emergency_stop, contract.lot_size, risk_percent=effective_risk_percent)
            if sizing['quantity'] == 0:
                return None

            position = Position(
                stock_symbol=contract.option_symbol, trade_type=signal.trade_type,
                entry_price=entry_price, quantity=int(sizing['quantity']),
                remaining_quantity=int(sizing['quantity']),
                stop_loss=emergency_stop, underlying_stop_loss=signal.stop_loss, underlying_target=signal.target_1,
                target_1=target_1, target_2=target_2, target_3=target_3,
                entry_time=datetime.now(), status=TradeStatus.PENDING,
                order_state=OrderState.SIGNAL.value,
                underlying_symbol=contract.underlying_symbol, option_type=contract.option_type,
                strike_price=contract.strike_price, expiry_date=contract.expiry_date,
                lot_size=contract.lot_size, security_id=contract.security_id,
                underlying_price_at_entry=signal.stock.current_price,
                signal_id=signal.signal_id, sector=signal.stock.sector,
                orb_high=signal.stock.orb_high, orb_low=signal.stock.orb_low,
                orb_range_percent=signal.stock.orb_range, breakout_price=signal.breakout_price,
                pullback_level=signal.pullback_price,
                vwap_at_signal=signal.vwap_at_signal, relative_volume=signal.relative_volume,
                score=signal.score,
                capital_allocated=sizing['capital_allocated'], risk_amount=sizing['risk_amount'],
                risk_per_share=sizing['risk_per_share'],
                regime_at_entry=signal.regime.value, atr_at_entry=signal.atr_at_signal,
                trailing_enabled=signal.trailing_enabled, trailing_stop_underlying=signal.stop_loss,
            )

            logger.info(
                f"Buying a {self.mode} {option_type} on {contract.underlying_symbol} ({signal.stock.sector}): "
                f"{position.quantity} units of {contract.option_symbol} (strike Rs.{contract.strike_price}) "
                f"at Rs.{entry_price:.2f} premium. Underlying stop Rs.{signal.stop_loss:.2f}, "
                f"emergency premium stop Rs.{emergency_stop:.2f}, targets Rs.{target_1:.2f}/Rs.{target_2:.2f}/Rs.{target_3:.2f}."
            )

            if self.live_trading_actually_enabled:
                self._execute_live_trade(position)
            else:
                if self.mode == "LIVE":
                    logger.error(f"LIVE_TRADING_ENABLED is false - NOT placing a real order for {position.stock_symbol}; cancelling instead of silently paper-filling.")
                    position.status = TradeStatus.CANCELLED
                else:
                    self._execute_paper_trade(position)

            if position.status == TradeStatus.CANCELLED:
                return None

            self.positions.append(position)
            self._save_positions()
            self.risk_manager.record_trade_opened()
            return position
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return None

    def _execute_paper_trade(self, position: Position) -> None:
        """PAPER mode NEVER places a real order - this method contains no order_placement call."""
        position.status = TradeStatus.OPEN
        position.order_state = OrderState.ORDER_FILLED.value
        position.filled_price = position.entry_price
        position.filled_quantity = position.quantity
        position.order_id = f"PAPER_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        logger.info(f"Simulated PAPER trade for {position.stock_symbol} - no real money involved, no order sent to any broker.")

    def _execute_live_trade(self, position: Position) -> None:
        """Places a real order and only marks the position OPEN after polling Dhan's
        actual order status and confirming a filled state (item F)."""
        if not self.client_manager:
            position.status = TradeStatus.CANCELLED
            return
        client = self.client_manager.get_client()
        if client is None:
            logger.error(f"Can't place a live order for {position.stock_symbol} - Dhan is not connected.")
            position.status = TradeStatus.CANCELLED
            return
        if not position.security_id:
            logger.error(f"Can't place a live order for {position.stock_symbol} - no resolved security_id.")
            position.status = TradeStatus.CANCELLED
            return

        try:
            position.order_state = OrderState.ORDER_INTENT.value
            logger.info(
                f"ORDER_SUBMITTED: LIVE BUY {position.quantity} x {position.stock_symbol} "
                f"(security_id {position.security_id}) at Rs.{position.entry_price:.2f} (real money)..."
            )
            position.order_state = OrderState.ORDER_SUBMITTED.value

            response = client.place_order(
                security_id=position.security_id,
                exchange_segment=Config.DERIVATIVE_EXCHANGE_SEGMENT,
                transaction_type="BUY",
                quantity=position.quantity,
                order_type="LIMIT",
                product_type=Config.OPTION_PRODUCT_TYPE,
                price=position.entry_price,
                trigger_price=0,
            )
            order_id = self._extract_order_id(response)
            if not order_id:
                logger.error(f"ORDER_REJECTED: {position.stock_symbol} - {self._response_remarks(response)}")
                position.status = TradeStatus.CANCELLED
                position.order_state = OrderState.ORDER_REJECTED.value
                return

            position.order_id = str(order_id)
            position.order_state = OrderState.ORDER_PENDING.value
            self._confirm_fill_or_cancel(position, client)
        except Exception as e:
            logger.error(f"Error placing live order for {position.stock_symbol}: {e}")
            position.status = TradeStatus.CANCELLED
            position.order_state = OrderState.ORDER_REJECTED.value

    @staticmethod
    def _extract_order_id(response: Any) -> Optional[str]:
        if not isinstance(response, dict) or response.get('status') != 'success':
            return None
        data = response.get('data', {})
        if isinstance(data, dict):
            order_id = data.get('orderId') or data.get('order_id')
            return str(order_id) if order_id else None
        return None

    @staticmethod
    def _response_remarks(response: Any) -> str:
        if isinstance(response, dict):
            return str(response.get('remarks') or 'Dhan returned no order ID')
        return str(response)

    def _confirm_fill_or_cancel(self, position: Position, client: Any) -> None:
        """
        Poll Dhan's real order status until it's filled, rejected, or a
        timeout elapses. NOTE: the exact status strings compared against
        Config.FILLED_ORDER_STATUSES / REJECTED_ORDER_STATUSES are the
        SDK's best-effort documented values - verify against a real
        filled order on your account before relying on this in size; see
        the deferred-items list in the module docstring.
        """
        deadline = time.time() + Config.ORDER_FILL_TIMEOUT_SECONDS
        while time.time() < deadline:
            try:
                response = client.get_order_by_id(position.order_id)
                detail = self._extract_order_detail(response)
                status_str = str(detail.get('orderStatus', '')).upper() if detail else ""

                if any(s in status_str for s in Config.FILLED_ORDER_STATUSES):
                    fill_price = self._extract_fill_price(detail) or position.entry_price
                    fill_qty = self._extract_fill_qty(detail) or position.quantity
                    position.status = TradeStatus.OPEN
                    position.order_state = OrderState.ORDER_FILLED.value
                    position.filled_price = fill_price
                    position.filled_quantity = fill_qty
                    logger.info(f"ORDER_FILLED: {position.stock_symbol} - {fill_qty} units at Rs.{fill_price:.2f} (order {position.order_id}).")
                    return

                if any(s in status_str for s in Config.REJECTED_ORDER_STATUSES):
                    logger.error(f"ORDER_REJECTED: {position.stock_symbol} - Dhan reported status '{status_str}' (order {position.order_id}).")
                    position.status = TradeStatus.CANCELLED
                    position.order_state = OrderState.ORDER_REJECTED.value
                    return

            except Exception as e:
                logger.error(f"Error polling order status for {position.order_id}: {e}")

            time.sleep(Config.ORDER_STATUS_POLL_INTERVAL_SECONDS)

        logger.error(
            f"ORDER_TIMEOUT: {position.stock_symbol} order {position.order_id} still not confirmed filled "
            f"after {Config.ORDER_FILL_TIMEOUT_SECONDS}s - cancelling to avoid a phantom position."
        )
        try:
            client.cancel_order(order_id=position.order_id)
        except Exception as e:
            logger.error(f"Couldn't cancel timed-out order {position.order_id}: {e}")
        position.status = TradeStatus.CANCELLED
        position.order_state = OrderState.ORDER_TIMEOUT.value

    @staticmethod
    def _extract_order_detail(response: Any) -> Dict[str, Any]:
        if not isinstance(response, dict) or response.get('status') != 'success':
            return {}
        data = response.get('data', {})
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict):
            return data
        return {}

    @staticmethod
    def _extract_fill_price(detail: Dict[str, Any]) -> float:
        for key in ('averageTradedPrice', 'average_price', 'price', 'tradedPrice'):
            if isinstance(detail, dict) and detail.get(key):
                try:
                    return float(detail[key])
                except (TypeError, ValueError):
                    continue
        return 0.0

    @staticmethod
    def _extract_fill_qty(detail: Dict[str, Any]) -> int:
        for key in ('filledQty', 'filled_quantity', 'quantity', 'tradedQuantity'):
            if isinstance(detail, dict) and detail.get(key):
                try:
                    return int(detail[key])
                except (TypeError, ValueError):
                    continue
        return 0

    # ---- Monitoring / exits ----

    def monitor_positions(self, data_fetcher: DataFetcher) -> None:
        open_positions = [p for p in self.positions if p.status == TradeStatus.OPEN]
        if not open_positions:
            return
        logger.info(f"Checking {len(open_positions)} open position(s)...")

        force_squareoff = is_squareoff_window()
        premiums = data_fetcher.get_option_premiums_batch([p.security_id for p in open_positions])
        for position in self.positions[:]:
            if position.status != TradeStatus.OPEN:
                continue
            current_premium = premiums.get(str(position.security_id), 0.0)
            if current_premium <= 0:
                if self._is_past_expiry(position):
                    logger.warning(
                        f"EXPIRED: {position.stock_symbol} expired on {position.expiry_date} and no "
                        f"premium is quotable any more - closing at Rs.0.00 (worthless)."
                    )
                    self._exit_full(position, 0.0, ExitReason.EXPIRED.value)
                else:
                    logger.warning(f"Couldn't get a current premium for {position.stock_symbol} this cycle - skipping check.")
                continue

            if force_squareoff:
                logger.info(f"EOD_SQUAREOFF: closing {position.stock_symbol} - past {Config.EOD_SQUAREOFF_TIME} IST square-off time.")
                self._exit_full(position, current_premium, ExitReason.EOD_SQUAREOFF.value)
                continue

            self._check_and_apply_exits(position, current_premium, data_fetcher)

    @staticmethod
    def _is_past_expiry(position: Position) -> bool:
        """True once the position's option contract has expired - the broker stops quoting
        a price for it after expiry, so a missing premium on such a position is expected,
        not a transient data glitch."""
        if not position.expiry_date:
            return False
        try:
            expiry = datetime.strptime(position.expiry_date, "%Y-%m-%d").date()
        except ValueError:
            return False
        return ist_now().date() > expiry

    def _update_trailing_stop(self, position: Position, underlying_price: float) -> None:
        """TRENDING regime positions (RegimeSnapshot.trailing_enabled()): once price has
        moved REGIME_TRAIL_TRIGGER_R in favor, ratchet the effective stop behind price by
        REGIME_TRAIL_ATR_MULTIPLE x ATR-at-entry - never loosens, only tightens."""
        if not position.trailing_enabled:
            return
        initial_risk = abs(position.underlying_price_at_entry - position.underlying_stop_loss)
        if initial_risk <= 0:
            return
        trigger_distance = Config.REGIME_TRAIL_TRIGGER_R * initial_risk
        trail_distance = (Config.REGIME_TRAIL_ATR_MULTIPLE * position.atr_at_entry) or initial_risk

        if position.trade_type == TradeType.CALL:
            if (underlying_price - position.underlying_price_at_entry) >= trigger_distance:
                candidate = underlying_price - trail_distance
                if candidate > position.trailing_stop_underlying:
                    position.trailing_stop_underlying = candidate
        else:
            if (position.underlying_price_at_entry - underlying_price) >= trigger_distance:
                candidate = underlying_price + trail_distance
                if candidate < position.trailing_stop_underlying:
                    position.trailing_stop_underlying = candidate

    def _check_and_apply_exits(self, position: Position, current_premium: float, data_fetcher: DataFetcher) -> None:
        underlying_price = data_fetcher.get_current_price(position.underlying_symbol) if position.underlying_symbol else 0.0

        # Primary stop: underlying structural invalidation (or the trailing stop, once armed)
        if underlying_price > 0:
            self._update_trailing_stop(position, underlying_price)
            effective_stop = (
                position.trailing_stop_underlying if position.trailing_enabled and position.trailing_stop_underlying
                else position.underlying_stop_loss
            )
            if position.trade_type == TradeType.CALL and underlying_price <= effective_stop:
                reason = ExitReason.STOP_LOSS.value if effective_stop <= position.underlying_stop_loss else ExitReason.TRAILING_STOP.value
                logger.info(f"{position.stock_symbol}: underlying {position.underlying_symbol} fell to Rs.{underlying_price:.2f} (<= stop Rs.{effective_stop:.2f}, {reason}) - exiting.")
                self._exit_full(position, current_premium, reason)
                return
            if position.trade_type == TradeType.PUT and underlying_price >= effective_stop:
                reason = ExitReason.STOP_LOSS.value if effective_stop >= position.underlying_stop_loss else ExitReason.TRAILING_STOP.value
                logger.info(f"{position.stock_symbol}: underlying {position.underlying_symbol} rose to Rs.{underlying_price:.2f} (>= stop Rs.{effective_stop:.2f}, {reason}) - exiting.")
                self._exit_full(position, current_premium, reason)
                return

            # 1:2+ RR target on the UNDERLYING (single, full exit) - positions opened under the
            # new structure-stop/RR-target scheme carry underlying_target; positions opened
            # before this field existed have it default to 0.0 and fall through to their old
            # premium-based T1/T2/T3 ladder below instead, unaffected by this change.
            # A TRENDING position with trailing enabled is still capped by this target - the
            # trailing stop lets it run BEYOND the trigger distance, but it's not left to ride
            # unbounded on nothing but EOD square-off if it reaches target before trailing arms.
            if position.underlying_target:
                if position.trade_type == TradeType.CALL and underlying_price >= position.underlying_target:
                    logger.info(f"{position.stock_symbol}: underlying {position.underlying_symbol} reached Rs.{underlying_price:.2f} (>= target Rs.{position.underlying_target:.2f}) - exiting.")
                    self._exit_full(position, current_premium, ExitReason.TARGET_1.value)
                    return

                if position.trade_type == TradeType.PUT and underlying_price <= position.underlying_target:
                    logger.info(f"{position.stock_symbol}: underlying {position.underlying_symbol} reached Rs.{underlying_price:.2f} (<= target Rs.{position.underlying_target:.2f}) - exiting.")
                    self._exit_full(position, current_premium, ExitReason.TARGET_1.value)
                    return

        # Secondary/emergency stop: option premium itself (safety net against gaps/decay even if the underlying hasn't hit its stop)
        if current_premium <= position.stop_loss:
            logger.info(f"{position.stock_symbol}: premium dropped to Rs.{current_premium:.2f} (emergency stop Rs.{position.stop_loss:.2f}) - exiting.")
            self._exit_full(position, current_premium, ExitReason.EMERGENCY_STOP.value)
            return

        # Legacy premium-based partial-target ladder (item M) - only for positions opened
        # before underlying_target existed; new positions are fully handled above.
        if not position.underlying_target:
            if not position.t3_hit and current_premium >= position.target_3:
                self._apply_target_exit(position, current_premium, 3, data_fetcher)
            elif not position.t2_hit and current_premium >= position.target_2:
                self._apply_target_exit(position, current_premium, 2, data_fetcher)
            elif not position.t1_hit and current_premium >= position.target_1:
                self._apply_target_exit(position, current_premium, 1, data_fetcher)

    def _apply_target_exit(self, position: Position, price: float, target_num: int, data_fetcher: DataFetcher) -> None:
        total_lots = position.quantity // position.lot_size if position.lot_size else 0

        if total_lots <= 1:
            policy = Config.SINGLE_LOT_FALLBACK
            if policy == "FULL_AT_T1" or target_num == 3:
                logger.info(f"{position.stock_symbol}: single-lot position hit target {target_num} - SINGLE_LOT_FALLBACK={policy}, exiting in full.")
                self._exit_full(position, price, getattr(ExitReason, f'TARGET_{target_num}').value)
            elif policy == "HOLD_TO_T2" and target_num < 2:
                logger.info(f"{position.stock_symbol}: single-lot position hit target {target_num}, but SINGLE_LOT_FALLBACK=HOLD_TO_T2 - holding.")
            elif policy == "HOLD_TO_T3" and target_num < 3:
                logger.info(f"{position.stock_symbol}: single-lot position hit target {target_num}, but SINGLE_LOT_FALLBACK=HOLD_TO_T3 - holding.")
            else:
                self._exit_full(position, price, getattr(ExitReason, f'TARGET_{target_num}').value)
            setattr(position, f't{target_num}_hit', True)
            return

        exit_pct = {1: Config.T1_EXIT_PERCENT, 2: Config.T2_EXIT_PERCENT, 3: Config.T3_EXIT_PERCENT}[target_num]
        exit_lots = max(1, round(total_lots * (exit_pct / 100)))
        exit_lots = min(exit_lots, position.remaining_quantity // position.lot_size)
        exit_qty = exit_lots * position.lot_size

        if exit_qty <= 0:
            setattr(position, f't{target_num}_hit', True)
            return

        pnl = (price - position.entry_price) * exit_qty
        position.partial_exits.append(PartialExit(quantity=exit_qty, exit_price=price, exit_time=datetime.now(),
                                                    exit_reason=f"TARGET_{target_num}", pnl=pnl))
        position.remaining_quantity -= exit_qty
        position.pnl += pnl
        setattr(position, f't{target_num}_hit', True)

        logger.info(f"TARGET_HIT: {position.stock_symbol} target {target_num} - closed {exit_qty} units at Rs.{price:.2f}, partial P&L Rs.{pnl:,.2f}. {position.remaining_quantity} units remain.")

        if self.live_trading_actually_enabled:
            self._place_exit_order(position, exit_qty)

        if target_num == 1:
            position.stop_loss = max(position.stop_loss, position.entry_price)  # move to breakeven after T1
            logger.info(f"{position.stock_symbol}: stop moved to breakeven (Rs.{position.entry_price:.2f}) after target 1.")

        if position.remaining_quantity <= 0:
            self._finalize_close(position, price, f"TARGET_{target_num}")
            self._save_positions()
        else:
            self._save_positions()

    def _exit_full(self, position: Position, price: float, reason: str) -> None:
        if position.remaining_quantity > 0:
            pnl = (price - position.entry_price) * position.remaining_quantity
            position.partial_exits.append(PartialExit(quantity=position.remaining_quantity, exit_price=price,
                                                        exit_time=datetime.now(), exit_reason=reason, pnl=pnl))
            position.pnl += pnl
            if self.live_trading_actually_enabled:
                self._place_exit_order(position, position.remaining_quantity)
            position.remaining_quantity = 0
        self._finalize_close(position, price, reason)
        self._save_positions()

    def _place_exit_order(self, position: Position, quantity: int) -> None:
        client = self.client_manager.get_client() if self.client_manager else None
        if client is None:
            return
        if not position.security_id:
            logger.error(f"Can't place an exit order for {position.stock_symbol} - no resolved security_id.")
            return
        try:
            logger.info(f"Sending a LIVE SELL order to Dhan to close {quantity} units of {position.stock_symbol}...")
            response = client.place_order(
                security_id=position.security_id,
                exchange_segment=Config.DERIVATIVE_EXCHANGE_SEGMENT,
                transaction_type="SELL",
                quantity=quantity,
                order_type="MARKET",
                product_type=Config.OPTION_PRODUCT_TYPE,
                price=0,
                trigger_price=0,
            )
            order_id = self._extract_order_id(response)
            if order_id:
                logger.info(f"Exit order for {quantity} units of {position.stock_symbol} placed successfully (order {order_id}).")
            else:
                logger.error(f"Exit order for {position.stock_symbol} failed - {self._response_remarks(response)}. Retry/alert policy needed manually.")
        except Exception as e:
            logger.error(f"Error placing exit order for {position.stock_symbol}: {e}")

    def _finalize_close(self, position: Position, price: float, reason: str) -> None:
        position.exit_price = price
        position.exit_time = datetime.now()
        position.exit_reason = reason
        position.status = TradeStatus.CLOSED

        self.closed_trades.append(position)
        self.positions.remove(position)
        self.risk_manager.record_realized_pnl(position.pnl)
        self._log_trade(position)

        result = "profit" if position.pnl >= 0 else "loss"
        logger.info(f"POSITION_CLOSED: {position.option_type} on {position.underlying_symbol} ({reason}) - {result} of Rs.{abs(position.pnl):,.2f} total.")

    def _log_trade(self, position: Position) -> None:
        try:
            holding_minutes = round((position.exit_time - position.entry_time).total_seconds() / 60, 1) if position.exit_time else 0.0
            pnl_percent = (position.pnl / position.capital_allocated) * 100 if position.capital_allocated else 0.0

            trade_data = {
                'trade_id': position.order_id, 'signal_id': position.signal_id, 'mode': self.mode,
                'date': position.entry_time.date().isoformat(),
                'underlying': position.underlying_symbol, 'sector': position.sector,
                'option_symbol': position.stock_symbol, 'option_type': position.option_type,
                'strike_price': position.strike_price, 'expiry_date': position.expiry_date,
                'lot_size': position.lot_size, 'score': position.score,
                'underlying_price_at_entry': position.underlying_price_at_entry,
                'orb_high': position.orb_high, 'orb_low': position.orb_low,
                'orb_range_percent': round(position.orb_range_percent, 3),
                'breakout_price': position.breakout_price, 'pullback_level': position.pullback_level,
                'vwap_at_signal': position.vwap_at_signal, 'relative_volume': round(position.relative_volume, 2),
                'premium_entry_requested': position.entry_price, 'premium_entry_filled': position.filled_price,
                'slippage': round(position.filled_price - position.entry_price, 2),
                'underlying_stop_loss': position.underlying_stop_loss, 'underlying_target': position.underlying_target,
                'premium_emergency_stop': position.stop_loss,
                'premium_target_1': position.target_1, 'premium_target_2': position.target_2, 'premium_target_3': position.target_3,
                'quantity': position.quantity, 'filled_quantity': position.filled_quantity,
                'capital_allocated': round(position.capital_allocated, 2), 'risk_amount': round(position.risk_amount, 2),
                'risk_per_unit': round(position.risk_per_share, 2),
                'entry_time': position.entry_time, 'exit_time': position.exit_time,
                'final_exit_price': position.exit_price, 'exit_reason': position.exit_reason,
                'holding_minutes': holding_minutes,
                'partial_exit_count': len(position.partial_exits),
                'gross_pnl': round(position.pnl, 2), 'net_pnl': round(position.pnl, 2),  # fees not modeled - see report
                'pnl_percent': round(pnl_percent, 2), 'r_multiple': round(position.pnl / position.risk_amount, 2) if position.risk_amount else 0.0,
                'status': position.status.value, 'order_id': position.order_id, 'order_state': position.order_state,
                'regime_at_entry': position.regime_at_entry, 'atr_at_entry': round(position.atr_at_entry, 2),
                'trailing_enabled': position.trailing_enabled, 'trailing_stop_underlying': round(position.trailing_stop_underlying, 2),
            }
            df = pd.DataFrame([trade_data])
            if os.path.exists(self.trades_file):
                existing = pd.read_csv(self.trades_file)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_csv(self.trades_file, index=False)
            logger.info(f"Trade record for {position.stock_symbol} written to {self.trades_file}")
        except Exception as e:
            logger.error(f"Error logging trade to CSV: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        try:
            total = len(self.closed_trades)
            if total == 0:
                return {'total_trades': 0}
            wins = [t for t in self.closed_trades if t.pnl > 0]
            losses = [t for t in self.closed_trades if t.pnl < 0]
            total_pnl = sum(t.pnl for t in self.closed_trades)
            gross_profit = sum(t.pnl for t in wins)
            gross_loss = abs(sum(t.pnl for t in losses))

            max_consec_wins = max_consec_losses = cur_w = cur_l = 0
            for t in self.closed_trades:
                if t.pnl > 0:
                    cur_w += 1; cur_l = 0
                elif t.pnl < 0:
                    cur_l += 1; cur_w = 0
                else:
                    cur_w = cur_l = 0
                max_consec_wins = max(max_consec_wins, cur_w)
                max_consec_losses = max(max_consec_losses, cur_l)

            return {
                'total_trades': total, 'winning_trades': len(wins), 'losing_trades': len(losses),
                'win_rate': (len(wins) / total) * 100, 'total_pnl': total_pnl, 'avg_pnl': total_pnl / total,
                'gross_profit': gross_profit, 'gross_loss': gross_loss,
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0.0,
                'avg_win': (sum(t.pnl for t in wins) / len(wins)) if wins else 0.0,
                'avg_loss': (sum(t.pnl for t in losses) / len(losses)) if losses else 0.0,
                'max_win': max((t.pnl for t in self.closed_trades), default=0.0),
                'max_loss': min((t.pnl for t in self.closed_trades), default=0.0),
                'max_consecutive_wins': max_consec_wins, 'max_consecutive_losses': max_consec_losses,
            }
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return {}


# ============================================
# MODULE: orchestrator
# ============================================

class TradingSystem:
    """Main orchestrator. Single loop with different effective cadences for position
    monitoring vs full signal scanning (see module docstring for the deferred true
    multi-threaded fast/medium/slow split)."""

    def __init__(self, mode: str = "PAPER") -> None:
        self.mode = mode.upper()
        logger.info(f"Setting up the trading system in {self.mode} mode...")

        self.client_manager = DhanClientManager(Config.DHAN_CLIENT_ID, Config.DHAN_ACCESS_TOKEN)
        self.data_fetcher = DataFetcher(self.client_manager)
        self.strategy_engine = StrategyEngine()
        self.trading_engine = TradingEngine(self.mode, self.client_manager)
        self.running = False
        self.trading_thread: Optional[threading.Thread] = None
        self._last_scan_time = 0.0
        self.last_regime: RegimeSnapshot = RegimeSnapshot()

        logger.info(f"Trading system is ready and running in {self.mode} mode.")
        if self.mode == "LIVE":
            logger.info(f"LIVE_TRADING_ENABLED = {Config.LIVE_TRADING_ENABLED}")

    def start(self) -> None:
        if self.running:
            logger.warning("Trading system is already running - ignoring duplicate start request.")
            return
        self.running = True
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        logger.info(f"Trading system started in {self.mode} mode.")

    def stop(self) -> None:
        logger.info("Stopping the trading system...")
        self.running = False
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        logger.info("Trading system stopped.")

    def _trading_loop(self) -> None:
        while self.running:
            try:
                if not is_market_open():
                    now = ist_now()
                    logger.info(f"Market is closed right now ({now.strftime('%H:%M')} IST; NSE trades 09:15-15:30 IST on weekdays) - checking again shortly.")
                    time.sleep(Config.SIGNAL_SCAN_INTERVAL_SECONDS)
                    continue

                # FAST cadence: always monitor/exit open positions, including EOD squareoff.
                self.trading_engine.monitor_positions(self.data_fetcher)

                # MEDIUM cadence: full sector/stock scan for new signals, only within
                # the entry window and only every SIGNAL_SCAN_INTERVAL_SECONDS.
                now_ts = time.time()
                if is_entry_window() and (now_ts - self._last_scan_time) >= Config.SIGNAL_SCAN_INTERVAL_SECONDS:
                    self._last_scan_time = now_ts
                    self._scan_for_signals()
                elif not is_entry_window():
                    logger.info(f"Outside the entry window ({Config.ENTRY_START_TIME}-{Config.ENTRY_END_TIME} IST) - monitoring existing positions only.")

                metrics = self.trading_engine.get_performance_metrics()
                if metrics.get('total_trades', 0) > 0:
                    logger.info(f"Performance so far: {metrics['total_trades']} trade(s), win rate {metrics['win_rate']:.1f}%, total P&L Rs.{metrics['total_pnl']:,.2f}.")

                time.sleep(Config.POSITION_MONITOR_INTERVAL_SECONDS)
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(5)

    def _scan_for_signals(self) -> None:
        logger.info("--- Starting a new signal scan ---")
        already_traded = {p.underlying_symbol for p in self.trading_engine.positions}
        scanned_stocks: List[Stock] = []

        # NIFTY/BANKNIFTY are scanned independently of the sector-heatmap funnel -
        # each runs its own direct-breakout ORB sequence.
        for idx_symbol in Config.TRADING_INDICES:
            if idx_symbol in already_traded:
                continue
            stock = self.data_fetcher.get_index_stock(idx_symbol)
            if stock:
                scanned_stocks.append(stock)

        heatmap_df = self.data_fetcher.get_sector_heatmap()
        if heatmap_df.empty:
            logger.warning("No heatmap data available this cycle.")
        else:
            scanned_stocks.extend(self.strategy_engine.gather_sector_universe(heatmap_df, self.data_fetcher, already_traded))

        # One regime read per scan cycle, from the FULL scanned universe (breadth needs
        # every symbol just fetched above) - shared by every breakout evaluated this cycle.
        regime = (
            self.data_fetcher.get_regime_snapshot(scanned_stocks)
            if Config.ENABLE_REGIME_ADAPTIVE else RegimeSnapshot()
        )
        if Config.ENABLE_REGIME_ADAPTIVE and not regime.allows_new_entries():
            logger.info(f"Regime {regime.regime.value} does not allow new entries this cycle - monitoring existing positions only.")
        self.last_regime = regime

        signals = self.strategy_engine.evaluate_universe(scanned_stocks, regime)

        signals.sort(key=lambda s: s.score, reverse=True)
        for signal in signals:
            open_count = len([p for p in self.trading_engine.positions if p.status == TradeStatus.OPEN])
            if open_count >= Config.MAX_OPEN_POSITIONS:
                logger.info(f"Already holding the maximum of {Config.MAX_OPEN_POSITIONS} open positions - not opening any more this scan.")
                break
            self.trading_engine.execute_signal(signal, self.data_fetcher)
        logger.info("--- Signal scan complete ---")

    def get_status(self) -> Dict[str, Any]:
        metrics = self.trading_engine.get_performance_metrics()
        return {
            'mode': self.mode, 'running': self.running,
            'market_open': is_market_open(), 'ist_time': ist_now().strftime('%Y-%m-%d %H:%M:%S'),
            'entry_window': is_entry_window(), 'squareoff_window': is_squareoff_window(),
            'open_positions': len([p for p in self.trading_engine.positions if p.status == TradeStatus.OPEN]),
            'trades_today': self.trading_engine.risk_manager.state.trades_today,
            'daily_pnl': self.trading_engine.risk_manager.state.realized_pnl,
            'trading_disabled': self.trading_engine.risk_manager.state.trading_disabled,
            'dhan_connected': self.client_manager.is_connected,
            'performance': metrics,
            'regime_adaptive_enabled': Config.ENABLE_REGIME_ADAPTIVE,
            'regime': self.last_regime.regime.value,
            'regime_adx': self.last_regime.adx, 'regime_atr': self.last_regime.atr,
            'regime_atr_percentile': self.last_regime.atr_percentile,
            'regime_vix': self.last_regime.india_vix, 'regime_breadth': self.last_regime.breadth_ratio,
            'regime_vwap_distance_pct': self.last_regime.vwap_distance_percent,
        }


# ============================================
# MAIN
# ============================================

def main() -> None:
    print("=" * 60)
    print("NSE Sector Heatmap Options Trading System (Hardened)")
    print("=" * 60)
    print("1. Paper Trading Mode")
    print("2. Live Trading Mode (dhanhq)")
    print("3. View Performance")
    print("4. Exit")
    print("=" * 60)

    system: Optional[TradingSystem] = None
    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()
            if choice == '1':
                print("\nStarting Paper Trading...")
                if system:
                    system.stop()
                system = TradingSystem(mode="PAPER")
                system.start()
                input("\nPress Enter to stop trading...")
                system.stop()
            elif choice == '2':
                print("\n" + "!" * 60)
                print("WARNING: Live trading will use real money to buy options!")
                print("!" * 60)
                confirm = input("Are you sure? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    if not DHANHQ_AVAILABLE:
                        print("\ndhanhq not available. Please install: pip install dhanhq")
                        continue
                    if Config.DHAN_CLIENT_ID == 'your_client_id' or Config.DHAN_ACCESS_TOKEN == 'your_access_token':
                        print("\nSet DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables first.")
                        continue
                    if not Config.LIVE_TRADING_ENABLED:
                        print("\nLIVE_TRADING_ENABLED is not 'true' - refusing to start live trading.")
                        print("Set: export LIVE_TRADING_ENABLED=true")
                        continue
                    print("\nStarting Live Trading...")
                    if system:
                        system.stop()
                    system = TradingSystem(mode="LIVE")
                    system.start()
                    input("\nPress Enter to stop trading...")
                    system.stop()
                else:
                    print("Live trading cancelled.")
            elif choice == '3':
                print("\n" + "=" * 60)
                print("Trading Performance")
                print("=" * 60)
                for label, path in (("Paper", Config.PAPER_TRADES_FILE), ("Live", Config.LIVE_TRADES_FILE)):
                    print(f"\n--- {label} trades ({path}) ---")
                    if not os.path.exists(path):
                        print("No trades recorded yet.")
                        continue
                    df = pd.read_csv(path)
                    print(f"Total Trades: {len(df)}")
                    if not df.empty:
                        print(f"  Total P&L: {df['net_pnl'].sum():.2f}")
                        print(f"  Win Rate: {(len(df[df['net_pnl'] > 0]) / len(df)) * 100:.2f}%")
            elif choice == '4':
                print("\nExiting system...")
                if system:
                    system.stop()
                break
            else:
                print("Invalid option. Please select 1-4.")
        except KeyboardInterrupt:
            print("\n\nSystem interrupted. Exiting...")
            if system:
                system.stop()
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()