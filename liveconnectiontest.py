"""
Regression tests for the raw-dhanhq rewrite. Confirms the strategy/risk
logic (unchanged from the hardening pass) still works, and that the new
instrument-master / option-chain-based selection logic works correctly
against mocked dhanhq responses shaped exactly like the real SDK's
verified response envelopes.
"""
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(0, '.')

from orbwheatmap import (
    Config, StrategyEngine, TradingEngine, DataFetcher, DhanClientManager,
    DhanDataFetcher, DhanInstrumentMaster, OptionSelector, Stock, TradeType,
    TradeStatus, BreakoutPhase, TradeSignal, OptionContract, ist_now,
)

passed, failed = [], []


def check(name, condition, detail=""):
    if condition:
        passed.append(name)
        print(f"  [PASS] {name}")
    else:
        failed.append(name)
        print(f"  [FAIL] {name}  {detail}")


def cleanup():
    for f in (Config.PAPER_TRADES_FILE, Config.LIVE_TRADES_FILE, Config.POSITIONS_FILE,
              Config.RISK_STATE_FILE, 'trading_system.log', Config.INSTRUMENT_MASTER_CACHE_FILE):
        if os.path.exists(f):
            os.remove(f)


cleanup()

print("\n--- Sector classification (unchanged logic) ---")
heatmap = pd.DataFrame([
    {'sector': 'NIFTY BANK', 'change_percent': 0.30, 'volatility': 1.0},
    {'sector': 'NIFTY AUTO', 'change_percent': -0.05, 'volatility': 1.0},
    {'sector': 'NIFTY METAL', 'change_percent': -0.20, 'volatility': 1.0},
])
se = StrategyEngine()
positive, negative = se.analyze_sector_heatmap(heatmap)
check("'less negative' sector not treated as positive", 'NIFTY AUTO' not in positive)
check("genuinely negative sector correctly classified", negative == ['NIFTY METAL'])

print("\n--- Breakout/pullback state machine (unchanged logic) ---")
se2 = StrategyEngine()


def make_stock(price):
    return Stock(symbol='RELIANCE', name='RELIANCE', sector='NIFTY ENERGY', current_price=price,
                 change_percent=0.0, volume=800000, orb_high=2450.0, orb_low=2420.0,
                 orb_range=1.24, orb_complete=True)


check("no signal on first touch beyond ORB", se2.update_breakout_state(make_stock(2451.5)) is None)
se2.update_breakout_state(make_stock(2465.0))  # displacement
sig = se2.update_breakout_state(make_stock(2451.0))  # pullback
check("signal fires only after breakout + displacement + retest", sig is not None)
check("repeated scan of same setup does not duplicate", se2.update_breakout_state(make_stock(2451.2)) is None)

print("\n--- DhanClientManager connects via DhanContext + verifies with get_fund_limits ---")
import orbwheatmap as ts_mod

fake_context_cls = MagicMock()
fake_client = MagicMock()
fake_client.get_fund_limits.return_value = {'status': 'success', 'data': {}}
fake_dhanhq_cls = MagicMock(return_value=fake_client)
ts_mod.DhanContext_class = fake_context_cls
ts_mod.dhanhq_class = fake_dhanhq_cls
ts_mod.DHANHQ_AVAILABLE = True

mgr = DhanClientManager("test_id", "test_token")
check("client manager connects when get_fund_limits reports success", mgr.is_connected)

fake_client2 = MagicMock()
fake_client2.get_fund_limits.return_value = {'status': 'failure', 'remarks': 'bad token'}
ts_mod.dhanhq_class = MagicMock(return_value=fake_client2)
mgr2 = DhanClientManager("test_id", "bad_token")
check("client manager correctly refuses when get_fund_limits reports failure", not mgr2.is_connected)

print("\n--- Chart response parsing matches dhanhq's real envelope shape ---")
ts_mod.dhanhq_class = fake_dhanhq_cls
ts_mod.dhanhq_class.NSE = 'NSE_EQ'
ts_mod.dhanhq_class.NSE_FNO = 'NSE_FNO'
ts_mod.dhanhq_class.BUY = 'BUY'
ts_mod.dhanhq_class.SELL = 'SELL'
ts_mod.dhanhq_class.MARKET = 'MARKET'
ts_mod.dhanhq_class.LIMIT = 'LIMIT'

mgr3 = DhanClientManager.__new__(DhanClientManager)
mgr3.client = fake_client

today = ist_now()
market_open = today.replace(hour=9, minute=15, second=0, microsecond=0)
epoch_timestamps = [int((market_open + timedelta(minutes=5 * i)).timestamp()) for i in range(4)]
chart_resp = {
    'status': 'success', 'remarks': '',
    'data': {
        'open': [2400, 2405, 2420, 2425], 'high': [2410, 2450, 2430, 2460],
        'low': [2395, 2400, 2415, 2420], 'close': [2405, 2420, 2425, 2451],
        'volume': [50000, 60000, 55000, 70000], 'timestamp': epoch_timestamps,
    }
}
fake_client.intraday_minute_data.return_value = chart_resp

instruments = DhanInstrumentMaster.__new__(DhanInstrumentMaster)
instruments.df = pd.DataFrame([
    {'SEM_TRADING_SYMBOL': 'RELIANCE', 'SEM_EXM_EXCH_ID': 'NSE', 'SEM_SMST_SECURITY_ID': '2885', 'SEM_OPTION_TYPE': None},
])
instruments._loaded_date = ist_now().date()
instruments.ensure_loaded = lambda: True

fetcher = DhanDataFetcher(mgr3, instruments)
ohlc = fetcher.get_intraday_ohlc('RELIANCE')
check("chart response envelope parsed correctly into a usable DataFrame",
      not ohlc.empty and list(ohlc['Close']) == [2405, 2420, 2425, 2451], f"got {ohlc}")

print("\n--- Option chain selection uses real available strikes, not a hard-coded step ---")
fake_client.expiry_list.return_value = {'status': 'success', 'data': {'data': ['2026-09-25']}}
fake_client.option_chain.return_value = {
    'status': 'success', 'data': {
        'data': {
            'last_price': 2451.0,
            'oc': {
                '2440.0': {'ce': {'last_price': 55.0, 'oi': 50000, 'volume': 20000, 'top_bid_price': 54.8, 'top_ask_price': 55.2},
                           'pe': {}},
                '2460.0': {'ce': {'last_price': 42.5, 'oi': 100000, 'volume': 50000, 'top_bid_price': 42.4, 'top_ask_price': 42.6},
                           'pe': {}},
                '2480.0': {'ce': {'last_price': 30.0, 'oi': 30000, 'volume': 15000, 'top_bid_price': 29.8, 'top_ask_price': 30.2},
                           'pe': {}},
            }
        }
    }
}
fake_client.ticker_data.return_value = {'status': 'success', 'data': {'data': {'NSE_EQ': {'2885': {'last_price': 2461.0}}}}}

instruments.df = pd.DataFrame([
    {'SEM_TRADING_SYMBOL': 'RELIANCE', 'SEM_EXM_EXCH_ID': 'NSE', 'SEM_SMST_SECURITY_ID': '2885', 'SEM_OPTION_TYPE': None},
    {'SEM_TRADING_SYMBOL': 'RELIANCE-Sep2026-2460-CE', 'SEM_EXM_EXCH_ID': 'NSE', 'SEM_SMST_SECURITY_ID': '99001',
     'SEM_OPTION_TYPE': 'CE', 'SEM_EXPIRY_DATE': '2026-09-25', 'SEM_STRIKE_PRICE': 2460.0, 'SEM_LOT_UNITS': 250},
    {'SEM_TRADING_SYMBOL': 'RELIANCE-Sep2026-2480-CE', 'SEM_EXM_EXCH_ID': 'NSE', 'SEM_SMST_SECURITY_ID': '99002',
     'SEM_OPTION_TYPE': 'CE', 'SEM_EXPIRY_DATE': '2026-09-25', 'SEM_STRIKE_PRICE': 2480.0, 'SEM_LOT_UNITS': 250},
    {'SEM_TRADING_SYMBOL': 'RELIANCE-Sep2026-2440-CE', 'SEM_EXM_EXCH_ID': 'NSE', 'SEM_SMST_SECURITY_ID': '99003',
     'SEM_OPTION_TYPE': 'CE', 'SEM_EXPIRY_DATE': '2026-09-25', 'SEM_STRIKE_PRICE': 2440.0, 'SEM_LOT_UNITS': 250},
])

selector = OptionSelector(fetcher)
contract = selector.select('RELIANCE', 'UP')
check("ATM correctly found as nearest listed strike to LTP (2460, since underlying is 2461)",
      contract is not None and contract.strike_price == 2460.0, f"got {contract.strike_price if contract else None}")
check("selected contract resolved to the correct real security_id (99001)",
      contract is not None and contract.security_id == '99001', f"got {contract.security_id if contract else None}")

print("\n--- Illiquid ATM correctly falls back to a liquid neighboring strike ---")
fake_client.option_chain.return_value = {
    'status': 'success', 'data': {'data': {'last_price': 2451.0, 'oc': {
        '2440.0': {'ce': {'last_price': 55.0, 'oi': 50000, 'volume': 20000, 'top_bid_price': 54.8, 'top_ask_price': 55.2}, 'pe': {}},
        '2460.0': {'ce': {'last_price': 42.5, 'oi': 0, 'volume': 0, 'top_bid_price': 42.4, 'top_ask_price': 42.6}, 'pe': {}},  # illiquid ATM
        '2480.0': {'ce': {'last_price': 30.0, 'oi': 30000, 'volume': 15000, 'top_bid_price': 29.8, 'top_ask_price': 30.2}, 'pe': {}},
    }}}
}
Config.MIN_OPTION_VOLUME = 1000
contract2 = selector.select('RELIANCE', 'UP')
Config.MIN_OPTION_VOLUME = 0
check("illiquid ATM rejected and selector fell back to ATM+1 (2480)",
      contract2 is not None and contract2.strike_price == 2480.0, f"got {contract2.strike_price if contract2 else None}")

print("\n--- Position sizing / risk manager (unchanged logic) ---")
te = TradingEngine(mode="PAPER", client_manager=None)
Config.CAPITAL = 10000.0
sizing = te.calculate_position_size(entry_price=50.0, stop_loss=10.0, lot_size=250)
check("one lot exceeding risk budget is skipped, not rounded up", sizing['quantity'] == 0)
Config.CAPITAL = 1000000.0

cleanup()

print("\n" + "=" * 70)
print(f"RESULT: {len(passed)} passed, {len(failed)} failed")
print("=" * 70)
if failed:
    for f in failed:
        print(f"  - {f}")
sys.exit(0 if not failed else 1)