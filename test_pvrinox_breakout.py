"""
Regression test for the PVRINOX breakout-state-transition bug:
  - ORB high 1166.90, stock observed already ~4.97% beyond it (1224.90) -
    must arm PULLBACK immediately on that SAME tick, not stay stuck logging
    "waiting for displacement" scan after scan.
  - State must persist across scan cycles even when ORB is recomputed with
    tiny floating-point differences between scans (no spurious reset back
    to WAITING_FOR_BREAKOUT).
  - A later retracement into the pullback zone must then confirm the signal.
"""
import sys
sys.path.insert(0, '.')

from orbwheatmap import Config, StrategyEngine, Stock, BreakoutPhase

passed, failed = [], []


def check(name, condition, detail=""):
    if condition:
        passed.append(name)
        print(f"  [PASS] {name}")
    else:
        failed.append(name)
        print(f"  [FAIL] {name}  {detail}")


ORB_HIGH = 1166.90
ORB_LOW = 1148.00  # ~1.65% range - within MIN/MAX_ORB_RANGE


def make_stock(price: float, orb_high: float = ORB_HIGH, orb_low: float = ORB_LOW) -> Stock:
    return Stock(
        symbol='PVRINOX', name='PVRINOX', sector='NIFTY MEDIA', current_price=price,
        change_percent=0.0, volume=900000, orb_high=orb_high, orb_low=orb_low,
        orb_range=((orb_high - orb_low) / orb_low) * 100, orb_complete=True,
    )


print(f"\nConfig: MIN_BREAKOUT_DISPLACEMENT_PERCENT={Config.MIN_BREAKOUT_DISPLACEMENT_PERCENT}%, "
      f"PULLBACK_ZONE_PERCENT={Config.PULLBACK_ZONE_PERCENT}%, BREAKOUT_BUFFER_PERCENT={Config.BREAKOUT_BUFFER_PERCENT}%")

print("\n--- Scenario 1: stock already far beyond ORB on the very first tick we see it ---")
se = StrategyEngine()
stock1 = make_stock(1224.90)  # ~4.97% above ORB high 1166.90
signal1 = se.update_breakout_state(stock1)
state1 = se._breakout_states['PVRINOX']

check("no signal yet (pullback just armed, not confirmed)", signal1 is None)
check(
    "state advanced past BREAKOUT_CONFIRMED to AWAITING_PULLBACK in the SAME tick "
    "(armed immediately, not stuck waiting for another scan)",
    state1.phase == BreakoutPhase.AWAITING_PULLBACK,
    f"got {state1.phase}",
)
check("direction correctly detected as UP", state1.direction == 'UP')
check("extreme_price tracks the max price seen (1224.90)", state1.extreme_price == 1224.90)

print("\n--- Scenario 2: state persists across scans despite tiny ORB float noise ---")
# Simulate the ORB being recomputed on a later scan with a floating-point-noise-sized
# difference (e.g. 1166.90 vs 1166.899999999) - must NOT reset the sequence.
stock2 = make_stock(1230.00, orb_high=ORB_HIGH - 1e-9, orb_low=ORB_LOW + 1e-9)
signal2 = se.update_breakout_state(stock2)
state2 = se._breakout_states['PVRINOX']
check(
    "state survives a floating-point-noise-sized ORB recomputation (still AWAITING_PULLBACK, not reset)",
    state2.phase == BreakoutPhase.AWAITING_PULLBACK,
    f"got {state2.phase}",
)
check("no signal yet (still just watching for retest)", signal2 is None)

print("\n--- Scenario 3: a GENUINE ORB change (not float noise) still correctly resets ---")
se_reset = StrategyEngine()
se_reset.update_breakout_state(make_stock(1224.90))
stock_changed = make_stock(1224.90, orb_high=ORB_HIGH + 5.0, orb_low=ORB_LOW)  # real change
se_reset.update_breakout_state(stock_changed)
state_changed = se_reset._breakout_states['PVRINOX']
check(
    "a real ORB level change resets the sequence back to a fresh state, not silently ignored",
    state_changed.orb_high == ORB_HIGH + 5.0,
    f"got orb_high={state_changed.orb_high}",
)

print("\n--- Scenario 4: retracement into the pullback zone confirms the signal ---")
# Price pulls back toward the ORB high, inside the (still-armed) pullback zone.
stock3 = make_stock(1168.50)  # just above ORB high, inside the retest zone
signal3 = se.update_breakout_state(stock3)
state3 = se._breakout_states['PVRINOX']
check("signal fires once price retraces into the pullback zone and holds", signal3 is not None)
check("sequence is now terminal (PULLBACK_CONFIRMED)", state3.phase == BreakoutPhase.PULLBACK_CONFIRMED)
if signal3:
    check("signal direction is CALL (bullish breakout)", signal3.trade_type.value == 'CALL')

print("\n--- Scenario 5: repeated scan of the same confirmed setup does not duplicate ---")
signal4 = se.update_breakout_state(make_stock(1168.60))
check("no duplicate signal on the next scan of an already-confirmed sequence", signal4 is None)

print("\n" + "=" * 70)
print(f"RESULT: {len(passed)} passed, {len(failed)} failed")
print("=" * 70)
if failed:
    for f in failed:
        print(f"  - {f}")
sys.exit(0 if not failed else 1)
