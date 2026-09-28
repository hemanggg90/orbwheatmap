"""
One-off/rerunnable export of every paper trade (open positions from positions.json
plus any already-closed trades from paper_trades.csv) into a single, complete CSV.

Run: python export_paper_trades.py
Output: paper_trades_full.csv
"""
import json
import os
from datetime import datetime

import pandas as pd

POSITIONS_FILE = "positions.json"
PAPER_TRADES_FILE = "paper_trades.csv"
OUTPUT_FILE = "paper_trades_full.csv"

COLUMNS = [
    "order_id", "signal_id", "status", "underlying_symbol", "sector",
    "stock_symbol", "option_type", "trade_type", "strike_price", "expiry_date",
    "lot_size", "score",
    "underlying_price_at_entry", "orb_high", "orb_low", "orb_range_percent",
    "breakout_price", "pullback_level",
    "entry_price", "filled_price", "quantity", "filled_quantity", "remaining_quantity",
    "underlying_stop_loss", "underlying_target", "stop_loss", "target_1", "target_2", "target_3",
    "t1_hit", "t2_hit", "t3_hit",
    "capital_allocated", "risk_amount", "risk_per_share",
    "entry_time", "exit_time", "exit_price", "exit_reason",
    "holding_minutes", "partial_exit_count", "pnl", "pnl_percent", "r_multiple",
    "order_state",
]


def holding_minutes(entry_time_str, exit_time_str):
    if not entry_time_str:
        return 0.0
    entry = datetime.fromisoformat(entry_time_str)
    end = datetime.fromisoformat(exit_time_str) if exit_time_str else datetime.now()
    return round((end - entry).total_seconds() / 60, 1)


def row_from_open_position(p: dict) -> dict:
    pnl = p.get("pnl", 0.0)
    capital = p.get("capital_allocated", 0.0)
    risk = p.get("risk_amount", 0.0)
    return {
        "order_id": p.get("order_id"), "signal_id": p.get("signal_id"), "status": p.get("status"),
        "underlying_symbol": p.get("underlying_symbol"), "sector": p.get("sector"),
        "stock_symbol": p.get("stock_symbol"), "option_type": p.get("option_type"),
        "trade_type": p.get("trade_type"), "strike_price": p.get("strike_price"),
        "expiry_date": p.get("expiry_date"), "lot_size": p.get("lot_size"), "score": p.get("score"),
        "underlying_price_at_entry": p.get("underlying_price_at_entry"),
        "orb_high": p.get("orb_high"), "orb_low": p.get("orb_low"),
        "orb_range_percent": round(p.get("orb_range_percent", 0.0), 3),
        "breakout_price": p.get("breakout_price"), "pullback_level": p.get("pullback_level"),
        "entry_price": p.get("entry_price"), "filled_price": p.get("filled_price"),
        "quantity": p.get("quantity"), "filled_quantity": p.get("filled_quantity"),
        "remaining_quantity": p.get("remaining_quantity"),
        "underlying_stop_loss": p.get("underlying_stop_loss"), "underlying_target": p.get("underlying_target"),
        "stop_loss": p.get("stop_loss"),
        "target_1": p.get("target_1"), "target_2": p.get("target_2"), "target_3": p.get("target_3"),
        "t1_hit": p.get("t1_hit"), "t2_hit": p.get("t2_hit"), "t3_hit": p.get("t3_hit"),
        "capital_allocated": round(capital, 2), "risk_amount": round(risk, 2),
        "risk_per_share": round(p.get("risk_per_share", 0.0), 2),
        "entry_time": p.get("entry_time"), "exit_time": p.get("exit_time"),
        "exit_price": p.get("exit_price"), "exit_reason": p.get("exit_reason"),
        "holding_minutes": holding_minutes(p.get("entry_time"), p.get("exit_time")),
        "partial_exit_count": len(p.get("partial_exits", [])),
        "pnl": round(pnl, 2),
        "pnl_percent": round((pnl / capital) * 100, 2) if capital else 0.0,
        "r_multiple": round(pnl / risk, 2) if risk else 0.0,
        "order_state": p.get("order_state"),
    }


rows = []

if os.path.exists(POSITIONS_FILE):
    with open(POSITIONS_FILE, "r") as f:
        positions = json.load(f)
    for p in positions:
        rows.append(row_from_open_position(p))

if os.path.exists(PAPER_TRADES_FILE):
    closed_df = pd.read_csv(PAPER_TRADES_FILE)
    closed_df = closed_df.rename(columns={
        "premium_entry_requested": "entry_price", "premium_entry_filled": "filled_price",
        "premium_emergency_stop": "stop_loss", "premium_target_1": "target_1",
        "premium_target_2": "target_2", "premium_target_3": "target_3",
        "final_exit_price": "exit_price", "net_pnl": "pnl",
    })
    for _, r in closed_df.iterrows():
        rows.append({col: r.get(col) for col in COLUMNS})

df = pd.DataFrame(rows, columns=COLUMNS)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Wrote {len(df)} paper trade(s) to {OUTPUT_FILE}")
