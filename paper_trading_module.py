import pandas as pd
import json
import os
from datetime import datetime

LOG_FILE = "paper_trades_log.csv"
STATE_FILE = "account_state.json"

def init_paper_trading():
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w") as f:
            json.dump({"balance": 100000.0, "total_realized_pnl": 0.0}, f)
            
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Trade_ID", "Timestamp", "Ticker_A", "Ticker_B", 
            "Entry_A", "Entry_B", "Gamma", "Direction", 
            "Size_USD", "Status", "Realized_PnL", 
            "Close_Timestamp", "Close_A", "Close_B"
        ])
        df.to_csv(LOG_FILE, index=False)

def get_account_state():
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def update_account_state(new_balance, r_pnl):
    state = get_account_state()
    state["balance"] = new_balance
    state["total_realized_pnl"] += r_pnl
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def get_trades():
    return pd.read_csv(LOG_FILE)

def execute_trade(ticker_a, ticker_b, entry_a, entry_b, gamma, direction, size_usd):
    df = get_trades()
    trade_id = f"TRD_{int(datetime.now().timestamp())}"
    new_trade = {
        "Trade_ID": trade_id,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker_A": ticker_a,
        "Ticker_B": ticker_b,
        "Entry_A": entry_a,
        "Entry_B": entry_b,
        "Gamma": gamma,
        "Direction": direction,
        "Size_USD": size_usd,
        "Status": "OPEN",
        "Realized_PnL": 0.0,
        "Close_Timestamp": "",
        "Close_A": 0.0,
        "Close_B": 0.0
    }
    df = pd.concat([df, pd.DataFrame([new_trade])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)
    
def close_trade(trade_id, close_a, close_b, r_pnl):
    df = get_trades()
    idx = df[df["Trade_ID"] == trade_id].index
    if not idx.empty:
        df.loc[idx, "Status"] = "CLOSED"
        df.loc[idx, "Realized_PnL"] = r_pnl
        df.loc[idx, "Close_Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.loc[idx, "Close_A"] = close_a
        df.loc[idx, "Close_B"] = close_b
        df.to_csv(LOG_FILE, index=False)
