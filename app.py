import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

from data_module import pre_screen_pairs, get_sp500_tickers, extract_price_data
from cointegration_module import run_kalman_filter, check_cointegration
from optimization_module import optimize_threshold
from backtest_module import simulate_backtest
from paper_trading_module import init_paper_trading, get_account_state, update_account_state, get_trades, execute_trade, close_trade

# Initialize State
init_paper_trading()

st.set_page_config(layout="wide", page_title="Institutional Pairs Trading System v1.1")
st.title("Dynamic Kalman Filter Pairs Trading (Version 1.1)")

tab1, tab2 = st.tabs(["Research & Backtest", "Paper Trading Simulator"])

with tab1:
    st.header("Research & Analysis")
    col1, col2, col3 = st.columns(3)
    start_date = col1.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = col2.date_input("End Date", pd.to_datetime("2024-01-01"))
    
    if "price_data" not in st.session_state:
        st.session_state.price_data = None
        st.session_state.top_pairs = None

    if col3.button("Fetch & Pre-screen Data"):
        with st.spinner("Downloading full S&P 500 Data & Filtering ≈125,000 combinations via NPD..."):
            top_pairs, price_data = pre_screen_pairs(start_date, end_date, top_n=50)
            st.session_state.top_pairs = top_pairs
            st.session_state.price_data = price_data
            st.success("Data fetched!")

    if st.session_state.top_pairs is not None:
        pair_opts = [(row['Ticker_A'], row['Ticker_B']) for idx, row in st.session_state.top_pairs.head(20).iterrows()]
        pair_strs = [f"{a} vs {b}" for a, b in pair_opts]
        
        selected_pair_str = st.selectbox("Select a Pair to Analyze", pair_strs)
        idx = pair_strs.index(selected_pair_str)
        t1, t2 = pair_opts[idx]
        
        st.subheader(f"Analyzing {t1} and {t2}")
        
        y1 = st.session_state.price_data[t1]
        y2 = st.session_state.price_data[t2]
        
        # 1. Kalman Filter & Cointegration
        with st.spinner("Running Kalman Filter..."):
            burn_in_days = 60
            kf_res = run_kalman_filter(y1, y2, burn_in=burn_in_days)
            is_coint, p_val = check_cointegration(kf_res['z_score'], burn_in=burn_in_days)
        
        st.markdown(f"**ADF Test p-value (post burn-in):** `{p_val:.4f}` {'(Cointegrated ✅)' if is_coint else '(Not Stationary ❌)'}")
        
        # 2. Optimization
        calc_s0_opt = optimize_threshold(kf_res['z_score'], burn_in=burn_in_days)
        cA, cB = st.columns(2)
        cA.markdown(f"**Calculated Optimal Threshold:** `±{calc_s0_opt:.2f}` Z-Score")
        s0_opt = cB.number_input("Tuning Threshold / Override (Z-Score)", min_value=0.1, max_value=4.0, value=round(calc_s0_opt, 2), step=0.1)
        
        # 3. Backtest
        bt_res = simulate_backtest(y1, y2, kf_res, s0_opt=s0_opt, hard_stop=4.0, fee_bps=5.0)
        
        # Plot 1: Normalized Log Prices
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=y1.index, y=np.log(y1/y1.iloc[0]), mode='lines', name=t1))
        fig1.add_trace(go.Scatter(x=y2.index, y=np.log(y2/y2.iloc[0]), mode='lines', name=t2))
        fig1.update_layout(title="Fig 1: Normalized Log-Prices", hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Plot 2: Dynamic KF Parameters
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=kf_res.index, y=kf_res['gamma'], mode='lines', name='Hedge Ratio (γ)'))
        fig2.add_trace(go.Scatter(x=kf_res.index, y=kf_res['mu'], mode='lines', name='Mean (μ)', yaxis='y2'))
        fig2.update_layout(
            title="Fig 2: Kalman Filter Dynamic Parameters", 
            hovermode="x unified",
            yaxis2=dict(title="Mean (μ)", overlaying='y', side='right')
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Plot 3: Dynamic Z-Score Spread & Thresholds
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=kf_res.index, y=kf_res['z_score'], mode='lines', name='Spread (Z-Score)'))
        fig3.add_trace(go.Scatter(x=kf_res.index, y=[0]*len(kf_res), mode='lines', line=dict(dash='dash', color='gray'), name='Zero Axis'))
        fig3.add_trace(go.Scatter(x=kf_res.index, y=[s0_opt]*len(kf_res), mode='lines', line=dict(dash='dash', color='red'), name='+s0* Threshold'))
        fig3.add_trace(go.Scatter(x=kf_res.index, y=[-s0_opt]*len(kf_res), mode='lines', line=dict(dash='dash', color='green'), name='-s0* Threshold'))
        fig3.add_trace(go.Scatter(x=kf_res.index, y=[4.0]*len(kf_res), mode='lines', line=dict(dash='dot', color='orange'), name='Hard Stop-Loss'))
        fig3.add_trace(go.Scatter(x=kf_res.index, y=[-4.0]*len(kf_res), mode='lines', line=dict(dash='dot', color='orange'), name='Hard Stop-Loss'))
        fig3.update_layout(title="Fig 3: Standardized Spread & Optimized Thresholds", hovermode="x unified")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Plot 4: Cumulative Return
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=bt_res.index, y=bt_res['Cum_PnL'], mode='lines', name='Cum PnL', line=dict(color='purple')))
        fig4.add_trace(go.Scatter(x=bt_res.index, y=bt_res['Position'], mode='lines', name='Position', yaxis='y2', opacity=0.3))
        fig4.update_layout(
            title="Fig 4: Cumulative Backtest PnL (Net of Fees)", 
            hovermode="x unified",
            yaxis2=dict(title="Position", overlaying='y', side='right', range=[-2, 2])
        )
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.header("Paper Trading Simulator")
    
    st.subheader("📡 Market Radar & Alerts")
    c_auto, c_ref = st.columns([1, 4])
    auto_refresh = c_auto.checkbox("Auto-Refresh (1m)")
    if auto_refresh:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60000, key="radar")
        
    if c_ref.button("Refresh Live Market Data"):
        st.rerun()
        
    alerts = []
    trades_df = get_trades()
    open_trades = trades_df[trades_df["Status"] == "OPEN"]
    closed_trades = trades_df[trades_df["Status"] == "CLOSED"]
    
    if not open_trades.empty or "exec_data" in st.session_state:
        # Collect pairs to track
        track_pairs = []
        for _, row in open_trades.iterrows():
            track_pairs.append({'Type': 'OPEN', 'A': row['Ticker_A'], 'B': row['Ticker_B'], 'Dir': row['Direction']})
            
        if "exec_data" in st.session_state:
            exec_A = st.session_state['exec_data']['Ticker_A']
            exec_B = st.session_state['exec_data']['Ticker_B']
            if not ((open_trades["Ticker_A"] == exec_A) & (open_trades["Ticker_B"] == exec_B)).any():
                track_pairs.append({'Type': 'ENTRY', 'A': exec_A, 'B': exec_B, 'Dir': None})
                
        # Evaluate dynamically using silent Kalman filter instance
        if track_pairs:
            with st.spinner("Market Radar scanning live metrics..."):
                all_radar_tickers = list(set([p['A'] for p in track_pairs] + [p['B'] for p in track_pairs]))
                end_d = pd.to_datetime('today')
                start_d = end_d - pd.DateOffset(years=2)
                
                try:
                    raw_all = yf.download(all_radar_tickers, start=start_d, end=end_d, progress=False)
                    d_all = extract_price_data(raw_all)
                    d_all.ffill(inplace=True)
                    
                    if not d_all.empty:
                        for item in track_pairs:
                            ta, tb = item['A'], item['B']
                            if ta in d_all.columns and tb in d_all.columns:
                                kf_res = run_kalman_filter(d_all[ta], d_all[tb], burn_in=60)
                                z_latest = kf_res['z_score'].iloc[-1]
                                
                                if item['Type'] == 'OPEN':
                                    if abs(z_latest) > 4.0:
                                        alerts.append({"st": st.error, "msg": f"🚨 [STOP LOSS] {ta} vs {tb} Z-Score diverged to {z_latest:.2f}! Hard Stop-Loss required."})
                                    elif abs(z_latest) <= 0.1:
                                        alerts.append({"st": st.success, "msg": f"✅ [TAKE PROFIT] {ta} vs {tb} Spread has reverted (Z-Score: {z_latest:.2f}). Ready to Close!"})
                                else:
                                    s0 = optimize_threshold(kf_res['z_score'], burn_in=60)
                                    if abs(z_latest) > s0:
                                        side = "Long" if z_latest < -s0 else "Short"
                                        alerts.append({"st": st.warning, "msg": f"🎯 [ENTRY SIGNAL] {ta} vs {tb} Z-Score ({z_latest:.2f}) exceeds threshold ({s0:.2f}). {side} highly recommended!"})
                except Exception as e:
                    alerts.append({"st": st.warning, "msg": f"📡 API Rate Limit Hit: Temporary network error preventing scan."})

    if alerts:
        for a in alerts:
            a["st"](a["msg"])
    else:
        st.info("No action required. Monitoring...")
        
    st.divider()
    
    # 1. State Management & Account Metrics
    account_state = get_account_state()
    
    # Calculate Live Unrealized PnL
    unrealized_pnl = 0.0
    open_positions_display = []
    
    if not open_trades.empty:
        # Fetch live prices for open components
        live_tickers = list(set(open_trades["Ticker_A"]).union(set(open_trades["Ticker_B"])))
        try:
            raw_live = yf.download(live_tickers, period="1d", progress=False)
            live_data = extract_price_data(raw_live)
            
            if not live_data.empty:
                live_data = live_data.iloc[-1]
            else:
                live_data = None
        except:
            live_data = None
            
        if live_data is not None:
            for _, row in open_trades.iterrows():
            ta = row["Ticker_A"]
            tb = row["Ticker_B"]
            gamma = row["Gamma"]
            ea = row["Entry_A"]
            eb = row["Entry_B"]
            live_a = live_data[ta]
            live_b = live_data[tb]
            
            size_usd = row["Size_USD"]
            dir = row["Direction"]
            
            w_a = 1.0 / (1.0 + abs(gamma))
            w_b = abs(gamma) / (1.0 + abs(gamma))
            
            ret_a = (live_a - ea) / ea
            ret_b = (live_b - eb) / eb
            
            if dir == "Long Spread":
                trade_pnl = size_usd * (w_a * ret_a - w_b * ret_b)
            else:
                trade_pnl = size_usd * (-w_a * ret_a + w_b * ret_b)
                
                open_positions_display.append({
                    "Trade_ID": row["Trade_ID"],
                    "Pair": f"{ta} vs {tb}",
                    "Direction": dir,
                    "Size": f"${size_usd:,.2f}",
                    "Live A": f"${live_a:.2f}",
                    "Live B": f"${live_b:.2f}",
                    "Unrealized PnL": trade_pnl
                })
                unrealized_pnl += trade_pnl
        else:
            st.error("Yahoo Finance API Rate Limit Hit. Prices unavailable.")
            
    # Metric Cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Balance", f"${account_state['balance'] + unrealized_pnl:,.2f}", f"{unrealized_pnl:,.2f} Unrealized")
    m2.metric("Total Unrealized PnL", f"${unrealized_pnl:,.2f}")
    m3.metric("Total Realized PnL", f"${account_state['total_realized_pnl']:,.2f}")
    
    st.divider()
    
    # 2. Execution Interface
    st.subheader("Execution Panel")
    exec1, exec2, exec3 = st.columns(3)
    
    # Read the live list of tickers instead of hardcoded
    all_tickers = get_sp500_tickers()
    ticker_a = exec1.selectbox("Stock A", all_tickers, index=all_tickers.index("AAPL") if "AAPL" in all_tickers else 0)
    ticker_b = exec2.selectbox("Stock B", all_tickers, index=all_tickers.index("MSFT") if "MSFT" in all_tickers else 1)
    
    if exec3.button("Fetch Live Pair Data"):
        with st.spinner("Fetching data and evaluating Kalman Filter..."):
            end_d = pd.to_datetime('today')
            start_d = end_d - pd.DateOffset(years=2)
            raw = yf.download([ticker_a, ticker_b], start=start_d, end=end_d)
            d = extract_price_data(raw)
            d.ffill(inplace=True)

            kf_res = run_kalman_filter(d[ticker_a], d[ticker_b], burn_in=60)
            
            # Save into session state
            st.session_state["exec_data"] = {
                "Z_Score": kf_res['z_score'].iloc[-1],
                "Gamma": kf_res['gamma'].iloc[-1],
                "Price_A": d[ticker_a].iloc[-1],
                "Price_B": d[ticker_b].iloc[-1],
                "Ticker_A": ticker_a,
                "Ticker_B": ticker_b
            }
            
    if "exec_data" in st.session_state and st.session_state["exec_data"]["Ticker_A"] == ticker_a and st.session_state["exec_data"]["Ticker_B"] == ticker_b:
        d = st.session_state["exec_data"]
        st.info(f"**Live Z-Score:** `{d['Z_Score']:.2f}` | **Hedge Ratio (\\gamma):** `{d['Gamma']:.4f}` | **{ticker_a}:** `${d['Price_A']:.2f}` | **{ticker_b}:** `${d['Price_B']:.2f}`")
        
        ts_size = st.number_input("Trade Size (USD)", min_value=100.0, value=10000.0, step=1000.0)
        
        b1, b2 = st.columns(2)
        if b1.button("Long Spread (Buy A, Short B)"):
            execute_trade(ticker_a, ticker_b, d['Price_A'], d['Price_B'], d['Gamma'], "Long Spread", ts_size)
            st.success("Trade Successfully Executed!")
            st.rerun()
            
        if b2.button("Short Spread (Short A, Buy B)"):
            execute_trade(ticker_a, ticker_b, d['Price_A'], d['Price_B'], d['Gamma'], "Short Spread", ts_size)
            st.success("Trade Successfully Executed!")
            st.rerun()

    st.divider()
    
    # 3. Portfolio Dashboard
    st.subheader("Open Positions")
    if open_positions_display:
        for pos in open_positions_display:
            col_a, col_b, col_c, col_d, col_e = st.columns([2,1,1,2,1])
            col_a.write(f"📈 **{pos['Pair']}** ({pos['Direction']})")
            col_b.write(f"Size: {pos['Size']}")
            col_c.write(f"PnL: **${pos['Unrealized PnL']:.2f}**")
            col_d.write(f"Live A: {pos['Live A']} | Live B: {pos['Live B']}")
            
            if col_e.button("Close Position", key=pos["Trade_ID"]):
                size_usd_val = open_trades[open_trades["Trade_ID"] == pos["Trade_ID"]].iloc[0]["Size_USD"]
                txn_fee = size_usd_val * 0.001
                r_pnl = pos["Unrealized PnL"] - txn_fee
                
                la = float(pos['Live A'].replace('$',''))
                lb = float(pos['Live B'].replace('$',''))
                
                close_trade(pos["Trade_ID"], la, lb, r_pnl)
                
                state = get_account_state()
                update_account_state(state["balance"] + r_pnl, r_pnl)
                st.rerun()
    else:
        st.write("No open positions.")
        
    st.subheader("Closed Trades History")
    if not closed_trades.empty:
        st.dataframe(closed_trades[["Timestamp", "Ticker_A", "Ticker_B", "Direction", "Size_USD", "Realized_PnL", "Close_Timestamp"]])
    else:
        st.write("No trades closed yet.")
