import yfinance as yf
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
import requests
import io

def extract_price_data(raw_data):
    """Safely extracts Close/Adj Close from yf.download regardless of yfinance version MultiIndex structure."""
    if isinstance(raw_data.columns, pd.MultiIndex):
        l0 = raw_data.columns.get_level_values(0)
        l1 = raw_data.columns.get_level_values(1)
        
        if 'Adj Close' in l0:
            return raw_data['Adj Close']
        if 'Close' in l0:
            return raw_data['Close']
        if 'Adj Close' in l1:
            return raw_data.xs('Adj Close', level=1, axis=1)
        if 'Close' in l1:
            return raw_data.xs('Close', level=1, axis=1)
    else:
        if 'Adj Close' in raw_data.columns:
            return pd.DataFrame(raw_data['Adj Close'])
        if 'Close' in raw_data.columns:
            return pd.DataFrame(raw_data['Close'])
            
    return raw_data

def get_sp500_tickers():
    """Fetches the current S&P 500 ticker list using multiple fallbacks for extreme reliability."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers] # yfinance format for class shares
        return tickers
    except Exception as e1:
        try:
            csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
            df = pd.read_csv(csv_url)
            tickers = df['Symbol'].tolist()
            tickers = [t.replace('.', '-') for t in tickers]
            return tickers
        except Exception as e2:
            import streamlit as st
            st.error(f"CRITICAL: Failed to load S&P 500. HTML Error: {str(e1)} | CSV Error: {str(e2)}")
            return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG"]

def fetch_sp500_data(start_date, end_date):
    """Downloads historical prices for the full S&P 500 universe."""
    tickers = get_sp500_tickers()
    print(f"Downloading data for {len(tickers)} S&P 500 tickers...")
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    
    data = extract_price_data(raw_data)
    data.dropna(axis=1, inplace=True) # Drop missing chunks
    return data

def calculate_npd(data):
    """Calculates Normalized Price Distance (NPD) for all combinations efficiently."""
    normalized_data = data / data.iloc[0]
    tickers = normalized_data.columns.tolist()
    n = len(tickers)
    
    # Fast evaluation of squared euclidean distance across O(N^2) combos
    dist_matrix = pdist(normalized_data.T.values, metric='sqeuclidean')
    
    pairs = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "Ticker_A": tickers[i],
                "Ticker_B": tickers[j],
                "NPD": dist_matrix[idx]
            })
            idx += 1
            
    pairs_df = pd.DataFrame(pairs)
    pairs_df.sort_values(by="NPD", inplace=True)
    return pairs_df.reset_index(drop=True)

def pre_screen_pairs(start_date, end_date, top_n=50):
    """Pre-screens the S&P 500 and filters out the top performing NPD pairs."""
    data = fetch_sp500_data(start_date, end_date)
    pairs_df = calculate_npd(data)
    
    top_pairs = pairs_df.head(top_n)
    return top_pairs, data
