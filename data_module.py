import yfinance as yf
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

import requests

def get_sp500_tickers():
    """Fetches the current S&P 500 ticker list from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        # Wikipedia actively blocks the default Python User-Agent with a 403 Forbidden error.
        # We spoof a standard browser User-Agent to bypass this block.
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        
        # pandas can read html tables directly from the source text
        tables = pd.read_html(response.text)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers] # yfinance format for class shares
        return tickers
    except Exception as e:
        print(f"CRITICAL: Failed to load S&P 500 from Wikipedia: {e}")
        # Fallback list if wiki fails to load
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG"]

def fetch_sp500_data(start_date, end_date):
    """
    Downloads historical prices for the full S&P 500 universe.
    """
    tickers = get_sp500_tickers()
    print(f"Downloading data for {len(tickers)} S&P 500 tickers...")
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    
    if 'Adj Close' in raw_data.columns.get_level_values(0):
        data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns.get_level_values(0):
        data = raw_data['Close']
    else:
        # Fallback if unindexed
        data = raw_data.iloc[:, list(raw_data.columns.get_level_values(0) == 'Close')]
        data.columns = data.columns.get_level_values(1)
        
    data.dropna(axis=1, inplace=True) # Drop missing chunks
    return data

def calculate_npd(data):
    """
    Calculates Normalized Price Distance (NPD) for all combinations efficiently.
    Uses scipy.spatial.distance.pdist for mathematically fast C-optimized vector loops.
    """
    normalized_data = data / data.iloc[0]
    tickers = normalized_data.columns.tolist()
    n = len(tickers)
    
    # Fast evaluation of squared euclidean distance ~ np.sum((x - y)**2) across all O(N^2) combos
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
    """
    Pre-screens the S&P 500 and filters out the top performing NPD pairs.
    """
    data = fetch_sp500_data(start_date, end_date)
    pairs_df = calculate_npd(data)
    
    top_pairs = pairs_df.head(top_n)
    return top_pairs, data
