import yfinance as yf
import pandas as pd
import numpy as np

def fetch_sp500_tech_data(start_date, end_date):
    """
    Downloads historical Adjusted Close prices for the S&P 500 Information Technology sector.
    """
    # Expanded list of prominent S&P 500 Info Tech companies as a proxy.
    # In a fully blown version we might scrape this, but a solid representative list works best for yfinance.
    tech_tickers = [
        "AAPL", "MSFT", "NVDA", "AVGO", "CSCO", "ACN", "ORCL", "CRM", "AMD",
        "TXN", "INTC", "QCOM", "IBM", "NOW", "INTU", "AMAT", "ADI", "MU",
        "LRCX", "PANW", "KLAC", "SNPS", "CDNS", "ROP", "HPQ", "TEL", "MSI",
        "GLW", "HPE", "FICO", "TYL", "NTAP", "WDC", "STX", "PTC", "TDY"
    ]
    
    print(f"Downloading data for {len(tech_tickers)} Tech tickers...")
    raw_data = yf.download(tech_tickers, start=start_date, end=end_date)
    
    # Newer versions of yfinance might omit 'Adj Close' and pre-adjust 'Close', or structure the index differently
    if 'Adj Close' in raw_data.columns.get_level_values(0):
        data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns.get_level_values(0):
        data = raw_data['Close']
    else:
        # Fallback if the top level doesn't match standard multi-index
        data = raw_data.iloc[:, list(raw_data.columns.get_level_values(0) == 'Close')]
        data.columns = data.columns.get_level_values(1)
        
    data.dropna(axis=1, inplace=True) # Drop tickers with incomplete data
    return data

def calculate_npd(data):
    """
    Calculates Normalized Price Distance (NPD) for all pairs.
    NPD = sum_t (p_1t / p_10 - p_2t / p_20)^2
    Returns the top pairs sorted by NPD.
    """
    normalized_data = data / data.iloc[0] # p_it / p_i0
    
    tickers = normalized_data.columns
    n = len(tickers)
    
    pairs = []
    
    # Calculate pairwise Euclidean distance on normalized prices
    for i in range(n):
        for j in range(i + 1, n):
            t1 = tickers[i]
            t2 = tickers[j]
            npd = np.sum((normalized_data[t1] - normalized_data[t2]) ** 2)
            pairs.append({
                "Ticker_A": t1,
                "Ticker_B": t2,
                "NPD": npd
            })
            
    pairs_df = pd.DataFrame(pairs)
    pairs_df.sort_values(by="NPD", inplace=True)
    return pairs_df.reset_index(drop=True)

def pre_screen_pairs(start_date, end_date, top_n=50):
    """
    Downloads data and returns the top_n pairs with the lowest NPD, 
    along with the raw pricing data for those tickers.
    """
    data = fetch_sp500_tech_data(start_date, end_date)
    pairs_df = calculate_npd(data)
    
    top_pairs = pairs_df.head(top_n)
    return top_pairs, data
