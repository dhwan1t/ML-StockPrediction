import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_stock_data(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Downloads historical OHLCV data for a given stock ticker using yfinance.
    
    This function fetches historical daily stock data. If no dates are provided,
    it defaults to fetching approximately 4 years of data (around 1000 trading days)
    ending today. The data is saved as a CSV file in 'data/raw/{ticker}.csv'.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        start (str, optional): The start date in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): The end date in 'YYYY-MM-DD' format. Defaults to None.
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the downloaded OHLCV data.
        
    Raises:
        ValueError: If dates are provided incorrectly or if no data is found for the ticker.
        Exception: Upon any network or unknown error during the download.
    """
    
    # Set default dates for approx 4 years (~1000-1200 rows of daily data)
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    if start is None:
        # 4 years ago roughly
        start = (datetime.today() - timedelta(days=4 * 365)).strftime('%Y-%m-%d')
        
    try:
        print(f"Downloading data for {ticker} from {start} to {end}...")
        
        # Download data
        data = yf.download(ticker, start=start, end=end)
        
        # Check if the dataframe is empty
        if data.empty:
            raise ValueError(f"No data found for ticker '{ticker}' between {start} and {end}.")
            
        print(f"Successfully downloaded {len(data)} rows.")
        
        # Save to CSV
        # Make sure the raw data directory exists
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_data_dir = os.path.join(base_dir, 'data', 'raw')
        os.makedirs(raw_data_dir, exist_ok=True)
        
        file_path = os.path.join(raw_data_dir, f"{ticker}.csv")
        data.to_csv(file_path)
        print(f"Data saved to {file_path}")
        
        return data
        
    except ValueError as ve:
        print(f"ValueError checking data: {ve}")
        raise ve
    except Exception as e:
        print(f"An error occurred while downloading the data: {e}")
        raise e

def create_target_variable(df):
    df = df.copy()
    future_return = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['Target'] = (future_return > 0.01).astype(int)
    df = df.iloc[:-3].reset_index(drop=False)
    up = df['Target'].sum()
    total = len(df)
    print(f"Target — Strong Up(1): {up} ({up/total*100:.1f}%)  Other(0): {total-up} ({(total-up)/total*100:.1f}%)")
    return df


if __name__ == "__main__":
    # Example usage: Download 4 years of Apple data.
    try:
        df = download_stock_data('AAPL')
        print("Raw Data Head:")
        print(df.head())
        
        # Create target variable
        df = create_target_variable(df, horizon=1)
        print("\nData Head with Target:")
        print(df[['Close', 'Target']].head())
    except Exception as e:
        print(f"Failed to run example: {e}")
