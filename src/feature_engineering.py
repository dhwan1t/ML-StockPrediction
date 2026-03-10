import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes exactly 8 technical features for stock prediction.
    
    Args:
        df: DataFrame with OHLCV data and Target column.
        
    Returns:
        DataFrame with only Target + 8 features, NaN rows removed.
    """
    df_feat = df.copy()
    
    # Ensure columns are 1D Series
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df_feat.columns and isinstance(df_feat[col], pd.DataFrame):
            df_feat[col] = df_feat[col].squeeze()
    
    close_series = df_feat['Close'].squeeze()
    high_series = df_feat['High'].squeeze()
    low_series = df_feat['Low'].squeeze()
    volume_series = df_feat['Volume'].squeeze()
    
    # 8 Core Features
    # 1. RSI_14
    df_feat['RSI_14'] = RSIIndicator(close=close_series, window=14).rsi()
    
    # 2. MACD_Hist
    macd_instance = MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)
    df_feat['MACD_Hist'] = macd_instance.macd_diff()
    
    # 3. BB_Width — (BB_Upper - BB_Lower) / SMA_20
    bb_instance = BollingerBands(close=close_series, window=20, window_dev=2)
    df_feat['BB_Width'] = bb_instance.bollinger_wband()
    
    # 4. Price_vs_SMA20
    sma_20 = SMAIndicator(close=close_series, window=20).sma_indicator()
    df_feat['Price_vs_SMA20'] = close_series / sma_20.squeeze()
    
    # 5. ROC_5
    df_feat['ROC_5'] = ROCIndicator(close=close_series, window=5).roc()
    
    # 6. ATR_14
    df_feat['ATR_14'] = AverageTrueRange(high=high_series, low=low_series, close=close_series, window=14).average_true_range()
    
    # 7. Volume_Ratio
    vol_sma_10 = volume_series.rolling(window=10).mean()
    df_feat['Volume_Ratio'] = volume_series / vol_sma_10.squeeze()
    
    # 8. Return_Lag_1
    daily_return = (close_series - close_series.shift(1)) / close_series.shift(1)
    df_feat['Return_Lag_1'] = daily_return.shift(1)
    
    # Keep only Target and 8 features
    features_to_keep = ['Target', 'RSI_14', 'MACD_Hist', 'BB_Width', 'Price_vs_SMA20', 'ROC_5', 'ATR_14', 'Volume_Ratio', 'Return_Lag_1']
    df_feat = df_feat[features_to_keep]
    
    initial_len = len(df_feat)
    df_feat.dropna(inplace=True)
    final_len = len(df_feat)
    
    print(f"Engineered Features built. Dropped {initial_len - final_len} rows due to NaN values.")
    print(f"Final dataset shape: {df_feat.shape}")
    
    return df_feat


if __name__ == "__main__":
    # Quick testing of the module over absolute filepath
    import os, sys
    
    # Attempt to use the existing data loader if run independently
    try:
        from data_loader import download_stock_data, create_target_variable
        
        print("Testing feature engineering...")
        df_raw = download_stock_data("MSFT", start="2020-01-01", end="2024-01-01")
        df_target = create_target_variable(df_raw, horizon=1)
        
        df_featured = build_features(df_target)
        
        # Test feature selection
        print("\nTesting feature selection...")
        X = df_featured.drop(columns=['Target'])
        y = df_featured['Target']
        
        # Only select numeric columns for scikit-learn
        X_numeric = X.select_dtypes(include=[np.number])
        
        X_selected = select_features(X_numeric, y, k=15)
        
        print("\nFeature Selection Test Successful. Selected Data:")
        print(X_selected.head())
        
    except ImportError:
        print("Could not import data_loader, ensure you are running from the project root.")
