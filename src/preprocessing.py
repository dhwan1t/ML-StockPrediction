import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_dataset(df: pd.DataFrame, feature_cols: list = None, target_col: str = 'Target', test_size: float = 0.2):
    """
    Chronological split: last 20% as test, no shuffling.
    StandardScaler fit on train only.
    
    Args:
        df (pd.DataFrame): Full dataset with features and target.
        feature_cols (list): Feature column names. If None, all except target.
        target_col (str): Target column name.
        test_size (float): Proportion for test set.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("\n--- Preparing Dataset (Chronological Split) ---")
    
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Chronological split: last 20% as test
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train Set Size: {len(X_train)} rows")
    print(f"Test Set Size:  {len(X_test)} rows")
    
    # StandardScaler fit on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    
    # Class balance
    train_balance = y_train.value_counts(normalize=True) * 100
    test_balance = y_test.value_counts(normalize=True) * 100
    
    print(f"\nTrain Class Balance:\n{train_balance.to_string()}")
    print(f"\nTest Class Balance:\n{test_balance.to_string()}")
    
    print("\nProcessing Complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_preprocessed(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, scaler: StandardScaler):
    """
    Saves the processed training and testing splits, along with the fitted scaler,
    to the 'data/processed' directory using joblib for downstream reuse.
    """
    # Ensure processed directory exists
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"\nSaving preprocessed artifacts to {processed_dir}...")
    
    joblib.dump(X_train, os.path.join(processed_dir, 'X_train.joblib'))
    joblib.dump(X_test, os.path.join(processed_dir, 'X_test.joblib'))
    joblib.dump(y_train, os.path.join(processed_dir, 'y_train.joblib'))
    joblib.dump(y_test, os.path.join(processed_dir, 'y_test.joblib'))
    joblib.dump(scaler, os.path.join(processed_dir, 'scaler.joblib'))
    
    print("Preprocessed artifacts saved successfully.")

if __name__ == "__main__":
    # Test script execution
    try:
        from data_loader import download_stock_data, create_target_variable
        from feature_engineering import build_features, select_features
        
        print("Testing preprocessing pipeline...")
        df_raw = download_stock_data("MSFT", start="2020-01-01", end="2024-01-01")
        df_tgt = create_target_variable(df_raw)
        df_feat = build_features(df_tgt)
        
        # Feature selection
        X = df_feat.drop(columns=['Target'])
        y = df_feat['Target']
        X_numeric = X.select_dtypes(include=[np.number])
        X_sel = select_features(X_numeric, y, k=15)
        
        # Add target back into the dataframe for the prepare_dataset function
        df_final = X_sel.copy()
        df_final['Target'] = y
        
        feature_cols = X_sel.columns.tolist()
        
        # Preprocessing
        X_train, X_test, y_train, y_test, scaler = prepare_dataset(df_final, feature_cols)
        
        # Save artifacts
        save_preprocessed(X_train, X_test, y_train, y_test, scaler)
        
    except ImportError as e:
        print(f"Testing failed due to import error: {e}")
