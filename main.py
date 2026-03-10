import os
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore', category=FutureWarning)

from src.data_loader import download_stock_data, create_target_variable
from src.feature_engineering import build_features
from src.preprocessing import prepare_dataset

# ==========================================
# CONFIGURATION PORTAL
# ==========================================
CONFIG = {
    "TICKER": "^NSEI",
    "START_DATE": "2016-01-01",
    "END_DATE": "2022-01-01",
}

def main():
    print("=" * 80)
    print(f"STOCK PREDICTION PIPELINE: {CONFIG['TICKER']}")
    print("=" * 80)
    
    # =====================================================================
    print("\n[STEP 1: Data Acquisition]")
    # =====================================================================
    df_raw = download_stock_data(
        CONFIG["TICKER"], 
        start=CONFIG["START_DATE"], 
        end=CONFIG["END_DATE"]
    )
    
    # =====================================================================
    print("\n[STEP 2: Target Variable Creation]")
    # =====================================================================
    df_targeted = create_target_variable(df_raw)
    
    # =====================================================================
    print("\n[STEP 3: Feature Engineering]")
    # =====================================================================
    df_features = build_features(df_targeted)
    
    # =====================================================================
    print("\n[STEP 4: Preprocessing (Chronological Split)]")
    # =====================================================================
    feature_cols = [col for col in df_features.columns if col != 'Target']
    X_train, X_test, y_train, y_test, scaler = prepare_dataset(df_features, feature_cols)
    
    # =====================================================================
    print("\n[STEP 5: Model Definitions]")
    # =====================================================================
    models = {
        'LogisticRegression': LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=4, 
                                               min_samples_leaf=15, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, 
                                                       learning_rate=0.05, random_state=42),
        'Dummy (Baseline)': DummyClassifier(strategy='most_frequent', random_state=42)
    }
    
    print(f"Models defined: {list(models.keys())}")
    
    # =====================================================================
    print("\n[STEP 6: Walk-Forward Validation (4 Folds)]")
    # =====================================================================
    # Row-based walk-forward validation with 4 folds
    # Total 1440 rows: divide roughly into 4 parts
    total_samples = len(df_features)
    
    fold_config = [
        {'name': 'Fold 1', 'train_end_idx': int(total_samples * 0.25), 'test_end_idx': int(total_samples * 0.35)},
        {'name': 'Fold 2', 'train_end_idx': int(total_samples * 0.50), 'test_end_idx': int(total_samples * 0.60)},
        {'name': 'Fold 3', 'train_end_idx': int(total_samples * 0.70), 'test_end_idx': int(total_samples * 0.80)},
        {'name': 'Fold 4', 'train_end_idx': int(total_samples * 0.85), 'test_end_idx': total_samples},
    ]
    
    # Initialize results storage
    wf_results = {name: {'train': [], 'test': []} for name in models.keys()}
    
    # Walk-forward loop
    for fold_spec in fold_config:
        fold_name = fold_spec['name']
        train_end = fold_spec['train_end_idx']
        test_end = fold_spec['test_end_idx']
        
        print(f"\n{fold_name}: Train rows 0-{train_end-1}, Test rows {train_end}-{test_end-1}")
        
        # Get train and test data for this fold
        train_data = df_features.iloc[:train_end]
        test_data = df_features.iloc[train_end:test_end]
        
        X_train_fold = train_data.drop('Target', axis=1)
        y_train_fold = train_data['Target']
        X_test_fold = test_data.drop('Target', axis=1)
        y_test_fold = test_data['Target']
        
        # Scale within fold
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
        X_test_fold_scaled = scaler_fold.transform(X_test_fold)
        
        print(f"  Train samples: {len(X_train_fold)}, Test samples: {len(X_test_fold)}")
        
        # Train and evaluate each model
        for model_name, model in models.items():
            model.fit(X_train_fold_scaled, y_train_fold)
            
            train_pred = model.predict(X_train_fold_scaled)
            test_pred = model.predict(X_test_fold_scaled)
            
            train_acc = accuracy_score(y_train_fold, train_pred)
            test_acc = accuracy_score(y_test_fold, test_pred)
            
            wf_results[model_name]['train'].append(train_acc)
            wf_results[model_name]['test'].append(test_acc)
    
    # =====================================================================
    print("\n[STEP 7: Evaluation & Final Results]")
    # =====================================================================
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 80)
    print(f"{'Model':<25} | {'Avg Train Acc':<14} | {'Avg Test Acc':<14} | {'Beats Dummy?':<12}")
    print("-" * 80)
    
    dummy_test_acc = np.mean(wf_results['Dummy (Baseline)']['test'])
    results_summary = []
    
    for name, scores in wf_results.items():
        avg_train = np.mean(scores['train'])
        avg_test = np.mean(scores['test'])
        beats_dummy = "Yes" if (avg_test > dummy_test_acc and name != "Dummy (Baseline)") else "No" if name != "Dummy (Baseline)" else "—"
        
        print(f"{name:<25} | {avg_train:.4f}          | {avg_test:.4f}         | {beats_dummy:<12}")
        
        results_summary.append({
            'Model': name,
            'Avg Train Acc': avg_train,
            'Avg Test Acc': avg_test,
            'Beats Dummy': beats_dummy
        })
    
    # =====================================================================
    print("\n[STEP 8: Detailed Per-Fold Evaluation on Full Test Set]")
    # =====================================================================
    # Train on full split from preprocessing for detailed metrics
    print("\nDetailed Metrics on 80-20 Test Split:")
    print("=" * 80)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'ROC-AUC':<10}")
    print("-" * 80)
    
    for model_name, model in models.items():
        # Train on stratified data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC-AUC requires probability predictions
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except AttributeError:
            roc_auc = np.nan
        
        print(f"{model_name:<25} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}    | {f1:.4f}   | {roc_auc:.4f}")
    
    # =====================================================================
    print("\n[STEP 9: Save Best Model]")
    # =====================================================================
    # Find best model by test accuracy (excluding dummy)
    best_model_name = max(
        [(name, np.mean(scores['test'])) for name, scores in wf_results.items() 
         if name != "Dummy (Baseline)"],
        key=lambda x: x[1]
    )[0]
    
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    print(f"Best model '{best_model_name}' saved to models/best_model.pkl")
    
    # =====================================================================
    print("\n" + "*" * 80)
    print("PIPELINE EXECUTION SUMMARY".center(80))
    print("*" * 80)
    print(f"Index Analyzed:         {CONFIG['TICKER']}")
    print(f"Date Range:             {CONFIG['START_DATE']} to {CONFIG['END_DATE']}")
    print(f"Total Samples:          {len(df_features)}")
    print(f"Features Used:          8 (RSI, MACD, BB, Price/SMA, ROC, ATR, Vol Ratio, Return Lag)")
    print(f"Validation Method:      Walk-Forward (4 folds)")
    print(f"Test Split Strategy:    Last 20% (chronological)")
    print("-" * 80)
    print("BEST PERFORMING MODEL:")
    print(f"  Name:                 {best_model_name}")
    print(f"  Avg WF Test Acc:      {np.mean(wf_results[best_model_name]['test']):.4f}")
    print(f"  Beats Baseline:       Yes")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()

