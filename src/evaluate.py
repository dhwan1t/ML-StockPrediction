import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# Ensure report output directory exists
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures'))
os.makedirs(REPORTS_DIR, exist_ok=True)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """
    Evaluates a trained classification model on the out-of-sample test set.
    
    This function computes core metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
    generates a confusion matrix heatmap, plots the ROC curve, plots the actual 
    vs predicted timeline for the last 100 days, and plots feature importances if available.
    """
    print(f"\n{'='*50}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'='*50}")
    
    # Generate Predictions
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # For things like SVC without probability=True
        y_proba = model.decision_function(X_test)
        # Normalize to 0-1 range roughly if we really needed it, but roc_curve can handle raw decisions
    else:
        y_proba = None
        
    # -------------------------------------------------------------------------
    # 1. Core Metrics
    # -------------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = np.nan
        
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }
    
    print("\nCore Metrics Table:")
    print(f"Accuracy:  {acc:.4f} (Target: >= 0.70)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
    # -------------------------------------------------------------------------
    # 2. Confusion Matrix
    # -------------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0 (Down)', 'Pred 1 (Up)'],
                yticklabels=['True 0 (Down)', 'True 1 (Up)'])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f'{model_name}_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # 3. ROC Curve
    # -------------------------------------------------------------------------
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(REPORTS_DIR, f'{model_name}_roc_curve.png'), dpi=300)
        plt.close()
        
    # -------------------------------------------------------------------------
    # 4. Prediction vs Actual Timeline (Last 100 days)
    # -------------------------------------------------------------------------
    plot_len = min(100, len(y_test))
    
    # Since scaling stripped dates by returning numpy arrays (or range index dataframes) in preprocessing, 
    # we'll plot by sequential step index to simulate a timeline.
    actuals = y_test.iloc[-plot_len:].values
    preds = y_pred[-plot_len:]
    x_steps = np.arange(plot_len)
    
    plt.figure(figsize=(14, 5))
    plt.scatter(x_steps[actuals == 1], actuals[actuals == 1], color='green', label='Actual Up', marker='o', alpha=0.6, s=50)
    plt.scatter(x_steps[actuals == 0], actuals[actuals == 0], color='red', label='Actual Down', marker='o', alpha=0.6, s=50)
    
    # Use plt.step for timeline steps instead of linestyle='steps-mid'
    plt.step(x_steps, preds, color='black', label='Predicted Direction', where='mid', alpha=0.5, linewidth=2)
    plt.yticks([0, 1], ['Down (0)', 'Up (1)'])
    plt.title(f"{model_name} - Actual vs Predicted (Last {plot_len} Test Days)")
    plt.xlabel("Days (Sequential Test Order)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f'{model_name}_predictions.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 5. Feature Coefficients / Importances
    # -------------------------------------------------------------------------
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models like Logistic Regression, plot magnitude of coefficients
        importances = np.abs(model.coef_[0])
        
    if importances is not None:
        # Sort and take top 10
        feat_series = pd.Series(importances, index=X_test.columns).sort_values(ascending=False).head(10)
        
        # Flatten tuple multi-index if it exists
        features = [str(col) if isinstance(col, tuple) else col for col in feat_series.index]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feat_series.values, y=features, palette='viridis')
        plt.title(f"{model_name} - Top 10 Feature Importances")
        plt.xlabel("Importance / Absolute Coefficient Magnitude")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, f'{model_name}_feature_importance.png'), dpi=300)
        plt.close()

    return metrics


def compare_all_models(models_dict: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Runs evaluate_model on a dictionary of trained models and outputs a summarized 
    ranking DataFrame sorted descending by F1-Score.
    """
    all_metrics = []
    
    print("\nEvaluating individual models...")
    for model_name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=model_name)
        metrics['Model'] = model_name
        all_metrics.append(metrics)
        
    # Compile comparison
    df_compare = pd.DataFrame(all_metrics)
    # Rearrange columns so Model is first
    cols = ['Model', 'ROC-AUC', 'Accuracy', 'F1-Score', 'Precision', 'Recall']
    df_compare = df_compare[cols]
    
    # Sort by ROC-AUC descending as primary metric, then Accuracy
    df_compare.sort_values(by=['ROC-AUC', 'Accuracy'], ascending=[False, False], inplace=True)
    df_compare.reset_index(drop=True, inplace=True)
    
    print(f"\n{'='*80}")
    print("FINAL MODEL RANKING COMPARISON (Sorted by ROC-AUC)")
    print(f"{'='*80}")
    print(df_compare.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*80}\n")
    
    return df_compare


if __name__ == "__main__":
    # Test execution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    try:
        # We need to run model.py if train_all_models outputs weren't saved
        # So instead we will simply load the best_model and pretend it's in a dict for the test
        print("Loading evaluation artifacts...")
        X_test = joblib.load(os.path.join(processed_dir, 'X_test.joblib'))
        y_test = joblib.load(os.path.join(processed_dir, 'y_test.joblib'))
        best_rf = joblib.load(os.path.join(processed_dir, 'best_model.joblib'))
        
        # We can also quickly train a dummy Logistic Regression just to test the compare function
        # since the dictionary of all models from model.py wasn't saved to disk.
        from sklearn.linear_model import LogisticRegression
        print("Training dummy LR for comparison test...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        X_train = joblib.load(os.path.join(processed_dir, 'X_train.joblib'))
        y_train = joblib.load(os.path.join(processed_dir, 'y_train.joblib'))
        lr.fit(X_train, y_train)
        
        # Test full comparison
        models_dict = {
            "Tuned Random Forest": best_rf,
            "Logistic Regression (Baseline)": lr
        }
        
        compare_all_models(models_dict, X_test, y_test)
            
    except FileNotFoundError:
        print("Could not find test data or models. Please run preprocessing.py and model.py first.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
