import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV

def get_calibrated_models():
    """
    Returns a dictionary of calibrated shallow models to prevent overfitting 
    on the small financial dataset, plus a baseline model.
    """
    print("\n--- Initializing Calibrated Models ---")
    
    models = {
        "Dummy (Baseline)": DummyClassifier(strategy='most_frequent'),
        
        # Heavily regularized (C=0.01) to rely only on the strongest signals
        "Logistic Regression": LogisticRegression(C=0.01, max_iter=1000, random_state=42),
        
        # Shallow trees (max_depth=3) and large leaves (min_samples_leaf=20) 
        # prevent fitting to noise
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=20, random_state=42, n_jobs=-1),
        
        # Slow learning (learning_rate=0.05), shallow trees (max_depth=2), 
        # and few estimators (n_estimators=50) to build a robust ensemble
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
    }
    
    return models

def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Performs grid search CV with TimeSeriesSplit to find the best hyperparameters 
    for the Random Forest classifier.
    
    Returns:
        The best fitted estimator model.
    """
    print("\n--- Hyperparameter Tuning ---")
    print("Tuning Random Forest Classifier...")
    
    # We tune Random Forest (or Gradient Boosting) as it typically performs best on this type of data.
    # Parameter Grid reasoning:
    # n_estimators: Number of trees. 100 is default, 200/300 might capture more complex signals.
    # max_depth: Limits tree depth. None can overfit small datasets. 5 or 10 ensures generalization.
    # min_samples_split: Higher values prevent learning noise at the leaves.
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
    
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Grid search across the parameters
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

if __name__ == "__main__":
    # Test script execution
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_dir = os.path.join(base_dir, 'data', 'processed')
        
        print("Loading preprocessed training data for model test...")
        X_train = joblib.load(os.path.join(processed_dir, 'X_train.joblib'))
        y_train = joblib.load(os.path.join(processed_dir, 'y_train.joblib'))
        
        # Test training all models
        models = train_all_models(X_train, y_train)
        
        # Test hyperparameter tuning
        best_rf = hyperparameter_tuning(X_train, y_train)
        
        # Save the best model
        model_path = os.path.join(processed_dir, 'best_model.joblib')
        joblib.dump(best_rf, model_path)
        print(f"\nSaved tuned model to {model_path}")
        
    except FileNotFoundError:
        print("Could not find processed data. Run preprocessing.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")
