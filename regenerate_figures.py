import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

# Global Style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

MODEL_COLORS = {
    "Logistic Regression": "#2ecc71",
    "Random Forest": "#3498db",
    "Gradient Boosting": "#e74c3c",
    "Dummy (Baseline)": "#95a5a6",
}

REPORTS_DIR = os.path.join("reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)


def clean_feature_names(cols):
    return [col[0] if isinstance(col, tuple) else str(col) for col in cols]


def main():
    print("Loading data...")
    try:
        X_train = joblib.load("data/processed/X_train.joblib")
        X_test = joblib.load("data/processed/X_test.joblib")
        y_train = joblib.load("data/processed/y_train.joblib")
        y_test = joblib.load("data/processed/y_test.joblib")
        print("Data loaded from processed directory.")
    except FileNotFoundError:
        print("Processed data not found, running pipeline...")
        from src.data_loader import create_target_variable, download_stock_data
        from src.feature_engineering import build_features
        from src.preprocessing import prepare_dataset

        df_raw = download_stock_data("^NSEI", start="2016-01-01", end="2022-01-01")
        df_targeted = create_target_variable(df_raw)
        df_features = build_features(df_targeted)
        feature_cols = [col for col in df_features.columns if col != "Target"]
        X_train, X_test, y_train, y_test, scaler = prepare_dataset(
            df_features, feature_cols
        )

    # Recombine for full dataset walk-forward
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    N = len(X_full)

    models = {
        "Logistic Regression": LogisticRegression(
            C=0.1, max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=15, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        ),
        "Dummy (Baseline)": DummyClassifier(strategy="most_frequent", random_state=42),
    }

    # 1. Walk-Forward Validation
    print("Running Walk-Forward Validation...")
    fold_config = [
        {"train_end": int(N * 0.25), "test_end": int(N * 0.35)},
        {"train_end": int(N * 0.50), "test_end": int(N * 0.60)},
        {"train_end": int(N * 0.70), "test_end": int(N * 0.80)},
        {"train_end": int(N * 0.85), "test_end": N},
    ]

    wf_test_accuracies = {m: [] for m in models}

    for fold, config in enumerate(fold_config, 1):
        train_end, test_end = config["train_end"], config["test_end"]
        X_train_f = X_full.iloc[:train_end]
        y_train_f = y_full.iloc[:train_end]
        X_test_f = X_full.iloc[train_end:test_end]
        y_test_f = y_full.iloc[train_end:test_end]

        scaler = StandardScaler()
        X_train_f_scaled = scaler.fit_transform(X_train_f)
        X_test_f_scaled = scaler.transform(X_test_f)

        for name, model in models.items():
            model.fit(X_train_f_scaled, y_train_f)
            preds = model.predict(X_test_f_scaled)
            wf_test_accuracies[name].append(accuracy_score(y_test_f, preds))

    # Plot Walk-Forward Accuracy
    plt.figure(figsize=(10, 6))
    folds = [1, 2, 3, 4]
    for name, accs in wf_test_accuracies.items():
        if name == "Dummy (Baseline)":
            avg_dummy = np.mean(accs)
            plt.axhline(
                y=avg_dummy,
                color=MODEL_COLORS[name],
                linestyle="--",
                label=f"{name} Avg ({avg_dummy:.3f})",
            )
        else:
            plt.plot(
                folds,
                accs,
                marker="o",
                color=MODEL_COLORS[name],
                label=name,
                linewidth=2,
            )

    plt.title("Walk-Forward Test Accuracy Across 4 Folds")
    plt.xlabel("Fold Number")
    plt.ylabel("Test Accuracy")
    plt.xticks(folds)
    plt.ylim(0.4, 1.0)
    plt.legend()
    plt.savefig(
        os.path.join(REPORTS_DIR, "walk_forward_accuracy.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Train on full training set and generate model figures
    print("Training on full train set and generating figures...")

    # Scale X_train and X_test for final evaluation
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=clean_feature_names(X_train.columns),
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_train_scaled.columns, index=X_test.index
    )

    final_metrics = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        final_metrics.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": roc_auc,
            }
        )

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        total = len(y_test)
        labels = [[f"{v}\n({v / total * 100:.1f}%)" for v in row] for row in cm]

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=np.array(labels),
            fmt="",
            cmap="Blues",
            xticklabels=["Down (0)", "Up (1)"],
            yticklabels=["Down (0)", "Up (1)"],
        )
        plt.title(f"{name} - Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig(
            os.path.join(REPORTS_DIR, f"{name}_confusion_matrix.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # ROC Curve
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.3f})",
            )
            plt.plot(
                [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess"
            )
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{name} - ROC Curve")
            plt.legend(loc="lower right")
            plt.savefig(
                os.path.join(REPORTS_DIR, f"{name}_roc_curve.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # Predictions Timeline
        plot_len = len(y_test)
        actuals = y_test.values
        preds = y_pred
        x_steps = np.arange(plot_len)

        plt.figure(figsize=(14, 5))
        plt.scatter(
            x_steps[actuals == 1],
            actuals[actuals == 1],
            color="green",
            label="Actual Up",
            alpha=0.6,
            s=30,
        )
        plt.scatter(
            x_steps[actuals == 0],
            actuals[actuals == 0],
            color="red",
            label="Actual Down",
            alpha=0.6,
            s=30,
        )
        plt.step(
            x_steps,
            preds,
            color="black",
            label="Predicted",
            where="mid",
            alpha=0.5,
            linewidth=1.5,
        )
        plt.yticks([0, 1], ["Down (0)", "Up (1)"])
        plt.title(f"{name} - Actual vs Predicted (Full Test Set)")
        plt.xlabel("Days (Test Order)")
        plt.legend(loc="best")
        plt.savefig(
            os.path.join(REPORTS_DIR, f"{name}_predictions.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Feature Importance
        if name != "Dummy (Baseline)":
            importances = None
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0])

            if importances is not None:
                feat_series = pd.Series(
                    importances, index=X_train_scaled.columns
                ).sort_values(ascending=True)

                plt.figure(figsize=(10, 6))
                sns.barplot(
                    x=feat_series.values, y=feat_series.index, palette="viridis"
                )
                plt.title(f"{name} - Feature Importance")
                plt.xlabel("Importance / Absolute Coefficient")
                plt.savefig(
                    os.path.join(REPORTS_DIR, f"{name}_feature_importance.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    # 3. Summary Figures
    print("Generating summary figures...")

    # Model Comparison Summary
    df_metrics = pd.DataFrame(final_metrics)
    metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_melted = df_metrics.melt(
        id_vars="Model", value_vars=metrics_list, var_name="Metric", value_name="Score"
    )

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_melted, x="Metric", y="Score", hue="Model", palette=MODEL_COLORS
    )
    plt.title("Model Performance Comparison — 80/20 Chronological Test Split")
    plt.ylim(0, 1.1)

    # Add value labels
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(
        os.path.join(REPORTS_DIR, "model_comparison_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Class Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    train_counts = y_train.value_counts()
    ax1.pie(
        train_counts,
        labels=[
            f"Class {k}\n{v} ({v / len(y_train) * 100:.1f}%)"
            for k, v in train_counts.items()
        ],
        autopct="",
        colors=["#e74c3c", "#2ecc71"],
    )
    ax1.set_title("Train Set Target Distribution")

    test_counts = y_test.value_counts()
    ax2.pie(
        test_counts,
        labels=[
            f"Class {k}\n{v} ({v / len(y_test) * 100:.1f}%)"
            for k, v in test_counts.items()
        ],
        autopct="",
        colors=["#e74c3c", "#2ecc71"],
    )
    ax2.set_title("Test Set Target Distribution")

    plt.suptitle("Target Class Distribution — Train vs Test")
    plt.savefig(
        os.path.join(REPORTS_DIR, "class_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Feature Correlation Heatmap
    df_train_corr = X_train.copy()
    df_train_corr.columns = clean_feature_names(df_train_corr.columns)
    df_train_corr["Target"] = y_train
    corr = df_train_corr.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("Feature Correlation Matrix (Training Set)")
    plt.savefig(
        os.path.join(REPORTS_DIR, "feature_correlation_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Mutual Info & RF Importance (EDA)
    mi = mutual_info_classif(X_train_scaled, y_train, random_state=42)
    mi_series = pd.Series(mi, index=X_train_scaled.columns).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    mi_series.plot(kind="barh", color="teal")
    plt.title("Mutual Information with Target")
    plt.xlabel("Mutual Information Score")
    plt.savefig(
        os.path.join(REPORTS_DIR, "mutual_info.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    rf_eda = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_eda.fit(X_train_scaled, y_train)
    rf_series = pd.Series(
        rf_eda.feature_importances_, index=X_train_scaled.columns
    ).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    rf_series.plot(kind="barh", color="darkblue")
    plt.title("Random Forest Feature Importance (Default Settings)")
    plt.xlabel("Importance")
    plt.savefig(
        os.path.join(REPORTS_DIR, "rf_importance.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Print summary table
    print("\n" + "=" * 80)
    print(
        f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}"
    )
    print("-" * 80)
    for m in final_metrics:
        print(
            f"{m['Model']:<25} | {m['Accuracy']:<10.3f} | {m['Precision']:<10.3f} | {m['Recall']:<10.3f} | {m['F1-Score']:<10.3f} | {m['ROC-AUC']:<10.3f}"
        )
    print("=" * 80 + "\n")
    print("All figures successfully regenerated in 'reports/figures/'.")


if __name__ == "__main__":
    main()
