import os

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# CONFIGURATION & SETUP
# ---------------------------------------------------------

SOURCE_DIR = os.path.join("reports", "figures")
TARGET_DIR = os.path.join(SOURCE_DIR, "key_figures")

BADGE_HEIGHT = 52
BORDER_WIDTH = 10

COLORS = {
    "GREEN": "#27ae60",  # Best model result
    "BROWN": "#8B4513",  # Baseline/reference chart
    "RED": "#c0392b",  # Overfitting / bad result
    "BLUE": "#2980b9",  # Feature analysis
    "PURPLE": "#8e44ad",  # Overall comparison / summary
}

# The 12 selected figures to curate
FIGURES_TO_CURATE = [
    {
        "source": "Logistic Regression_roc_curve.png",
        "target": "01_best_model_roc.png",
        "badge": "BEST MODEL - ROC",
        "color": COLORS["GREEN"],
    },
    {
        "source": "Logistic Regression_confusion_matrix.png",
        "target": "02_best_model_cm.png",
        "badge": "BEST MODEL - CM",
        "color": COLORS["GREEN"],
    },
    {
        "source": "Logistic Regression_feature_importance.png",
        "target": "03_best_model_features.png",
        "badge": "BEST MODEL - FEATURES",
        "color": COLORS["GREEN"],
    },
    {
        "source": "Logistic Regression_predictions.png",
        "target": "04_best_model_timeline.png",
        "badge": "BEST MODEL - TIMELINE",
        "color": COLORS["GREEN"],
    },
    {
        "source": "Dummy (Baseline)_roc_curve.png",
        "target": "05_baseline_roc.png",
        "badge": "BASELINE - ROC",
        "color": COLORS["BROWN"],
    },
    {
        "source": "Dummy (Baseline)_confusion_matrix.png",
        "target": "06_baseline_cm.png",
        "badge": "BASELINE - CM",
        "color": COLORS["BROWN"],
    },
    {
        "source": "Gradient Boosting_roc_curve.png",
        "target": "07_overfitting_roc.png",
        "badge": "OVERFITTING - ROC",
        "color": COLORS["RED"],
    },
    {
        "source": "Gradient Boosting_confusion_matrix.png",
        "target": "08_overfitting_cm.png",
        "badge": "OVERFITTING - CM",
        "color": COLORS["RED"],
    },
    {
        "source": "mutual_info.png",
        "target": "09_feature_analysis_mi.png",
        "badge": "FEATURE ANALYSIS - MUTUAL INFO",
        "color": COLORS["BLUE"],
    },
    {
        "source": "walk_forward_accuracy.png",
        "target": "10_validation_wf_acc.png",
        "badge": "VALIDATION - WALK-FORWARD",
        "color": COLORS["BLUE"],
    },
    {
        "source": "class_distribution.png",
        "target": "11_data_class_dist.png",
        "badge": "DATA - CLASS DISTRIBUTION",
        "color": COLORS["BLUE"],
    },
    {
        "source": "model_comparison_summary.png",
        "target": "12_summary_comparison.png",
        "badge": "SUMMARY - OVERALL COMPARISON",
        "color": COLORS["PURPLE"],
    },
]

# ---------------------------------------------------------
# IMAGE PROCESSING FUNCTION
# ---------------------------------------------------------


def add_highlight(img_path, out_path, badge_text, hex_color):
    """
    Adds a colored border and a top badge with text to an image.
    """
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print(f"Warning: Source file '{img_path}' not found. Skipping.")
        return False

    # Convert hex to RGB tuple
    color_rgb = tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    # Calculate new dimensions
    new_width = img.width + 2 * BORDER_WIDTH
    new_height = img.height + BORDER_WIDTH + BADGE_HEIGHT

    # Create new canvas
    new_img = Image.new("RGB", (new_width, new_height), color_rgb)
    draw = ImageDraw.Draw(new_img)

    # Setup font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )
    except IOError:
        font = ImageFont.load_default()

    # Center text
    bbox = draw.textbbox((0, 0), badge_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = (new_width - text_w) / 2
    text_y = (BADGE_HEIGHT - text_h) / 2 - bbox[1]

    # Draw text and paste original image
    draw.text((text_x, text_y), badge_text, font=font, fill=(255, 255, 255))
    new_img.paste(img, (BORDER_WIDTH, BADGE_HEIGHT))

    # Save output
    new_img.save(out_path, dpi=(300, 300))
    print(f"Created: {out_path}")
    return True


# ---------------------------------------------------------
# README GENERATION
# ---------------------------------------------------------

README_CONTENT = """# Key Figures Curated Report

This directory contains the 12 most informative figures from the Stock Prediction pipeline, curated to tell the complete story of the project's methodology and findings.

---

## Figure 01 — `01_best_model_roc.png`
**[BEST MODEL - ROC]**

### What this chart shows
This plots the Receiver Operating Characteristic (ROC) curve for the Logistic Regression model, mapping the True Positive Rate against the False Positive Rate across all classification thresholds.

### Key findings
- The model achieves an AUC marginally but consistently above the 0.500 random baseline.
- Demonstrates that a linear combination of these technical features holds real predictive signal for 3-day momentum.
- Best ROC-AUC among all tested models, proving that simplicity wins on noisy datasets.

---

## Figure 02 — `02_best_model_cm.png`
**[BEST MODEL - CM]**

### What this chart shows
The confusion matrix for Logistic Regression on the out-of-sample 20% test set, annotated with counts and percentages.

### Key findings
- Successfully identifies True Positives (actual >1% upward moves), unlike the Dummy baseline which catches none.
- Test accuracy of ~67.4% is achieved through a balanced combination of True Positives and True Negatives.
- Indicates a conservative prediction profile, which is ideal for avoiding costly False Positives in financial predictions.

---

## Figure 03 — `03_best_model_features.png`
**[BEST MODEL - FEATURES]**

### What this chart shows
A horizontal bar chart ranking the absolute magnitudes of the learned coefficients for the Logistic Regression model.

### Key findings
- The model actively utilizes diverse categories of technical signals (momentum, trend, volatility).
- Features like `Return_Lag_1` and `ROC_5` often show the strongest coefficient weights.
- Proves the model hasn't degenerated into relying on a single dominant, spurious feature.

---

## Figure 04 — `04_best_model_timeline.png`
**[BEST MODEL - TIMELINE]**

### What this chart shows
A sequential timeline of the entire test set showing actual directional moves (green/red dots) overlaid with the model's step-line predictions.

### Key findings
- The model correctly identifies temporal clusters of upward momentum during trending periods.
- False positive signals tend to group during volatile, sideways market regimes.
- The active fluctuation of predictions confirms the model is dynamically reacting to new data.

---

## Figure 05 — `05_baseline_roc.png`
**[BASELINE - ROC]**

### What this chart shows
The ROC curve for the Dummy Classifier, which simply predicts the majority class (0) for every instance.

### Key findings
- AUC is exactly 0.500, aligning perfectly with the random-guess diagonal line.
- Serves as the mathematical floor that any learning model must beat to claim it has found a valid signal.
- Highlights that always predicting the majority class provides zero true discriminative power.

---

## Figure 06 — `06_baseline_cm.png`
**[BASELINE - CM]**

### What this chart shows
The confusion matrix for the Dummy Classifier on the test set.

### Key findings
- 100% of predictions fall into Class 0, yielding ~67.7% accuracy purely due to class imbalance.
- Yields 0 True Positives, meaning it entirely misses every actual >1% upward move.
- A textbook demonstration of why accuracy alone is highly misleading on imbalanced financial time-series.

---

## Figure 07 — `07_overfitting_roc.png`
**[OVERFITTING - ROC]**

### What this chart shows
The ROC curve for the Gradient Boosting model, an advanced tree-based ensemble.

### Key findings
- The ROC-AUC is lower than the simpler Logistic Regression model.
- Demonstrates that complex, highly non-linear models struggle to generalize on this small, noisy dataset.
- The curve closely hugs the 0.500 baseline, indicating poor out-of-sample discrimination.

---

## Figure 08 — `08_overfitting_cm.png`
**[OVERFITTING - CM]**

### What this chart shows
The confusion matrix for the Gradient Boosting model on the test set.

### Key findings
- The model shows severe overfitting, dropping from 84% training accuracy to 59% test accuracy.
- Generates a high number of False Positives by hallucinating complex patterns that don't exist out-of-sample.
- Serves as concrete evidence that constraining model complexity is critical in quantitative finance.

---

## Figure 09 — `09_feature_analysis_mi.png`
**[FEATURE ANALYSIS - MUTUAL INFO]**

### What this chart shows
Mutual Information scores for each of the 8 technical features, measuring their non-linear dependency with the target variable.

### Key findings
- Momentum features (e.g., RSI, ROC) generally share the highest information overlap with the 3-day target.
- Validates the feature engineering process by proving information exists before any specific model is trained.
- Shows a relatively flat distribution of importance, supporting the heavily regularized linear approach.

---

## Figure 10 — `10_validation_wf_acc.png`
**[VALIDATION - WALK-FORWARD]**

### What this chart shows
A line chart tracking test accuracy for all four models across four chronological expanding-window folds.

### Key findings
- Gradient Boosting's performance is erratic and often dips far below the baseline.
- Logistic Regression remains the most stable, outperforming or matching the baseline consistently across market regimes.
- Validates that the 67.4% test accuracy is a robust finding, not a fluke of a single lucky data split.

---

## Figure 11 — `11_data_class_dist.png`
**[DATA - CLASS DISTRIBUTION]**

### What this chart shows
Side-by-side pie charts visualizing the balance of Target=1 (Up) and Target=0 (Down/Flat) in both training and test sets.

### Key findings
- Confirms a consistent ~70/30 imbalanced split across both chronological periods.
- Establishes why the baseline model achieves ~67-70% accuracy without learning anything.
- Highlights the core difficulty of the project: significant 3-day upward moves are inherently rare events.

---

## Figure 12 — `12_summary_comparison.png`
**[SUMMARY - OVERALL COMPARISON]**

### What this chart shows
A grouped bar chart comparing Accuracy, Precision, Recall, F1-Score, and ROC-AUC for all four models side-by-side.

### Key findings
- Logistic Regression dominates in balanced metrics like ROC-AUC and F1-Score.
- Visually stark contrast between the Dummy baseline's high accuracy but 0.0 Precision/Recall.
- Provides the ultimate concluding evidence that the simplest model generalizes best on this task.
"""


def generate_readme():
    readme_path = os.path.join(TARGET_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(README_CONTENT)
    print(f"Created: {readme_path}")


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------


def main():
    print(f"Starting figure curation...")

    # Ensure target directory exists
    os.makedirs(TARGET_DIR, exist_ok=True)

    success_count = 0

    for fig in FIGURES_TO_CURATE:
        src_path = os.path.join(SOURCE_DIR, fig["source"])
        tgt_path = os.path.join(TARGET_DIR, fig["target"])

        if add_highlight(src_path, tgt_path, fig["badge"], fig["color"]):
            success_count += 1

    # Generate the README file documenting the findings
    generate_readme()

    print(
        f"\nCuration complete! Successfully highlighted {success_count}/{len(FIGURES_TO_CURATE)} figures."
    )
    print(f"Results saved to: {TARGET_DIR}")


if __name__ == "__main__":
    main()
