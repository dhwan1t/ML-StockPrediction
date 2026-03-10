# Quick Start Guide

Get the Stock Prediction System up and running in under 5 minutes. This pipeline doesn't try to guess whether the market will close up or down tomorrow — instead, it predicts whether a stock index will experience a **significant upward move (>1%) over the next 3 trading days**, targeting real momentum signals rather than daily noise.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.13+ (the included virtual environment uses 3.13) |
| **OS** | macOS (tested), Linux, or Windows with minor path adjustments |
| **Internet** | Required on first run to download stock data from Yahoo Finance |

---

## 1. Activate the Virtual Environment

```bash
cd "Sem 4 Mini project"
source stock_env/bin/activate
```

> **Windows users**: use `stock_env\Scripts\activate` instead.

You should see `(stock_env)` in your terminal prompt.

---

## 2. Install Dependencies

```bash
pip install -r stock_prediction/requirements.txt
```

This installs: `pandas`, `numpy`, `scikit-learn`, `yfinance`, `ta`, `matplotlib`, `seaborn`, `joblib`, and `imbalanced-learn`.

---

## 3. Run the Pipeline

```bash
cd stock_prediction
python main.py
```

That's it. The script handles everything end-to-end:

1. Downloads historical stock data (2016–2022) from Yahoo Finance (NIFTY 50 by default)
2. Creates the binary target (will the price rise >1% over the next 3 trading days?)
3. Engineers 8 technical indicator features
4. Splits data chronologically (80/20)
5. Runs walk-forward validation (4 folds)
6. Trains & evaluates Logistic Regression, Random Forest, Gradient Boosting, and a Dummy baseline
7. Saves the best model to `models/best_model.pkl`
8. Prints a full results summary to your terminal

**Expected runtime**: ~30–60 seconds (depending on network speed for the data download).

---

## 4. Read the Output

When the pipeline finishes, you'll see a summary like this:

```
********************************************************************************
                         PIPELINE EXECUTION SUMMARY
********************************************************************************
Index Analyzed:         ^NSEI
Date Range:             2016-01-01 to 2022-01-01
Total Samples:          ~1440
Features Used:          8 (RSI, MACD, BB, Price/SMA, ROC, ATR, Vol Ratio, Return Lag)
Validation Method:      Walk-Forward (4 folds)
Test Split Strategy:    Last 20% (chronological)
--------------------------------------------------------------------------------
BEST PERFORMING MODEL:
  Name:                 LogisticRegression
  Avg WF Test Acc:      ~0.6738
  Beats Baseline:       Yes
********************************************************************************
```

---

## Optional: Explore Further

### View Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

### Auto-Update the Academic Report

```bash
python run_and_update_report.py
```

This re-runs the pipeline and injects the latest metrics into `reports/REPORT.md`.

### Read the Full Report

Open `reports/REPORT.md` for a detailed academic write-up covering methodology, results, and analysis.

### Check Generated Visualizations

Browse `reports/figures/` for confusion matrices, ROC curves, feature importance plots, and prediction timelines for every model.

---

## Customization

To predict a different stock or index, edit the `CONFIG` block at the top of `main.py`:

```python
CONFIG = {
    "TICKER": "^NSEI",          # Any valid Yahoo Finance ticker (e.g., "AAPL", "^GSPC")
    "START_DATE": "2016-01-01", # Training data start
    "END_DATE": "2022-01-01",   # Training data end
}
```

Then re-run `python main.py`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Make sure the virtual environment is activated and dependencies are installed |
| `No data found for ticker` | Check your internet connection; verify the ticker exists on Yahoo Finance |
| `yfinance` download hangs | Yahoo Finance may be rate-limiting; wait a minute and retry |
| Plots not appearing | Plots are saved to `reports/figures/`, not displayed interactively |
| Different results than README | Market data downloads can vary slightly; small metric differences are normal |

---

## Project Layout at a Glance

```
stock_prediction/
├── main.py                    ← Run this
├── src/
│   ├── data_loader.py         ← Downloads data & creates target
│   ├── feature_engineering.py ← 8 technical indicators
│   ├── preprocessing.py       ← Chronological split & scaling
│   ├── model.py               ← Model definitions
│   └── evaluate.py            ← Metrics & visualizations
├── models/                    ← Saved model artifacts
├── data/raw/                  ← Downloaded CSVs
├── data/processed/            ← Serialized train/test splits
├── notebooks/eda.ipynb        ← Exploratory analysis
├── reports/REPORT.md          ← Full academic report
└── reports/figures/           ← Generated plots
```

---

*For the full project documentation, see [README.md](README.md).*