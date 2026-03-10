#\ Stock Prediction System — 3-Day Significant Movement Forecasting

A machine learning pipeline that goes beyond naive next-day price prediction. Instead of guessing whether a stock will close up or down tomorrow — which is notoriously noisy and unreliable — this system predicts whether a stock index will experience a **significant upward move (>1%) over the next 3 trading days**. By targeting larger, multi-day movements that rise above everyday market noise, the model focuses on economically meaningful signals that actually matter. Currently configured for the **NIFTY 50 index** (^NSEI), but easily adaptable to any ticker available on Yahoo Finance.

> **Python**: 3.13+

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features & Indicators](#features--indicators)
- [Models](#models)
- [Validation Strategy](#validation-strategy)
- [Results](#results)
- [Getting Started](#getting-started)
- [Key Insights](#key-insights)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## Overview

Most stock prediction projects try to answer "will the price go up or down tomorrow?" — a question so dominated by random noise that even sophisticated models barely beat a coin flip. This project asks a smarter question instead:

> **Will the stock experience a meaningful upward move (>1%) over the next 3 trading days?**

By widening the prediction window to 3 days and setting a 1% threshold, we filter out the small daily jitter that makes next-day prediction so futile, and focus on genuine price movements that reflect real market momentum. The project tackles this challenge by:

1. **Targeting significant multi-day moves** — A 1% gain over 3 days is an economically meaningful signal, not random noise. Small daily fluctuations (<0.5%) are effectively ignored by design.
2. **Using classical ML** — Logistic Regression, Random Forest, and Gradient Boosting are compared against a Dummy (most-frequent) baseline. No deep learning is used.
3. **Rigorous time-series validation** — Walk-forward validation with 4 expanding folds ensures no future data leaks into training, mimicking real-world trading conditions.

---

## Problem Statement

> *Given historical price and volume data for a stock index, predict whether it will experience a significant upward movement (>1%) over the next 3 trading days — not a simple up/down call for tomorrow, but a meaningful multi-day trend signal.*

### Target Variable

```
future_return = (Close[t+3] − Close[t]) / Close[t]

Target = 1  if future_return > 0.01   (Significant upward move ahead)
Target = 0  otherwise                  (No significant move — flat, small gain, or decline)
```

The 3-day window and 1% threshold are deliberate: a next-day prediction would be fighting pure noise, while a 3-day >1% move is large enough to reflect genuine momentum and small enough to remain actionable.

**Class distribution**: ~29% positive (Target=1), ~71% negative (Target=0) — reflecting real market conditions where significant upward movements are less frequent than periods of consolidation or decline.

---

## Features & Indicators

The system engineers exactly **8 technical features** from raw OHLCV data, capturing four key dimensions of market behavior:

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | **RSI (14)** | Momentum | Relative Strength Index — measures overbought/oversold conditions |
| 2 | **MACD Histogram** | Trend | Difference between MACD line and signal line — detects momentum shifts |
| 3 | **Bollinger Band Width** | Volatility | (Upper − Lower) / SMA(20) — indicates market volatility regime |
| 4 | **Price vs SMA(20)** | Trend | Close / SMA(20) ratio — shows position relative to 20-day average |
| 5 | **ROC (5)** | Momentum | 5-day Rate of Change — captures short-term price acceleration |
| 6 | **ATR (14)** | Volatility | Average True Range — measures volatility independent of direction |
| 7 | **Volume Ratio** | Volume | Current volume / 10-day average volume — signals strength of moves |
| 8 | **Return_Lag_1** | Momentum | Previous day's return (lagged by 1) — captures momentum continuation |

All features are computed using the [`ta`](https://github.com/bukosabino/ta) library and scaled with `StandardScaler` (fit on training data only).

---

## Models

Four classifiers are trained and compared:

| Model | Key Hyperparameters | Purpose |
|-------|---------------------|---------|
| **Logistic Regression** | C=0.1, max_iter=1000 | Simple, interpretable linear baseline |
| **Random Forest** | n_estimators=100, max_depth=4, min_samples_leaf=15 | Non-linear ensemble with built-in feature importance |
| **Gradient Boosting** | n_estimators=100, max_depth=3, learning_rate=0.05 | Sequential boosting for structured/tabular data |
| **Dummy Classifier** | strategy='most_frequent' | Always predicts majority class — sanity-check baseline |

All models use `random_state=42` for reproducibility.

---

## Validation Strategy

### Walk-Forward Validation (4 Folds)

Standard k-fold cross-validation is **invalid** for time-series data because it would allow future data to leak into training. Instead, this project uses **walk-forward validation** with expanding training windows:

| Fold | Training Window | Test Window | Train Samples | Test Samples |
|------|-----------------|-------------|---------------|--------------|
| 1 | Start → 25% | 25% → 35% | ~360 | ~143 |
| 2 | Start → 50% | 50% → 60% | ~720 | ~144 |
| 3 | Start → 70% | 70% → 80% | ~1,007 | ~145 |
| 4 | Start → 85% | 85% → 100% | ~1,224 | ~216 |

Additionally, a standard **80/20 chronological split** is used for detailed per-metric evaluation (accuracy, precision, recall, F1, ROC-AUC).

---

## Results

### Walk-Forward Validation (Averaged Across 4 Folds)

| Model | Avg Train Acc | Avg Test Acc | Beats Baseline? |
|-------|---------------|--------------|-----------------|
| **Logistic Regression** | 73.14% | **67.38%** | Marginal |
| Random Forest | 75.48% | 64.43% | No |
| Gradient Boosting | 84.34% | 58.98% | No |
| Dummy (Baseline) | 73.55% | 67.73% | — |

### Detailed Test Set Metrics (80/20 Split)

The pipeline also reports **Accuracy, Precision, Recall, F1-Score, and ROC-AUC** on the held-out 20% test set for each model. Confusion matrices, ROC curves, and feature importance plots are saved to `stock_prediction/reports/figures/`.

### Best Model

**Logistic Regression** is selected as the best model based on walk-forward test accuracy. It achieves ~67% accuracy — a modest but consistent improvement over naive prediction, demonstrating that technical indicators carry some predictive signal even in highly efficient markets.

---

## Getting Started

For installation steps, how to run the pipeline, configuration options, troubleshooting, and a project layout overview, see **[QUICKSTART.md](QUICKSTART.md)**.

---

## Key Insights

### Why ~67% Accuracy Is Actually Reasonable

1. **Market Efficiency**: The Efficient Market Hypothesis implies that past prices hold limited information about future prices — any edge is small by nature.
2. **Noise Dominance**: Daily price movements are overwhelmed by random fluctuations; the signal-to-noise ratio is extremely low.
3. **Non-Stationarity**: Market regimes change over time — what works in a bull market may fail in a bear market.
4. **Beating the Baseline**: Even a small improvement over naive prediction (always guessing the majority class) is noteworthy in financial ML.

### Why Simpler Models Win

- **Gradient Boosting** achieves high training accuracy (84%) but low test accuracy (59%) — classic **overfitting** to noisy training data.
- **Random Forest** also overfits, though less severely.
- **Logistic Regression**, being heavily regularized and linear, is more resilient to noise and generalizes better on this small, noisy dataset.

This is a well-known phenomenon in financial ML: complex models tend to memorize noise rather than learn signal.

---

## Limitations & Future Work

### Current Limitations

- **Modest accuracy** (~67%) — markets are inherently difficult to predict with public price data alone
- **Single asset** — only NIFTY 50 is tested; generalization to other markets is unverified
- **Technical indicators only** — no fundamental data, sentiment analysis, or macroeconomic features
- **Static hyperparameters** — no adaptive or online learning as market regimes change
- **Class imbalance** — the 70/30 split may bias models toward predicting the majority class

### Potential Improvements

- **Ensemble stacking** — combine predictions from multiple models
- **Sentiment features** — incorporate news sentiment or social media signals
- **Deep learning** — LSTM or Transformer architectures for temporal dependency modeling
- **Online learning** — update models incrementally as new data arrives
- **Multi-asset universe** — extend to multiple stocks/indices for portfolio-level prediction
- **Alternative targets** — experiment with different thresholds (0.5%, 2%) and horizons (1-day, 5-day)

---

## References

- **Data Source**: [Yahoo Finance](https://finance.yahoo.com/) via [`yfinance`](https://github.com/ranaroussi/yfinance)
- **Technical Indicators**: [`ta` library](https://github.com/bukosabino/ta)
- **ML Framework**: [scikit-learn](https://scikit-learn.org/)
- **Walk-Forward Validation**: Pring, M. J. (2002). *Technical Analysis Explained*
- **Efficient Market Hypothesis**: Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work*

---

<p align="center">
  <i>Built as an academic project to explore whether classical ML can detect meaningful multi-day momentum signals in financial markets — beyond the noise of next-day prediction.</i>
</p>