# Forecasting-S&P500-Direction

## 📌 Project Overview
Financial markets are characterized by high noise, non-stationarity, and strong influence from macroeconomic and behavioral factors. This project simplifies the complex problem of price level prediction into a binary classification of price direction.

The core research question is: **Does combining simple machine learning models with diverse data sources (technical, macro, and calendar) provide a statistical advantage over random guessing?**.

## 🎯 Problem Definition
  **Task Type:** Binary Classification.
  **Time Interval:** Daily.
  **Target Variable:** $y_{t+1}=1$ if $Close_{t+1} > Close_{t}$, otherwise 0.
  **Base Instrument:** SPY ETF (S&P 500) due to high liquidity and lower noise.
  **Success Metric:** Achieve accuracy between 52-55% on test data.

## 📊 Data & Features
The model utilizes 10-15 years of historical data from multiple sources:

### 1. Market Data (yfinance)
* OHLCV data and technical indicators.
* Features: RSI, MACD, SMA (50/200), Historical Volatility, and Bollinger Bands.

### 2. Macroeconomic Data (FRED)
* Fed Funds Rate, 10Y and 2Y Bond Yields, and the 10Y-2Y Spread.
* VIX Index and US Dollar Index (DXY).

### 3. Calendar & Event Features
* Day of the week (one-hot encoding) and Month.
* Anomalies: "Sell in May" effect and "Santa Claus Rally".
* Event-based: FOMC meeting days and US Election years.

## 🧠 Machine Learning Architecture
The project implements an **Ensemble Learning** strategy using a `VotingClassifier` with **Majority Voting (hard)**.

| Model | Justification |
| :--- | :--- |
| **Logistic Regression (L2)** | Base model, high interpretability. |
| **Random Forest (shallow)** | Capturing non-linearities. |
| **SVM (RBF)** | Handling periods of high volatility. |
| **Naive Bayes** | Efficient for binary and event features. |

## 🧪 Validation & Methodology
To ensure methodological correctness and prevent **Data Leakage**, we use:
  **Walk-Forward Validation:** Sliding training window with no data shuffling to simulate real-world usage.
  **Evaluation Metrics:** Accuracy (primary), F1-score, and Confusion Matrix.
  **Benchmarks:** Comparison against "Coin Flip" and "Always Up" strategies.

## 🛠 Tech Stack
* Will be soon

## ⚠️ Limitations
* No transaction costs included.
* Market non-stationarity and "Black Swan" events.
* Accuracy does not always equal profitability.




