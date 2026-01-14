# Forecasting Stock Price Direction 

This project aims to build a robust machine learning ensemble system to predict the daily direction of stock price movements (Binary Classification: Up/Down). By combining technical analysis indicators with macroeconomic data and sentiment analysis, the model seeks to achieve a stable edge in financial forecasting.

## Project Goal
The primary objective is to predict whether the closing price of an asset (e.g., MSFT, AAPL, GOOGL) on day $t+1$ will be higher (**1**) or lower (**0**) than on day $t$. The project tests the hypothesis that a **Majority Voting Ensemble** of diverse classifiers can outperform individual "weak" learners.

## Methodology & Pipeline

### 1. Data Sources & Integration
We utilize a multi-dimensional approach to data:
* **Market Data (OHLCV):** Historical prices sourced via `yfinance`.
* **Technical Indicators:** Calculations of RSI, MACD, Bollinger Bands, and Moving Averages using `pandas_ta`.
* **Macroeconomic Indicators:** Integration of VIX (Fear Index), Treasury Yields, and Interest Rates (FRED data).
* **Sentiment Analysis:** Processing financial news/sentiment using the `vaderSentiment` library.

### 2. Feature Engineering
* Logarithmic returns for stationarity.
* Standardization of features using `StandardScaler`.
* Correlation analysis (Heatmaps) to identify and remove redundant features.

### 3. Machine Learning Strategy (In Progress)
The core of the project is an **Ensemble Strategy** consisting of:
* **Model A (Linear):** Logistic Regression with L2 regularization (Current Baseline).
* **Model B (Tree-based):** Random Forest with limited `max_depth` to prevent overfitting.
* **Model C (Distance-based):** SVM (Support Vector Machine) with RBF kernel.
* **Model D (Boosting):** XGBoost/LightGBM for capturing non-linear patterns.
* **Final Decision:** `VotingClassifier` (Hard/Soft voting) from `scikit-learn`.



### 4. Validation & Testing
* **Walk-Forward Validation:** Ensuring no data leakage by maintaining chronological order (no shuffling).
* **Metrics:** Accuracy, F1-Score, and Confusion Matrix analysis.

## Current Status
The project is currently in the **Baseline & Data Engineering** phase:
- [x] Data collection script for major tech tickers (AAPL, MSFT, GOOGL).
- [x] Integration of Technical Indicators and VIX data.
- [x] Implementation of the Baseline Logistic Regression model.
- [ ] Development of individual ensemble members (RF, SVM, XGBoost).
- [ ] Final Voting Classifier implementation.
- [ ] Backtesting and result visualization.

## Technology Stack
* **Language:** Python 3.x
* **Data Handling:** `pandas`, `numpy`
* **ML Libraries:** `scikit-learn`, `xgboost`
* **Financial Tools:** `yfinance`, `pandas_ta`
* **NLP:** `vaderSentiment`
* **Visualization:** `matplotlib`, `seaborn`
