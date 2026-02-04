# Forecasting Stock Price Direction: A Robust Ensemble Approach

## Abstract
This project presents a machine learning framework designed to forecast the directional movement of major technology stocks (AAPL, MSFT, GOOGL). Addressing the low signal-to-noise ratio characteristic of financial markets, the system employs a **Voting Ensemble** of diverse classifiers. A critical component of this research is the implementation of rigorous validation techniques, specifically **Purged Walk-Forward Validation** with embargo, to eliminate look-ahead bias and data leakage. Furthermore, the project contrasts standard binary classification with the **Triple Barrier Method (TBM)**, introducing dynamic volatility-based labeling to filter market noise.

## 1. Project Overview
The primary objective is to predict price directionality over a short-term horizon. The project investigates two distinct labeling strategies to address the stochastic nature of financial time series:
1.  **Binary Classification with Volatility Filtering:** Predicting Up/Down movements while filtering out insignificant price changes ("dead zones") based on daily volatility thresholds.
2.  **Triple Barrier Method (TBM):** A multi-class approach (Profit, Loss, Neutral) that defines success based on dynamic profit-taking and stop-loss barriers relative to asset volatility.

## 2. Methodology

### 2.1. Data Acquisition and Preprocessing
The dataset integrates multi-modal data sources to capture a holistic market view:
* **Market Data:** Historical OHLCV data for AAPL, MSFT, and GOOGL sourced via `yfinance`.
* **Macroeconomic Indicators:** Integration of Federal Reserve data (FRED) including CPI, Treasury Yields (DGS10), Federal Funds Rate, and the Dollar Index to capture systemic risk.
* **Sentiment Analysis:** NLP-derived sentiment scores from financial news headlines utilizing **FinBERT** (financial domain-specific) and **VADER**, alongside the VIX (Fear Index).

### 2.2. Feature Engineering
Raw data is transformed into stationary features suitable for machine learning algorithms:
* **Technical Indicators:** Calculation of RSI, MACD, Bollinger Bands, Moving Averages, and volatility metrics (ATR) using `pandas_ta`.
* **Stationarity Transformations:** Logarithmic returns and differencing are applied to non-stationary time series.
* **Feature Selection:** Recursive Feature Elimination (RFE) and correlation matrix analysis are employed to reduce dimensionality and mitigate multicollinearity.

### 2.3. Labeling Strategies
To enhance signal quality, static thresholds were replaced with dynamic, volatility-adjusted barriers:
* **Dynamic TBM:** Barriers are set as a function of the rolling standard deviation ($\sigma$). This ensures the model adapts to changing market regimes (high vs. low volatility periods).
* **Volatility Filter (Binary):** A dynamic buffer is implemented where observations with returns $<\mid k \cdot \sigma \mid$ are excluded from training. This prevents the model from fitting to market noise in low-volatility environments.

### 2.4. Model Architecture
The core predictive engine is a **Voting Ensemble** (`VotingClassifier`) comprising:
* **Logistic Regression:** Serving as a linear baseline with L2 regularization.
* **Random Forest:** Capturing non-linear interactions via bagging.
* **XGBoost:** Gradient boosting for handling complex patterns and feature importance derivation.
* **Support Vector Machine (SVM):** Utilizing RBF kernels for high-dimensional separation.

Hyperparameters for individual estimators are optimized using **Optuna**.

## 3. Validation Framework
A cornerstone of this project is the rejection of standard K-Fold cross-validation in favor of methods strictly respecting temporal order, following the methodology of *Advances in Financial Machine Learning* (Lopez de Prado):

* **Purged Walk-Forward Validation:** The dataset is split into expanding training windows and sliding test windows. Crucially, training samples that overlap temporally with the test set are **purged** to prevent data leakage.
* **Embargo:** A temporal gap is enforced after the training set to further insulate the test set from correlation leakage.
* **Statistical Significance:** **Block Bootstrapping** is applied to model predictions to generate confidence intervals for performance metrics, ensuring results are statistically robust.

## 4. Repository Structure

```text
├── Baseline_model/          # Implementation of individual baseline models (LR, RF, SVM, XGB)
├── data/                    # Dataset storage (Market, Macro, Sentiment)
├── main_model/              # Core production code
│   ├── functions/           # Helper libraries (functions_all.py containing validation logic)
│   ├── results/             # JSON and CSV outputs of model performance
│   ├── feature_selector.ipynb  # RFE and feature importance analysis
│   └── wf_bt.ipynb          # Main execution pipeline (Walk-Forward Backtest)
├── models_results/          # Detailed logs and classification reports
├── selected_features/       # JSON files containing optimal feature sets per ticker
├── feature_engineering.ipynb # Data transformation pipelines
└── vizualization.ipynb      # Performance plotting (ROC curves, Confusion Matrices) 
```



## 5. Results
The model performance is evaluated using accuracy, precision, recall, and ROC-AUC scores. Results indicate that the application of volatility-based filtering and the ensemble approach yields a measurable edge over random baseline strategies. The TBM approach specifically highlights the trade-off between trade frequency and precision. Detailed performance metrics for each ticker (AAPL, MSFT, GOOGL) are available in the `main_model/results/` directory.

## 6. Technologies
* **Core:** Python 3.x
* **Data Analysis:** `pandas`, `numpy`, `pandas_ta`
* **Machine Learning:** `scikit-learn`, `xgboost`, `optuna`
* **NLP:** `transformers` (Hugging Face), `vaderSentiment`
* **Visualization:** `matplotlib`, `seaborn`

## 7. Future Work
* Implementation of Deep Learning architectures (LSTM/GRU) to explicitly model sequential dependencies.
* Expansion of the asset universe to include non-tech sectors.
* Integration of alternative data sources (e.g., social media sentiment volume).

---
**License:** MIT

