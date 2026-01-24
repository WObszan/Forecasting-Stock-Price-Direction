
### IMPORT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
import json


### just for test
aapl_with_features = pd.read_csv('../data/all_data/all_AAPL_data.csv')
googl_with_features = pd.read_csv('../data/all_data/all_GOOGL_data.csv')
msft_with_features = pd.read_csv('../data/all_data/all_MSFT_data.csv')

tickers = ['AAPL', 'GOOGL', 'MSFT']
data_dict = {
    'AAPL': aapl_with_features,
    'GOOGL': googl_with_features,
    'MSFT': msft_with_features
}
color_dict = {
     'AAPL': 'grey',
    'GOOGL': 'yellow',
    'MSFT': 'green'
}

#statistics = ['accuracy', 'precision', 'recall', 'roc_auc']
statistics = ['accuracy']



### Set the target
def get_target(input_df, ticker):
    df = input_df.copy()
    df['Target'] = (df[f'Close_{ticker}'].shift(-1) > df[f'Close_{ticker}']).astype(int)
    df.dropna(inplace=True)
    return df


###
def best_features(data_dict, tickers, statistics):
    warnings.filterwarnings('ignore')
    feature_dict = {}

    for i, share in enumerate(tickers):
        feature_dict[share] = {}
        df = get_target(data_dict[share], share)
        features = [col for col in df.columns if col not in ['Target', 'index', 'DATE']]

        to_remove = [f'Volume_{share}_lag1', f'Volume_{share}_lag2', f'Volume_{share}_lag3', f'Volume_{share}_lag5',
                         f'RSI_14_lag1', 'RSI_14_lag2', 'RSI_14_lag3', 'RSI_14_lag5', 'log_return_lag1', 'log_return_lag2',
                         'log_return_lag3', 'log_return_lag5']

        '''if share == 'AAPL':
            additional_to_remove = ['rolling_max_20', 'rolling_max_20', 'dist_to_max_20', 'dist_to_min_20', 'rolling_max_60', 'rolling_max_60', 'dist_to_max_60', 'dist_to_min_60']
            to_remove += additional_to_remove
            '''

        features = [f for f in features if f not in to_remove]

        X = df[features]
        y = df['Target']

        model_judge_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)

        # Second model to check
        model_judge_xgb = XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )

        ### test on different models for best acc for each company
        '''if share != 'MSFT':
            model_judge = model_judge_rf
        else:
            model_judge = model_judge_xgb
            '''

        model_judge = model_judge_rf

        for stat in statistics:
            cv_split = TimeSeriesSplit(n_splits=5)

            min_feats = 10 if share == 'AAPL' else 7
            rfecv = RFECV(
                estimator=model_judge,
                min_features_to_select=min_feats,
                step=1,
                cv=cv_split,
                scoring=stat,
                n_jobs=-1)

            rfecv.fit(X, y)

            print(f"Optimal features numer by RFECV : {rfecv.n_features_}")
            selected_features = [f for f, s in zip(features, rfecv.support_) if s]

            if share == 'AAPL' or share == 'MSFT':
                X_refined = X[selected_features]
                desired_features = 10
                rfe_final = RFE(
                    estimator=model_judge,
                    n_features_to_select=desired_features,
                    step=1
                )

                rfe_final.fit(X_refined, y)
                final_aapl_features = [f for f, s in zip(selected_features, rfe_final.support_) if s]

                selected_features = final_aapl_features
            print(f"Winner features for {stat} ({len(selected_features)}):")
            print(selected_features)
            feature_dict[share][stat] = selected_features

    with open("../models_results/feature_dict.json", "w") as f:
        json.dump(feature_dict, f, indent=4)

    return feature_dict



#### MODEL TRAINING

def build_ensemble_model():
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=42))
    ])

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42
        ))
    ])

    xgb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        ))
    ])

    model = VotingClassifier(
        estimators=[
            ("lr", lr_pipeline),
            ("rf", rf_pipeline),
            ("svm", svm_pipeline),
            ("xgb", xgb_pipeline),
        ],
        voting="soft"
    )
    return model


### Sample weights

    def get_sample_weight(df, ticker, horizon=5):
        # create a binary matrix indicating which days are covered by which barrier
        num_rows = len(df)
        concurrency = np.zeros(num_rows)

        for i in range(num_rows - horizon):
            concurrency[i  : i + horizon] += 1

        uniqueness = 1.0 / np.maximum(concurrency, 1)

        weights = pd.Series(index=df.index, dtype=float)
        for i in range(num_rows - horizon):
            weights.iloc[i] = uniqueness[i : i + horizon].mean()

        return weights.fillna(0)


#### Walking Forward

def walk_forward_validation(
    df,
    features,
    target_col="Target",
    date_col="DATE",
    start_year=2010,
    first_train_end_year=2015,
    last_test_year=2023
):
    """
    Train: start_year -> train_end_year
    Test : train_end_year+1
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    fold_rows = []

    for train_end_year in range(first_train_end_year, last_test_year):
        test_year = train_end_year + 1

        train_mask = (df[date_col].dt.year >= start_year) & (df[date_col].dt.year <= train_end_year)
        test_mask = (df[date_col].dt.year == test_year)

        train_df = df[train_mask]
        test_df = df[test_mask]

        # jeżeli jakiś rok nie ma danych to skip
        if len(train_df) < 200 or len(test_df) < 50:
            continue

        X_train = train_df[features]
        y_train = train_df[target_col]

        X_test = test_df[features]
        y_test = test_df[target_col]

        model = build_ensemble_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # ważne dla ROC-AUC

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        fold_rows.append({
            "train_end_year": train_end_year,
            "test_year": test_year,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "roc_auc": auc
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())

    folds_df = pd.DataFrame(fold_rows)

    return folds_df, np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba)


### Block Bootstrap

def block_bootstrap_accuracy(y_true, y_pred, block_size=20, n_bootstrap=1000, random_state=42):
    """
    Bootstrap na wynikach testowych (y_true/y_pred), losowanie blokami.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    if n < block_size:
        raise ValueError("Za mało danych do bootstrapa w tej konfiguracji.")

    acc_samples = []

    for _ in range(n_bootstrap):
        sampled_idx = []

        while len(sampled_idx) < n:
            start = rng.integers(0, n - block_size + 1)
            block = list(range(start, start + block_size))
            sampled_idx.extend(block)

        sampled_idx = sampled_idx[:n]
        y_true_bs = y_true[sampled_idx]
        y_pred_bs = y_pred[sampled_idx]

        acc = accuracy_score(y_true_bs, y_pred_bs)
        acc_samples.append(acc)

    acc_samples = np.array(acc_samples)
    ci_low = np.percentile(acc_samples, 2.5)
    ci_high = np.percentile(acc_samples, 97.5)

    return acc_samples, ci_low, ci_high


### Walk forward + block bootstrap

def run_stage4_for_ticker(df_raw, ticker, selected_features):
    df = get_target(df_raw, ticker)

    selected_features = [f for f in selected_features if f in df.columns]

    folds_df, y_true_all, y_pred_all, y_proba_all = walk_forward_validation(
        df=df,
        features=selected_features,
        target_col="Target",
        date_col="DATE",
        start_year=2010,
        first_train_end_year=2015,
        last_test_year=2023
    )

    print("=" * 60)
    print(f" WALK-FORWARD RESULTS for {ticker}")
    print(folds_df)

    print("\n--- Summary ---")
    print("Mean accuracy:", folds_df["accuracy"].mean())
    print("Std  accuracy:", folds_df["accuracy"].std())
    print("Mean roc_auc :", folds_df["roc_auc"].mean())

    # Block Bootstrap
    acc_samples, ci_low, ci_high = block_bootstrap_accuracy(
        y_true=y_true_all,
        y_pred=y_pred_all,
        block_size=20,
        n_bootstrap=1000
    )

    print("\n" + "=" * 60)
    print(f" BLOCK BOOTSTRAP for {ticker}")
    print(f"95% CI accuracy: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Bootstrap mean accuracy: {acc_samples.mean():.4f}")

    return folds_df, acc_samples





