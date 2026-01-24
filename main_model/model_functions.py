
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

def model_train(tickers, statistics, feature_dict):
    result_dict = {}

    for i, share in enumerate(tickers):
        df = get_target(data_dict[share], share)
        features = [col for col in df.columns if col not in ['Target', 'index', 'DATE']]

        X = df[features]
        y = df['Target']
        result_dict[share] = {}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        for stat in statistics:
            selected_cols = feature_dict[share][stat]

            rf_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('rf',
                 RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10, random_state=42, n_jobs=-1))
            ])

            svm_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)),
            ])

            xgb_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('xgb',
                 XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, eval_metric='logloss', random_state=42,
                               n_jobs=-1)),
            ])

            main_model = VotingClassifier(
                estimators=[('xgb', xgb_pipeline), ('svc', svm_pipeline), ('rf', rf_pipeline), ],
                voting='soft')

            main_model.fit(X_train[selected_cols], y_train)

            y_pred = main_model.predict(X_test[selected_cols])
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            print('*' * 40)
            print(f"Results for {share} for best stat: {stat}:")
            print("Accuracy:", acc)
            print("Precision:", precision)
            print("Recall:", recall)
            print("ROC AUC score:", roc_auc)
            result_dict[share][stat] = [acc, precision, recall, roc_auc]
    return result_dict




