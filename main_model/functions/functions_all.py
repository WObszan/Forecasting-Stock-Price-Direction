import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Target TBM ( Triple Barrier Method)
def get_tbm_target(df, ticker, horizon=5, pt_sl=[1.3,1]):
    df = df.copy()
    close = df[f'Close_{ticker}']
    
    log_ret = np.log(close / close.shift(1))
    volatility = log_ret.rolling(window=20).std()
    
    targets = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df) - horizon):
        price_start = close.iloc[i]
        current_vol = volatility.iloc[i] ### dynamic barrier for each day
        
        upper_barrier = price_start * (1 + current_vol * pt_sl[0])
        lower_barrier = price_start * (1 - current_vol * pt_sl[1])
        
        future_prices = close.iloc[i+1 : i+ 1 + horizon]
        
        targets.iloc[i] = 0
        
        for price_future in future_prices:
            if price_future >= upper_barrier:
                targets.iloc[i] = 1 # profit taking hit
                break
            elif price_future <= lower_barrier:
                targets.iloc[i] = -1 # stop loss hit
                break
    df['Target'] = targets
    return df.dropna(subset=['Target'])


# Target Binary
def get_target(input_df, ticker):
    df = input_df.copy()
    df["Target"] = (df[f"Close_{ticker}"].shift(-1) > df[f"Close_{ticker}"]).astype(int)
    df.dropna(inplace=True)
    return df


# Model: Ensemble (Voting Soft)
### With optuna to optimized
def build_ensemble_model_vote(X_train, y_train):
    def objective(trial):
        # for Random Forest
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 200)
        rf_max_depth = trial.suggest_int("rf_max_depth", 3, 10)
        
        # for XGBoost
        xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True)
        xgb_max_depth = trial.suggest_int("xgb_max_depth", 2, 6)
        
        # for SVM
        svm_c = trial.suggest_float("svm_c", 0.1, 10.0, log=True)
        
        lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ("lr", LogisticRegression(max_iter=1000,
                    solver='lbfgs',
                    #multi_class='multinomial',
                    random_state=42))
        ])
        rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators= rf_n_estimators,
            max_depth= rf_max_depth,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=1
        ))
        ])  
        svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ("svc", SVC(
            kernel="rbf",
            C=svm_c,
            gamma="scale",
            class_weight='balanced',
            probability=False,
            random_state=42
        ))
        ])
        
        xgb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=100,
            max_depth= xgb_max_depth,
            learning_rate= xgb_learning_rate,
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            eval_metric="logloss"
        ))
        ])
        
        ensemble = VotingClassifier(
            estimators= [('lr', lr_pipe),('rf', rf_pipe), ('svm', svm_pipe), ('xgb', xgb_pipe)],
            voting='hard'
        )
        
        split = int(len(X_train) * 0.8)

        X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
        y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

        ensemble.fit(X_tr, y_tr)
        y_pred = ensemble.predict(X_val)

        return precision_score(y_val, y_pred, average="weighted")

    
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10, timeout=60)
        
    best = study.best_params    
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ("lr", LogisticRegression(max_iter=1000,
                                  solver='lbfgs',
                                  #multi_class='multinomial',
                                  random_state=42))
    ])

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=best['rf_n_estimators'],
            max_depth=best['rf_max_depth'],
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ("svc", SVC(
            kernel="rbf",
            C=best['svm_c'],
            gamma="scale",
            class_weight='balanced',
            probability=True,
            random_state=42
        ))
    ])

    xgb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=100,
            max_depth=best['xgb_max_depth'],
            learning_rate=best['xgb_learning_rate'],
            tree_method="hist",
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

def get_sample_weights(df, horizon=5):
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


# Walk Forward validation with purging and embargo in validation
def walk_forward_validation_with_purging(
    df,
    features,
    type,
    model_version,
    target_col="Target",
    date_col="DATE",
    start_year=2010,
    first_train_end_year=2015,
    last_test_year=2023,
    horizon = 5,
    embargo_pct = 0.01,
    ticker = None,
    save_path = None
):

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    all_dates = []
    all_returns = []
    all_closes = []

    fold_rows = []

    sample_weights_all = get_sample_weights(df, horizon)
    
    for train_end_year in range(first_train_end_year, last_test_year):
        test_year = train_end_year + 1

        train_mask = (df[date_col].dt.year >= start_year) & (df[date_col].dt.year <= train_end_year)
        test_mask = (df[date_col].dt.year == test_year)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < 200 or len(test_df) < 50:
            continue

        ## Purging
        train_df_purged = train_df.iloc[:-horizon]
        weights_train = sample_weights_all.loc[train_df_purged.index] 
        
        ## Embargo
        embargo_size = int(len(df) * embargo_pct)
        test_df_embargo = test_df.iloc[embargo_size:]
        
        if len(test_df_embargo) < 10: continue
        
        
        X_train = train_df_purged[features]
        y_train = train_df_purged[target_col]

        X_test = test_df_embargo[features]
        y_test = test_df_embargo[target_col]

        model = model_version(X_train,  y_train)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if type == 'binary':
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict_proba(X_test) 

        acc = accuracy_score(y_test, y_pred)
        if type == 'binary':
            prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        else:
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = roc_auc_score(
                y_test,
                y_proba,
                average="weighted",
                multi_class="ovr"
            )

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
        all_dates.extend(test_df_embargo[date_col].values)

        if 'log_return' in test_df_embargo.columns:
            all_returns.extend(test_df_embargo['log_return'].values)
        else:
            all_returns.extend([0] * len(test_df_embargo))

        if ticker and f'Close_{ticker}' in test_df_embargo.columns:
            all_closes.extend(test_df_embargo[f'Close_{ticker}'].values)
        else:
            all_closes.extend([0] * len(test_df_embargo))

    folds_df = pd.DataFrame(fold_rows)

    if save_path is not None:
        detailed_df = pd.DataFrame({
            'Date': all_dates,
            'Actual': all_y_true,
            'Predicted': all_y_pred,
            'Log_Return': all_returns,
            'Close': all_closes
        })
        try:
            proba_arr = np.array(all_y_proba)
            if len(proba_arr.shape) > 1:  # Multiclass
                for i in range(proba_arr.shape[1]):
                    detailed_df[f'Proba_{i}'] = proba_arr[:, i]
            else:  # Binary
                detailed_df['Proba'] = proba_arr
        except:
            detailed_df['Proba'] = [str(x) for x in all_y_proba]

        detailed_df.to_csv(save_path, index=False)

    return folds_df, np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba)

# Block bootstrap
def block_bootstrap_accuracy(y_true, y_pred, block_size=20, n_bootstrap=1000, random_state=42):
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


# Walk_forward + Bootstrap
def run_stage4_for_ticker(df_raw, ticker, selected_features, model_version, target_type, save_detailed_results=False):
    if target_type == "tbm":
        df = get_tbm_target(df_raw, ticker)
    elif target_type == "binary":
        df = get_target(df_raw, ticker)
    result = {}

    selected_features = [f for f in selected_features if f in df.columns]

    save_path = None
    if save_detailed_results:
        save_path = f'../models_results/detailed_res_{ticker}_{target_type}.csv'

    folds_df, y_true_all, y_pred_all, y_proba_all = walk_forward_validation_with_purging(
        df=df,
        features=selected_features,
        type=target_type,
        model_version=model_version,
        target_col="Target",
        date_col="DATE",
        start_year=2010,
        first_train_end_year=2015,
        last_test_year=2023,
        ticker=ticker,
        save_path=save_path,
    )

    print("=" * 60)
    print(f" WALK-FORWARD RESULTS for {ticker}")
    metrics = ["accuracy", "precision", "recall", "roc_auc"]
    summary_mean = folds_df[metrics].mean()

    print("\n--- Summary ---")
    print(summary_mean.to_frame(name="Mean").T)
     
    print("\n--- Detailed Classification Report (Whole Period) ---")   
    if target_type == "binary":
        print(classification_report(
        y_true_all,
        y_pred_all,
        labels=[0, 1],
        target_names=["Down", "Up"],
        zero_division=0
        ))
    else:
        print(classification_report(y_true_all, y_pred_all, zero_division=0))
    
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

    plt.figure(figsize=(8, 6))
    if target_type == "binary":
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Down', 'Up'],
            yticklabels=['Down', 'Up']
        )
    else: 
        cm = confusion_matrix(y_true_all, y_pred_all)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Loss (-1)', 'Neutral (0)', 'Profit (1)'],
                    yticklabels=['Loss (-1)', 'Neutral (0)', 'Profit (1)'])
    plt.title(f'Confusion Matrix for {ticker}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    result[ticker] = {
        'summary': summary_mean.to_frame(name="Mean").T,
        'Bootstrap mean accuracy': acc_samples.mean(),
    }
    #file_path = f"../models_results/main_model_for_{ticker}_result.json"
    #with open(file_path, "w") as f:
        #json.dump(result, f, indent=2)
    

    return folds_df, acc_samples, result

def get_all_features(df):
    return [c for c in df.columns if c not in ["DATE", "index", "Target"]]

def json_converter(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    if isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

