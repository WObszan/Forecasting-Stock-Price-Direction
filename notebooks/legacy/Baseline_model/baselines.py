from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_xgboost_model(X_train=None, y_train=None):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            n_estimators=50, 
            max_depth=3, 
            learning_rate=0.1, 
            eval_metric='logloss', 
            random_state=42,
            n_jobs=-1
        ))
    ])

def get_svm_model(X_train=None, y_train=None):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            kernel='rbf', 
            C=1.0, 
            gamma='scale', 
            probability=True, 
            random_state=42
        ))
    ])

def get_rf_model(X_train=None, y_train=None):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=100, 
            max_depth=4, 
            min_samples_leaf=10, 
            random_state=42, 
            n_jobs=-1
        ))
    ])

def get_lr_model(X_train=None, y_train=None):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=10000, 
            random_state=42
        ))
    ])

### Always buy
def get_dummy_model(X_train=None, y_train=None):
    return DummyClassifier(strategy='constant', constant=1)