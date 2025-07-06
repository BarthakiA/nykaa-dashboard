from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

def train_churn_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_churn(model, X_test):
    return model.predict_proba(X_test)[:, 1]

def compute_roc(y_test, y_score):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc
