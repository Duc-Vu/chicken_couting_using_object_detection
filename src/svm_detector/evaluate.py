from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
)
import numpy as np


def evaluate_classifier(model, X, y):
    y_pred = model.predict(X)
    scores = model.decision_function(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, scores),
        "mse": mse,
        "rmse": rmse,
    }