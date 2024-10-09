from typing import Any, Dict, List, Union
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

def evaluate_cv_folds_with_underlying_model(
        X_train_all, y_train_all, kf,model: Any, task: str
) -> pd.DataFrame:
    """Mimic the FLAML CV process to calculate the metrics across each fold.

    :param X_train_all: X training data
    :param y_train_all: y training data
    :param kf: The splitter object to use to generate the folds
    :param model: The estimator to fit to the data during the CV process
    :param task: classification or regression
    :return: An array containing the metrics
    """
    rng = np.random.RandomState(2020)
    all_fold_metrics: List[Dict[str, Union[int, float]]] = []
    for train_index, val_index in kf.split(X_train_all, y_train_all):
        X_train_split, y_train_split = X_train_all, y_train_all
        train_index = rng.permutation(train_index)
        X_train = X_train_split.iloc[train_index]
        X_val = X_train_split.iloc[val_index]
        y_train, y_val = y_train_split[train_index], y_train_split[val_index]
        model_type = type(model)
        if model_type is not CatBoostClassifier and model_type is not CatBoostRegressor:
            model.fit(X_train, y_train)
        else:
            use_best_model = True
            n = max(int(len(y_train) * 0.9), len(y_train) - 1000) if use_best_model else len(y_train)
            X_tr, y_tr = (X_train)[:n], y_train[:n]
            eval_set = (
                Pool(data=X_train[n:], label=y_train[n:], cat_features=[]) if use_best_model else None
            )
            model.fit(X_tr, y_tr, eval_set=eval_set, use_best_model=True)
        y_pred_classes = model.predict(X_val)
        if task == 'classification':
            reproduced_metric = 1 - f1_score(y_val, y_pred_classes)
        else:
            reproduced_metric = 1 - r2_score(y_val, y_pred_classes)
        all_fold_metrics.append(reproduced_metric)
    return all_fold_metrics