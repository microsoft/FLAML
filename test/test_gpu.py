import pickle
import shutil
import sys

import pytest


def test_xgboost():
    import numpy as np
    import scipy.sparse
    from sklearn.datasets import make_moons
    from xgboost.core import XGBoostError

    from flaml import AutoML

    try:
        X_train = scipy.sparse.eye(900000)
        y_train = np.random.randint(2, size=900000)
        automl = AutoML()
        automl.fit(
            X_train,
            y_train,
            estimator_list=["xgb_limitdepth", "xgboost"],
            time_budget=5,
            gpu_per_trial=1,
        )

        train, label = make_moons(n_samples=300000, shuffle=True, noise=0.3, random_state=None)
        automl = AutoML()
        automl.fit(
            train,
            label,
            estimator_list=["xgb_limitdepth", "xgboost"],
            time_budget=5,
            gpu_per_trial=1,
        )
        automl.fit(
            train,
            label,
            estimator_list=["xgb_limitdepth", "xgboost"],
            time_budget=5,
        )
    except XGBoostError:
        # No visible GPU is found for XGBoost.
        return


if __name__ == "__main__":
    test_xgboost()
