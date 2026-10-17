import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from flaml import AutoML
from flaml.automl.training_log import training_log_writer
from flaml.fabric.autofe import Featurization


@pytest.mark.parametrize("with_featurization", [False, True])
def test_logged_estimator_preserves_unfitted_featurization(tmp_path, with_featurization):
    config = {"n_estimators": 4, "max_features": 1.0}
    if with_featurization:
        config["fe.numerical"] = "scaler_standard"
    path = str(tmp_path / "training.log")
    with training_log_writer(path) as writer:
        writer.append(0, 0.1, 0.1, 0.1, 0.1, config, "rf", 40)

    estimator = AutoML().get_estimator_from_log(path, 0, "regression")
    if with_featurization:
        assert isinstance(estimator, Pipeline)
        featurization = estimator.named_steps["autofe"]
        assert isinstance(featurization, Featurization)
        assert featurization.params == {"fe.numerical": "scaler_standard"}
        assert not featurization.__sklearn_is_fitted__()
    else:
        assert isinstance(estimator, RandomForestRegressor)

    X = pd.DataFrame(np.random.RandomState(0).normal(size=(40, 3)) + 100, columns=["a", "b", "c"])
    y = X["a"] - X["b"]
    estimator.fit(X, y)
    assert len(estimator.predict(X)) == len(X)
    if with_featurization:
        assert featurization.__sklearn_is_fitted__()
        np.testing.assert_allclose(featurization.transform(X).mean(), 0, atol=1e-12)
