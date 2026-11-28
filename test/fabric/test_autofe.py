import json
import warnings

import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn import datasets
from sklearn.model_selection import train_test_split

from flaml import AutoML
from flaml.automl import Featurization

warnings.filterwarnings("ignore")

skip_autofe = pytest.mark.skipif(Version(sklearn_version) < Version("1.3.0"), reason="AutoFe requires sklearn>=1.3.0")


def _classification_frame():
    X, y = datasets.make_classification(n_samples=120, n_features=4, n_informative=3, n_redundant=0, random_state=123)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame["segment"] = pd.Categorical(np.where(frame["a"] > 0, "positive", "negative"))
    frame["col_need_to_drop"] = pd.Categorical([f"row-{index % 90}" for index in range(len(frame))])
    frame["target"] = y
    return frame


def run_autofe(featurization):
    df = _classification_frame()
    train, test = train_test_split(df, test_size=0.2, random_state=123)
    automl = AutoML()
    automl.fit(
        dataframe=train,
        label="target",
        max_iter=3,
        estimator_list=["rf"],
        n_jobs=1,
        mlflow_logging=False,
        featurization=featurization,
        task="classification",
    )
    predictions = automl.predict(test)
    assert predictions.shape == (len(test),)
    assert set(predictions) <= {0, 1}
    transformer = automl.feature_transformer
    test_transformed = transformer.transform(test)
    autofe = automl.model.autofe
    if autofe is not None:
        assert autofe.__sklearn_is_fitted__()
        autofe.show_transformations()
    return autofe, test_transformed.columns


@skip_autofe
def test_numpy_autofe():
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        max_iter=3,
        estimator_list=["rf"],
        n_jobs=1,
        mlflow_logging=False,
        featurization="auto",
        task="regression",
    )
    predictions = automl.predict(X_test)
    assert predictions.shape == y_test.shape
    assert np.isfinite(predictions).all()
    transformer = automl.feature_transformer
    transformer.transform(X_test)
    autofe = automl.model.autofe
    autofe.show_transformations()


@skip_autofe
def test_autofe():
    features, columns = run_autofe("auto")
    assert features is not None
    assert len(columns) > 0


@skip_autofe
def test_autofe_off():
    _, final_cols = run_autofe("off")
    assert "col_need_to_drop" in final_cols, "Should not drop col_need_to_drop"


@skip_autofe
def test_autofe_force():
    _, final_cols = run_autofe("force")
    assert "col_need_to_drop" not in final_cols, "Should drop col_need_to_drop"


@skip_autofe
def test_reconstruct():
    df = _classification_frame()
    train, test = train_test_split(df, test_size=0.2, random_state=123)
    train_X = train.drop(columns=["target"])
    train_y = train["target"]
    test_X = test.drop(columns=["target"])

    automl = AutoML()
    automl.fit(
        X_train=train_X,
        y_train=train_y,
        max_iter=3,
        estimator_list=["rf"],
        n_jobs=1,
        mlflow_logging=False,
        featurization="auto",
    )

    autofe = automl.model.autofe
    config = json.loads(json.dumps(autofe.config))
    test_transformed = automl.feature_transformer.transform(test_X)
    reconstruct_autofe = Featurization(config=config, task="classification")

    reconstruct_autofe.fit(train_X, train_y)
    test_re_transformed = reconstruct_autofe.transform(test_X)
    assert set(test_re_transformed.columns) == set(
        test_transformed.columns
    ), "Reconstructed autofe should be the same as the original one"
    np.testing.assert_allclose(test_re_transformed, test_transformed)


if __name__ == "__main__":
    test_reconstruct()
