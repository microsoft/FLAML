import random
import uuid
import warnings

import pandas as pd
import pytest
from sklearn import __version__ as sklearn_version
from sklearn import datasets
from sklearn.model_selection import train_test_split

from flaml import AutoML
from flaml.automl import Featurization

warnings.filterwarnings("ignore")

skip_autofe = pytest.mark.skipif(sklearn_version < "1.3.0", reason="AutoFe requires sklearn>=1.3.0")


def run_autofe(featurization):
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

    train, test = train_test_split(df, test_size=0.2, random_state=123)
    high_cardinality = [random.randint(1000, 9999) for _ in range(len(train))]
    test_cardinality = [random.choice(high_cardinality) for _ in range(len(test))]
    train["col_need_to_drop"] = high_cardinality
    test["col_need_to_drop"] = test_cardinality

    automl = AutoML()
    automl.fit(dataframe=train, label="survived", max_iter=5, featurization=featurization, task="classification")
    automl.predict(test)
    transformer = automl.feature_transformer
    test_transformed = transformer.transform(test)
    autofe = automl.model.autofe
    if autofe is not None:
        autofe.show_transformations()
    return autofe, test_transformed.columns


@skip_autofe
def test_numpy_autofe():
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    automl = AutoML()
    automl.fit(X_train=X_train, y_train=y_train, max_iter=5, featurization="auto", task="classification")
    automl.predict(X_test)
    transformer = automl.feature_transformer
    transformer.transform(X_test)
    autofe = automl.model.autofe
    autofe.show_transformations()


@skip_autofe
def test_autofe():
    run_autofe("auto")


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
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

    train, test = train_test_split(df, test_size=0.2, random_state=123)
    high_cardinality = [str(uuid.uuid4())[:4] for _ in range(len(train))]
    test_cardinality = [random.choice(high_cardinality) for _ in range(len(test))]
    train["col_need_to_drop"] = high_cardinality
    test["col_need_to_drop"] = test_cardinality
    train_X = train.drop(columns=["survived"])
    train_y = train["survived"]
    test_X = test.drop(columns=["survived"])

    automl = AutoML()
    automl.fit(X_train=train_X, y_train=train_y, max_iter=20, featurization="auto")

    autofe = automl.model.autofe
    config = autofe.config
    test_transformed = autofe.transform(test_X)
    reconstruct_autofe = Featurization(config=config, task="classification")

    reconstruct_autofe.fit(train_X, train_y)
    test_re_transformed = reconstruct_autofe.transform(test_X)
    assert set(test_re_transformed.columns) == set(
        test_transformed.columns
    ), "Reconstructed autofe should be the same as the original one"


if __name__ == "__main__":
    test_reconstruct()
