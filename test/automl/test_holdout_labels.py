import numpy as np
import pytest
from sklearn.metrics import f1_score, log_loss

from flaml import AutoML

SETTINGS = {
    "task": "classification",
    "estimator_list": ["lrl1"],
    "max_iter": 3,
    "n_jobs": 1,
    "verbose": 0,
    "retrain_full": False,
}


def _three_class_data(n=240, seed=0):
    rng = np.random.RandomState(seed)
    y = np.arange(n) % 3
    X = rng.randn(n, 4)
    X[:, 0] += 2 * y
    return X, y


def test_log_loss_with_validation_set_missing_a_class():
    X, y = _three_class_data()
    X_train, y_train = X[:180], y[:180]
    keep = y[180:] != 2
    X_val, y_val = X[180:][keep], y[180:][keep]

    automl = AutoML()
    automl.fit(X_train, y_train, X_val=X_val, y_val=y_val, metric="log_loss", **SETTINGS)

    assert automl.best_estimator == "lrl1"
    expected = log_loss(y_val, automl.predict_proba(X_val), labels=automl.classes_)
    assert automl.best_loss == pytest.approx(expected)


def test_time_split_with_class_absent_from_latest_rows():
    X, y = _three_class_data(n=300)
    y[-60:] = np.where(y[-60:] == 2, 0, y[-60:])

    automl = AutoML()
    automl.fit(X, y, eval_method="holdout", split_type="time", split_ratio=0.2, **SETTINGS)

    assert automl.best_estimator == "lrl1"
    assert np.isfinite(automl.best_loss)


@pytest.mark.parametrize("eval_method", ["holdout", "cv"])
def test_custom_metric_receives_training_labels(eval_method):
    X, y = _three_class_data()
    received = []

    def custom_metric(X_val, y_val, estimator, labels, *args, **kwargs):
        received.append(labels)
        return 0.5, {}

    automl = AutoML()
    automl.fit(X, y, metric=custom_metric, eval_method=eval_method, n_splits=3, **SETTINGS)

    assert received
    assert all(np.array_equal(labels, [0, 1, 2]) for labels in received)


def test_sampled_holdout_uses_full_training_labels():
    rng = np.random.RandomState(1)
    y = rng.choice([0, 1], size=3000)
    y[rng.choice(len(y), 40, replace=False)] = 2
    X = rng.randn(len(y), 4)
    X[:, 0] += 2 * y
    y_val = np.arange(300) % 3
    X_val = rng.randn(len(y_val), 4)
    X_val[:, 0] += 2 * y_val
    sampled_classes = []

    def macro_f1(X_val, y_val, estimator, labels, *args, **kwargs):
        sampled_classes.append(estimator.classes_)
        assert np.array_equal(labels, [0, 1, 2])
        predictions = estimator.predict(X_val)
        score = f1_score(y_val, predictions, labels=labels, average="macro")
        expected = f1_score(y_val, predictions, labels=[0, 1, 2], average="macro")
        assert score == pytest.approx(expected)
        return 1 - score, {}

    automl = AutoML()
    automl.fit(
        X,
        y,
        X_val=X_val,
        y_val=y_val,
        task="classification",
        metric=macro_f1,
        estimator_list=["lgbm"],
        max_iter=6,
        min_sample_size=50,
        sample=True,
        starting_points={},
        auto_augment=False,
        n_jobs=1,
        verbose=0,
        retrain_full=False,
    )

    assert any(len(classes) == 2 for classes in sampled_classes)
    assert np.isfinite(automl.best_loss)
