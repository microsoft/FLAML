from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import f1_score, log_loss
from xgboost import XGBClassifier

from flaml import AutoML
from flaml.automl import ml
from flaml.automl.model import LGBMEstimator, LRL1Classifier

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
    if eval_method == "holdout":
        assert all(labels is None for labels in received)
    else:
        assert all(np.array_equal(labels, [0, 1, 2]) for labels in received)


def test_sampled_holdout_custom_metric_keeps_label_inference():
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
        assert labels is None
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


def test_documented_callable_log_loss_with_sampled_estimator():
    rng = np.random.RandomState(1)
    y = rng.choice([0, 1], size=3000)
    y[rng.choice(len(y), 40, replace=False)] = 2
    X = rng.randn(len(y), 4)
    X[:, 0] += 2 * y
    y_val = np.arange(300) % 2
    X_val = rng.randn(len(y_val), 4)
    X_val[:, 0] += 2 * y_val
    sampled_classes = []

    def custom_metric(X_val, y_val, estimator, labels, X_train, y_train, weight_val, weight_train, *args):
        sampled_classes.append(estimator.classes_.copy())
        val_loss = log_loss(y_val, estimator.predict_proba(X_val), labels=labels, sample_weight=weight_val)
        train_loss = log_loss(y_train, estimator.predict_proba(X_train), labels=labels, sample_weight=weight_train)
        return 1.5 * val_loss - 0.5 * train_loss, {"val_loss": val_loss, "train_loss": train_loss}

    automl = AutoML()
    automl.fit(
        X,
        y,
        X_val=X_val,
        y_val=y_val,
        task="classification",
        metric=custom_metric,
        estimator_list=["lgbm"],
        max_iter=2,
        seed=1,
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


def test_custom_metric_cannot_mutate_shared_labels():
    X, y = _three_class_data()
    labels = np.array([0, 1, 2])
    received = []

    def mutating_metric(X_val, y_val, estimator, callback_labels, *args):
        received.append(callback_labels.copy())
        callback_labels[0] = 99
        return 0.5, {}

    for _ in range(2):
        ml._eval_estimator({}, object(), X, y, X, y, None, None, mutating_metric, "classification", labels)
    assert all(np.array_equal(value, [0, 1, 2]) for value in received)
    np.testing.assert_array_equal(labels, [0, 1, 2])


@pytest.mark.parametrize("validation_has_rare_class", [False, True])
def test_sampled_log_loss_with_training_metric(monkeypatch, validation_has_rare_class):
    rng = np.random.RandomState(1)
    y = rng.choice([0, 1], size=3000)
    y[rng.choice(len(y), 40, replace=False)] = 2
    X = rng.randn(len(y), 4)
    X[:, 0] += 2 * y
    y_val = np.arange(300) % (3 if validation_has_rare_class else 2)
    X_val = rng.randn(len(y_val), 4)
    X_val[:, 0] += 2 * y_val
    sampled_classes = []
    original = ml._eval_estimator

    def recording(config, estimator, X_train, y_train, X_val, y_val, *args, **kwargs):
        result = original(config, estimator, X_train, y_train, X_val, y_val, *args, **kwargs)
        sampled_classes.append(estimator.classes_)
        for features, target, actual in (
            (X_val, y_val, result[0]),
            (X_train, y_train, result[1]["train_loss"]),
        ):
            probabilities = np.zeros((len(target), 3))
            probabilities[:, estimator.classes_.astype(int)] = estimator.predict_proba(features)
            assert actual == pytest.approx(log_loss(target, probabilities, labels=[0, 1, 2]))
            assert np.isfinite(actual)
        return result

    monkeypatch.setattr(ml, "_eval_estimator", recording)
    automl = AutoML()
    automl.fit(
        X,
        y,
        X_val=X_val,
        y_val=y_val,
        task="classification",
        metric="log_loss",
        log_training_metric=True,
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


def test_compute_estimator_accepts_string_task():
    X, y = _three_class_data()
    result = ml.compute_estimator(
        X[:180],
        y[:180],
        X[180:],
        y[180:],
        None,
        None,
        None,
        None,
        {"C": 1.0},
        "classification",
        "lrl1",
        "holdout",
        "accuracy",
        estimator_class=LRL1Classifier,
    )
    assert np.isfinite(result[1])


@pytest.mark.parametrize("learner", ["lgbm", "xgboost", "xgb_limitdepth"])
@pytest.mark.parametrize("log_training_metric", [False, True])
@pytest.mark.parametrize("num_classes", [2, 3])
def test_single_class_boosted_sample_has_finite_log_loss(monkeypatch, log_training_metric, learner, num_classes):
    rng = np.random.RandomState(81)
    y = np.zeros(3000, dtype=int)
    y[1500], y[2500] = 1, num_classes - 1
    X = rng.randn(len(y), 4)
    y_val = (np.arange(300) % 5 == 0).astype(int) if num_classes == 2 else np.arange(300) % 2
    X_val = rng.randn(len(y_val), 4)
    sampled_classes = []
    original = ml._eval_estimator

    def recording(config, estimator, X_train, y_train, X_val, y_val, *args, **kwargs):
        result = original(config, estimator, X_train, y_train, X_val, y_val, *args, **kwargs)
        sampled_classes.append(estimator.classes_.copy())
        cases = [(X_val, y_val, result[0])]
        if log_training_metric:
            cases.append((X_train, y_train, result[1]["train_loss"]))
        for features, target, actual in cases:
            raw = estimator.predict_proba(features)
            assert raw.shape == (len(target), 2)
            if num_classes == 2:
                expected = raw
            else:
                expected = np.zeros((len(target), num_classes), dtype=raw.dtype)
                expected[:, 0] = 1.0
            assert np.isfinite(actual)
            assert actual == pytest.approx(log_loss(target, expected, labels=np.arange(num_classes)))
        return result

    monkeypatch.setattr(ml, "_eval_estimator", recording)
    automl = AutoML()
    automl.fit(
        X,
        y,
        X_val=X_val,
        y_val=y_val,
        task="classification",
        metric="log_loss",
        log_training_metric=log_training_metric,
        estimator_list=[learner],
        max_iter=2,
        seed=1,
        min_sample_size=50,
        sample=True,
        starting_points={},
        auto_augment=False,
        n_jobs=1,
        verbose=0,
        retrain_full=False,
    )
    assert len(sampled_classes) == 2
    assert all(np.array_equal(classes, [0]) for classes in sampled_classes)
    assert np.isfinite(automl.best_loss)
    assert automl.model is not None


@pytest.mark.parametrize(
    "sole_class, labels",
    [(0, [0, 1, 2]), (1, [2, 1, 0]), (7, [1, 7, 3]), ("b", ["c", "b", "a"])],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_single_class_lightgbm_probability_alignment(sole_class, labels, wrapped):
    X = np.random.RandomState(81).randn(20, 4)
    estimator = LGBMEstimator(task="classification", n_estimators=2, n_jobs=1, verbosity=-1)
    estimator.fit(X, np.full(len(X), sole_class))
    raw = estimator.predict_proba(X)
    assert raw.shape == (len(X), 2)
    np.testing.assert_array_equal(estimator.classes_, [sole_class])
    before = raw.copy()
    actual = ml._align_log_loss_probabilities(raw, estimator if wrapped else estimator.model, labels)
    expected = np.zeros((len(X), len(labels)))
    expected[:, list(np.unique(labels)).index(sole_class)] = 1.0
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize("wrapped", [False, True])
def test_single_class_xgboost_probability_alignment(wrapped):
    X = np.random.RandomState(81).randn(20, 4)
    estimator = XGBClassifier(n_estimators=2, n_jobs=1)
    estimator.fit(X, np.zeros(len(X), dtype=int))
    raw = estimator.predict_proba(X)
    assert raw.shape == (len(X), 2)
    before = raw.copy()
    classifier = SimpleNamespace(model=estimator, classes_=estimator.classes_) if wrapped else estimator
    actual = ml._align_log_loss_probabilities(raw, classifier, [2, 0, 1])
    np.testing.assert_array_equal(actual, np.tile([1.0, 0.0, 0.0], (len(X), 1)))
    np.testing.assert_array_equal(actual.sum(axis=1), np.ones(len(X)))
    np.testing.assert_array_equal(raw, before)


def test_unknown_single_class_estimator_keeps_unexpected_probability_width():
    raw = np.array([[0.2, 0.8]])
    estimator = SimpleNamespace(classes_=np.array([0]))
    assert ml._align_log_loss_probabilities(raw, estimator, [0, 1, 2]) is raw


def test_probability_width_guard_without_lightgbm(monkeypatch):
    monkeypatch.setattr(ml, "LGBMClassifier", None, raising=False)
    raw = np.array([[0.2, 0.8]])
    estimator = SimpleNamespace(classes_=np.array([0]))
    assert ml._align_log_loss_probabilities(raw, estimator, [0, 1, 2]) is raw


@pytest.mark.parametrize(
    "classes, labels, expected",
    [
        ([2, 0], [0, 1, 2], [[0.8, 0.0, 0.2]]),
        (["z", "a"], ["z", "a", "m"], [[0.8, 0.0, 0.2]]),
        ([0, 1], [1, 0], [[0.2, 0.8]]),
    ],
)
def test_log_loss_probability_column_alignment(classes, labels, expected):
    probabilities = np.array([[0.2, 0.8]])
    estimator = SimpleNamespace(classes_=np.array(classes))
    actual = ml._align_log_loss_probabilities(probabilities, estimator, labels)
    np.testing.assert_allclose(actual, expected)


def test_explicit_label_list_overrides_cached_labels():
    X, y = _three_class_data()
    received = []

    class Estimator:
        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

    def custom_metric(X_val, y_val, estimator, labels, *args, **kwargs):
        received.append(labels)
        return 0.5, {}

    override = ["explicit", "labels"]
    ml.compute_estimator(
        X[:180],
        y[:180],
        X[180:],
        y[180:],
        None,
        None,
        None,
        None,
        {},
        "classification",
        "custom",
        "holdout",
        custom_metric,
        estimator_class=Estimator,
        fit_kwargs={"label_list": override},
        training_labels=np.array([0, 1, 2]),
    )
    assert received == [override]


def test_holdout_labels_cached_and_refreshed_per_fit(monkeypatch):
    X, y = _three_class_data()
    calls = []
    original = ml.len_labels

    def counting(values, *args, **kwargs):
        calls.append(len(values))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(ml, "len_labels", counting)
    automl = AutoML()
    for current_y in (y, y % 2):
        seen = []

        def custom_metric(X_val, y_val, estimator, labels, *args, **kwargs):
            seen.append(labels)
            return 0.5, {}

        automl.fit(
            X[:180],
            current_y[:180],
            X_val=X[180:],
            y_val=current_y[180:],
            metric=custom_metric,
            **SETTINGS,
        )
        assert seen
        assert all(labels is None for labels in seen)
        np.testing.assert_array_equal(automl._state.holdout_labels, np.unique(current_y))
    assert calls == []
