from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import f1_score, log_loss
from xgboost import XGBClassifier

from flaml import AutoML
from flaml.automl import ml
from flaml.automl.model import BaseEstimator, LGBMEstimator, LRL1Classifier, SparkGBTEstimator, SparkNaiveBayesEstimator
from flaml.automl.spark import psDataFrame, psSeries
from flaml.automl.state import AutoMLState
from flaml.automl.task.factory import task_factory

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


@pytest.mark.parametrize("input_mode", ["transform", "skip", "sparse"])
@pytest.mark.parametrize("retrain_full", [False, True])
def test_public_probabilities_with_complete_class_set(monkeypatch, input_mode, retrain_full):
    X, y = _three_class_data()
    if input_mode != "transform":
        y = 10 * (y + 1)
    if input_mode == "sparse":
        X = csr_matrix(X)
    retrained_sizes = []
    original = AutoMLState._train_with_config

    def recording(self, estimator, config, sample_size=None, is_retrain=False):
        if is_retrain:
            retrained_sizes.append(sample_size)
        return original(self, estimator, config, sample_size, is_retrain)

    monkeypatch.setattr(AutoMLState, "_train_with_config", recording)
    automl = AutoML()
    automl.fit(
        X,
        y,
        eval_method="holdout",
        metric="log_loss",
        sample=False,
        skip_transform=input_mode == "skip",
        **{**SETTINGS, "retrain_full": retrain_full},
    )

    assert retrained_sizes == ([X.shape[0]] if retrain_full else [])
    raw = automl.model.predict_proba(automl.preprocess(X[180:]))
    np.testing.assert_array_equal(automl.predict_proba(X[180:]), raw)
    np.testing.assert_array_equal(automl.classes_, np.unique(y))


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
@pytest.mark.parametrize("string_labels", [False, True])
def test_single_class_boosted_sample_has_finite_log_loss(
    monkeypatch, log_training_metric, learner, num_classes, string_labels
):
    rng = np.random.RandomState(81)
    y = np.zeros(3000, dtype=int)
    y[1500], y[2500] = 1, num_classes - 1
    X = rng.randn(len(y), 4)
    y_val = (np.arange(300) % 5 == 0).astype(int) if num_classes == 2 else np.arange(300) % 2
    X_val = rng.randn(len(y_val), 4)
    labels = np.array(["a", "b", "c"][:num_classes]) if string_labels else 10 * (np.arange(num_classes) + 1)
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
            expected = np.zeros((len(target), num_classes), dtype=raw.dtype)
            expected[:, 0] = 1.0
            assert np.isfinite(actual)
            assert actual == pytest.approx(log_loss(target, expected, labels=np.arange(num_classes)))
        return result

    monkeypatch.setattr(ml, "_eval_estimator", recording)
    automl = AutoML()
    automl.fit(
        X,
        labels[y],
        X_val=X_val,
        y_val=labels[y_val],
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
    raw = automl.model.predict_proba(automl.preprocess(X_val))
    probabilities = automl.predict_proba(X_val)
    assert probabilities.shape == (len(y_val), num_classes)
    np.testing.assert_array_equal(automl.classes_, labels)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(probabilities[:, 0], 1)
    np.testing.assert_array_equal(probabilities[:, 1:], 0)
    assert log_loss(labels[y_val], probabilities, labels=automl.classes_) == pytest.approx(automl.best_loss)
    np.testing.assert_array_equal(automl.model.predict_proba(automl.preprocess(X_val)), raw)


def test_single_class_binary_lightgbm_public_class_axis():
    rng = np.random.RandomState(81)
    y = np.full(3000, 20)
    y[[1500, 2500]] = 10
    X = rng.randn(len(y), 4)
    X_val = rng.randn(300, 4)
    y_val = np.where(np.arange(len(X_val)) % 5 == 0, 10, 20)
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
    np.testing.assert_array_equal(automl.model.classes_, [1])
    np.testing.assert_array_equal(automl.classes_, [10, 20])
    predictions = automl.predict(X_val)
    np.testing.assert_array_equal(predictions, 20)
    probabilities = automl.predict_proba(X_val)
    np.testing.assert_array_equal(probabilities, np.tile([0.0, 1.0], (len(X_val), 1)))
    np.testing.assert_array_equal(automl.classes_[probabilities.argmax(axis=1)], predictions)
    assert log_loss(y_val, probabilities, labels=automl.classes_) == pytest.approx(automl.best_loss)


@pytest.mark.parametrize("learner", ["xgboost", "xgb_limitdepth"])
@pytest.mark.parametrize("input_mode", ["transform", "skip", "sparse"])
def test_single_class_xgboost_public_probabilities_preserve_nan(learner, input_mode):
    rng = np.random.RandomState(81)
    y = np.zeros(3000, dtype=int)
    y[1500], y[2500] = 1, 2
    X = rng.randn(len(y), 4)
    X_val = rng.randn(300, 4)
    if input_mode == "sparse":
        X, X_val = csr_matrix(X), csr_matrix(X_val)
    automl = AutoML()
    automl.fit(
        X,
        y,
        X_val=X_val,
        y_val=np.arange(X_val.shape[0]) % 2,
        task="classification",
        metric="log_loss",
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
        skip_transform=input_mode == "skip",
    )
    np.testing.assert_array_equal(automl.model.classes_, [0])
    base_margin = np.zeros(X_val.shape[0])
    base_margin[::2] = np.nan
    raw = automl.model.predict_proba(automl.preprocess(X_val), base_margin=base_margin)
    assert np.isnan(raw[::2]).all()
    assert np.isfinite(raw[1::2]).all()
    probabilities = automl.predict_proba(X_val, base_margin=base_margin)
    assert probabilities.shape == (X_val.shape[0], 3)
    assert np.isnan(probabilities[::2, 0]).all()
    np.testing.assert_array_equal(probabilities[:, 1:], 0)
    np.testing.assert_array_equal(probabilities[1::2, 0], 1)


@pytest.mark.parametrize("input_mode", ["skip", "sparse"])
@pytest.mark.parametrize(
    "learner, labels, sole_class",
    [
        ("lgbm", [0, 1], 0),
        ("lgbm", [0, 1], 1),
        ("lgbm", [10, 20, 30], 20),
        ("lgbm", ["a", "b", "c"], "b"),
        ("xgboost", [0, 1, 2], 0),
        ("xgb_limitdepth", [0, 1, 2], 0),
    ],
)
def test_native_single_class_sample_public_probabilities(monkeypatch, input_mode, learner, labels, sole_class):
    rng = np.random.RandomState(81)
    y = np.full(3000, sole_class)
    other_classes = [label for label in labels if label != sole_class]
    y[1500], y[2500] = other_classes[0], other_classes[-1]
    X = rng.randn(len(y), 4)
    X_val = rng.randn(300, 4)
    y_val = np.resize(labels, len(X_val))
    if input_mode == "sparse":
        X, X_val = csr_matrix(X), csr_matrix(X_val)
    original = ml._eval_estimator
    sampled_classes = []

    def recording(config, estimator, X_train, y_train, X_val, y_val, *args, **kwargs):
        result = original(config, estimator, X_train, y_train, X_val, y_val, *args, **kwargs)
        sampled_classes.append(estimator.classes_.copy())
        for target, loss in [(y_val, result[0]), (y_train, result[1]["train_loss"])]:
            expected = np.zeros((len(target), len(labels)), dtype=estimator.predict_proba(X_val).dtype)
            expected[:, labels.index(sole_class)] = 1
            assert loss == pytest.approx(log_loss(target, expected, labels=labels))
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
        skip_transform=input_mode == "skip",
    )
    assert not automl.label_transformer
    assert len(sampled_classes) == 2
    assert all(np.array_equal(classes, [sole_class]) for classes in sampled_classes)
    np.testing.assert_array_equal(automl.classes_, labels)
    raw = automl.model.predict_proba(automl.preprocess(X_val))
    assert raw.shape == (X_val.shape[0], 2)
    probabilities = automl.predict_proba(X_val)
    expected = np.zeros((X_val.shape[0], len(labels)))
    expected[:, labels.index(sole_class)] = 1
    np.testing.assert_array_equal(probabilities, expected)
    np.testing.assert_array_equal(automl.classes_[probabilities.argmax(axis=1)], automl.predict(X_val))
    assert log_loss(y_val, probabilities, labels=automl.classes_) == pytest.approx(automl.best_loss)
    np.testing.assert_array_equal(automl.model.predict_proba(automl.preprocess(X_val)), raw)


@pytest.mark.parametrize("input_mode", ["skip", "sparse"])
@pytest.mark.parametrize("missing_class", [10, 20, 30])
def test_native_sampled_lightgbm_probability_columns(input_mode, missing_class):
    rng = np.random.RandomState(1)
    labels = np.array([10, 20, 30])
    present_classes = labels[labels != missing_class]
    y = rng.choice(present_classes, size=3000)
    y[rng.choice(len(y), 40, replace=False)] = missing_class
    X = rng.randn(len(y), 4)
    X_val = rng.randn(300, 4)
    y_val = np.resize(labels, len(X_val))
    if input_mode == "sparse":
        X, X_val = csr_matrix(X), csr_matrix(X_val)
    automl = AutoML()
    automl.fit(
        X,
        y,
        X_val=X_val,
        y_val=y_val,
        task="classification",
        metric="log_loss",
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
        skip_transform=input_mode == "skip",
    )
    np.testing.assert_array_equal(automl.model.classes_, present_classes)
    np.testing.assert_array_equal(automl.classes_, labels)
    raw = automl.model.predict_proba(automl.preprocess(X_val))
    probabilities = automl.predict_proba(X_val)
    np.testing.assert_array_equal(probabilities[:, labels != missing_class], raw)
    np.testing.assert_array_equal(probabilities[:, labels == missing_class], 0)
    assert log_loss(y_val, probabilities, labels=automl.classes_) == pytest.approx(automl.best_loss)


@pytest.mark.parametrize("input_mode", ["skip", "sparse"])
@pytest.mark.parametrize("eval_method", ["holdout", "cv"])
def test_native_class_axis_does_not_change_callable_metric_labels(input_mode, eval_method):
    X, y = _three_class_data()
    y = 10 * (y + 1)
    if input_mode == "sparse":
        X = csr_matrix(X)
    received = []

    def custom_metric(X_val, y_val, estimator, labels, *args, **kwargs):
        received.append(labels)
        return 0.5, {}

    automl = AutoML()
    automl.fit(
        X,
        y,
        metric=custom_metric,
        eval_method=eval_method,
        n_splits=3,
        skip_transform=input_mode == "skip",
        **SETTINGS,
    )
    assert received
    if eval_method == "holdout":
        assert all(labels is None for labels in received)
    else:
        assert all(np.array_equal(labels, [10, 20, 30]) for labels in received)
    np.testing.assert_array_equal(automl.classes_, [10, 20, 30])
    np.testing.assert_array_equal(automl.predict_proba(X), automl.model.predict_proba(automl.preprocess(X)))


@pytest.mark.parametrize("input_mode", ["skip", "sparse"])
@pytest.mark.parametrize("eval_method", ["holdout", "cv"])
@pytest.mark.parametrize("string_labels", [False, True])
def test_native_class_axis_refreshed_when_refitting(monkeypatch, input_mode, eval_method, string_labels):
    X, y = _three_class_data()
    if input_mode == "sparse":
        X = csr_matrix(X)
    from flaml.automl import automl as automl_module

    calls = []
    original = automl_module.len_labels

    def counting(values, *args, **kwargs):
        calls.append(len(values))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(automl_module, "len_labels", counting)
    automl = AutoML()
    for current_y in (10 * (y + 1), (y % 2) + 5):
        if string_labels:
            current_y = current_y.astype(str)
        calls.clear()
        automl.fit(
            X,
            current_y,
            metric="log_loss",
            eval_method=eval_method,
            n_splits=3,
            skip_transform=input_mode == "skip",
            **SETTINGS,
        )
        assert calls == [len(y)]
        np.testing.assert_array_equal(automl.classes_, np.unique(current_y))
        np.testing.assert_array_equal(automl.predict_proba(X), automl.model.predict_proba(automl.preprocess(X)))
    automl.fit(
        X,
        y.astype(float),
        task="regression",
        estimator_list=["lgbm"],
        max_iter=1,
        n_jobs=1,
        verbose=0,
        skip_transform=input_mode == "skip",
    )
    assert automl._classes is None
    # Returning to transformed classification must not reuse native labels.
    X, y = _three_class_data()
    automl.fit(X, y, task="classification", skip_transform=False, **{k: v for k, v in SETTINGS.items() if k != "task"})
    assert automl._classes is None
    np.testing.assert_array_equal(automl.classes_, [0, 1, 2])


@pytest.mark.parametrize("input_mode", ["skip", "sparse"])
@pytest.mark.parametrize("eval_method", ["holdout", "cv"])
def test_rejected_native_refit_preserves_installed_class_axis(input_mode, eval_method):
    X, y = _three_class_data(n=120)
    if input_mode == "sparse":
        X = csr_matrix(X)
    settings = {
        **SETTINGS,
        "estimator_list": ["lgbm"],
        "max_iter": 2,
        "eval_method": eval_method,
        "n_splits": 3,
        "skip_transform": input_mode == "skip",
    }
    automl = AutoML()
    automl.fit(X, y % 2, metric="log_loss", **settings)
    installed = automl.model
    classes = automl.classes_.copy()
    probabilities = automl.predict_proba(X)
    assert probabilities.shape == (120, 2)

    with pytest.raises(ValueError, match="only supports binary classification"):
        automl.fit(X, 10 * (y + 1), metric="roc_auc", **settings)

    assert automl.model is installed
    np.testing.assert_array_equal(automl.predict_proba(X), probabilities)
    np.testing.assert_array_equal(automl.classes_, classes)
    assert automl.predict_proba(X).shape[1] == len(automl.classes_)

    automl.fit(X, 10 * (y + 1), metric="log_loss", **settings)
    assert automl.model is not installed
    np.testing.assert_array_equal(automl.classes_, [10, 20, 30])
    assert automl.predict_proba(X).shape == (120, 3)


@pytest.mark.parametrize("input_mode", ["skip", "sparse"])
def test_rejected_task_change_preserves_sampled_probability_axis(input_mode):
    rng = np.random.RandomState(131)
    X = rng.normal(size=(3000, 4))
    y = np.arange(len(X)) % 2
    y[-1] = 2
    X_val = rng.normal(size=(80, 4))
    if input_mode == "sparse":
        X, X_val = csr_matrix(X), csr_matrix(X_val)
    settings = dict(
        estimator_list=["lgbm"],
        n_jobs=1,
        verbose=0,
        max_iter=2,
        auto_augment=False,
        skip_transform=input_mode == "skip",
        retrain_full=False,
    )
    automl = AutoML()
    automl.fit(
        X,
        y,
        X_val=X_val,
        y_val=np.arange(80) % 2,
        task="classification",
        metric="log_loss",
        sample=True,
        min_sample_size=50,
        starting_points={},
        seed=1,
        **settings,
    )
    installed = automl.model
    assert len(installed.classes_) == 2
    classes = automl.classes_.copy()
    probabilities = automl.predict_proba(X_val)
    assert probabilities.shape == (80, 3)

    with pytest.raises(ValueError, match="metric.*parameter"):
        automl.fit(X, np.arange(3000, dtype=float), task="regression", metric=42, **settings)

    assert automl.model is installed
    np.testing.assert_array_equal(automl.classes_, classes)
    np.testing.assert_array_equal(automl.predict_proba(X_val), probabilities)


@pytest.mark.parametrize("estimator_class", [SparkNaiveBayesEstimator, SparkGBTEstimator])
@pytest.mark.parametrize("return_all", [False, True])
def test_spark_probability_facade_bypasses_missing_class_metadata(monkeypatch, estimator_class, return_all):
    # Controlled public facade: no Spark training or native Spark runtime is required.
    # Use repository Spark types without constructing a live Spark session.
    output = object.__new__(psDataFrame if return_all else psSeries)
    estimator = estimator_class.__new__(estimator_class)
    BaseEstimator.__init__(estimator, task="binary")
    estimator._model = SimpleNamespace()  # Like these Spark models, has no classes_.
    with pytest.raises(AttributeError):
        _ = estimator.classes_

    def predict_proba(X, **kwargs):
        assert kwargs == {"return_all": return_all}
        return output

    monkeypatch.setattr(estimator, "predict_proba", predict_proba)
    automl = AutoML()
    automl._state.task = task_factory("classification")
    automl._transformer = automl._label_transformer = None
    automl.preserve_checkpoint = True
    automl._trained_estimator = estimator
    assert automl.predict_proba(np.zeros((2, 3)), return_all=return_all) is output


@pytest.mark.parametrize("input_mode", ["skip", "sparse"])
@pytest.mark.parametrize("eval_method", ["holdout", "cv"])
def test_native_class_axis_retrain_from_log(tmp_path, input_mode, eval_method):
    X, y = _three_class_data()
    if input_mode == "sparse":
        X = csr_matrix(X)
    y = np.array(["a", "b", "c"])[y]
    log_file = str(tmp_path / "native.log")
    automl = AutoML()
    settings = {**SETTINGS, "estimator_list": ["lgbm"], "max_iter": 2, "skip_transform": input_mode == "skip"}
    automl.fit(X, y, metric="log_loss", eval_method=eval_method, n_splits=3, log_file_name=log_file, **settings)

    # Replaying on an existing AutoML must replace its old native class axis.
    automl.fit(X, np.arange(len(y)) % 2, metric="log_loss", **settings)
    automl.retrain_from_log(
        log_file,
        X_train=X,
        y_train=y,
        task="classification",
        eval_method=eval_method,
        n_splits=3,
        train_full=True,
        record_id=0,
        n_jobs=1,
        skip_transform=input_mode == "skip",
    )
    np.testing.assert_array_equal(automl.classes_, ["a", "b", "c"])
    probabilities = automl.predict_proba(X)
    assert probabilities.shape == (len(y), 3)
    np.testing.assert_array_equal(probabilities, automl.model.predict_proba(automl.preprocess(X)))


@pytest.mark.parametrize("output", ["raw_score", "pred_leaf", "pred_contrib", None])
@pytest.mark.parametrize("input_mode", ["transform", "skip", "sparse"])
def test_sampled_lightgbm_prediction_options(output, input_mode):
    rng = np.random.RandomState(81)
    y = np.zeros(3000, dtype=int)
    y[1500], y[2500] = 1, 2
    # One feature makes contribution scores have the same width as binary probabilities.
    X = rng.randn(len(y), 1)
    X_val = rng.randn(300, 1)
    if input_mode == "sparse":
        X, X_val = csr_matrix(X), csr_matrix(X_val)
    automl = AutoML()
    automl.fit(
        X,
        y,
        X_val=X_val,
        y_val=np.arange(X_val.shape[0]) % 2,
        task="classification",
        metric="log_loss",
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
        skip_transform=input_mode == "skip",
    )
    np.testing.assert_array_equal(automl.model.classes_, [0])
    kwargs = {"num_iteration": 2}
    if output is not None:
        kwargs[output] = True
    raw = automl.model.predict_proba(automl.preprocess(X_val), **kwargs)
    actual = automl.predict_proba(X_val, **kwargs)
    if output is None:
        assert actual.shape == (X_val.shape[0], len(automl.classes_))
        np.testing.assert_array_equal(actual[:, 0], 1)
        np.testing.assert_array_equal(actual[:, 1:], 0)
    elif issparse(raw):
        assert issparse(actual)
        assert actual.shape == raw.shape
        assert (actual != raw).nnz == 0
    else:
        np.testing.assert_array_equal(actual, raw)


@pytest.mark.parametrize("missing_class", [0, 1, 2])
@pytest.mark.parametrize("learner", ["lgbm", "rf"])
def test_public_probabilities_from_sampled_classifier(missing_class, learner):
    rng = np.random.RandomState(1)
    present_classes = np.delete(np.arange(3), missing_class)
    y = rng.choice(present_classes, size=3000)
    y[rng.choice(len(y), 40, replace=False)] = missing_class
    X = rng.randn(len(y), 4)
    X[:, 0] += 2 * y
    y_val = np.arange(300) % 3
    X_val = rng.randn(len(y_val), 4)
    X_val[:, 0] += 2 * y_val
    labels = np.array(["alpha", "beta", "gamma"])

    automl = AutoML()
    automl.fit(
        X,
        labels[y],
        X_val=X_val,
        y_val=labels[y_val],
        task="classification",
        metric="log_loss",
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

    np.testing.assert_array_equal(automl.model.classes_, present_classes)
    np.testing.assert_array_equal(automl.classes_, labels)
    features = automl.preprocess(X_val)
    raw = automl.model.predict_proba(features)
    probabilities = automl.predict_proba(X_val)
    assert probabilities.shape == (len(y_val), 3)
    np.testing.assert_array_equal(probabilities[:, present_classes], raw)
    np.testing.assert_array_equal(probabilities[:, missing_class], 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert log_loss(labels[y_val], probabilities, labels=automl.classes_) == pytest.approx(automl.best_loss)
    np.testing.assert_array_equal(automl.model.predict_proba(features), raw)


@pytest.mark.parametrize(
    "sole_class, labels",
    [(0, [0, 1]), (1, [0, 1]), (0, [0, 1, 2]), (1, [2, 1, 0]), (7, [1, 7, 3]), ("b", ["c", "b", "a"])],
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


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("labels", [[0, 1], [0, 1, 2]])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_class_probability_alignment_preserves_nonfinite_rows(invalid, labels, dtype):
    estimator = SimpleNamespace(model=ml.LGBMClassifier(), classes_=np.array([1]))
    raw = np.array([[0.8, 0.2], [0.5, invalid], [invalid, 0.5]], dtype=dtype)
    before = raw.copy()
    actual = ml._align_log_loss_probabilities(raw, estimator, labels)
    assert actual.dtype == raw.dtype
    assert actual.shape == (3, len(labels))
    assert actual[0, 1] == 1
    assert np.isnan(actual[1:, 1]).all()
    np.testing.assert_array_equal(actual[:, np.array(labels) != 1], 0)
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
