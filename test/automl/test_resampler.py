"""Tests for the `resampler=` kwarg on `AutoML.fit` (issue #1200).

Core tests use a local stub resampler (no imbalanced-learn dependency) so they
always run in CI; one end-to-end SMOTE integration test runs only where
imbalanced-learn is installed.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification

from flaml import AutoML


class DuplicateMinorityResampler(BaseEstimator):
    """Deterministic stand-in for an imblearn sampler: duplicates minority rows
    until classes are balanced. Class-level counters record every invocation so
    tests can assert the per-fold hook fired (counters survive sklearn.base.clone,
    which creates fresh instances)."""

    n_calls = 0
    seen_input_sizes = []
    seen_output_sizes = []

    def fit_resample(self, X, y):
        type(self).n_calls += 1
        y_arr = np.asarray(y)
        type(self).seen_input_sizes.append(len(y_arr))
        values, counts = np.unique(y_arr, return_counts=True)
        majority_count = counts.max()
        extra_X, extra_y = [], []
        for v, c in zip(values, counts):
            if c < majority_count:
                idx = np.where(y_arr == v)[0]
                take = np.resize(idx, majority_count - c)
                extra_X.append(X.iloc[take] if hasattr(X, "iloc") else X[take])
                extra_y.append(y_arr[take])
        if extra_X:
            if hasattr(X, "iloc"):
                X_out = pd.concat([X] + extra_X, ignore_index=True)
            else:
                X_out = np.concatenate([X] + extra_X)
            y_out = np.concatenate([y_arr] + extra_y)
        else:
            X_out, y_out = X, y_arr
        type(self).seen_output_sizes.append(len(y_out))
        return X_out, y_out

    @classmethod
    def reset_counters(cls):
        cls.n_calls = 0
        cls.seen_input_sizes = []
        cls.seen_output_sizes = []


def _imbalanced_dataset(seed: int = 0):
    X, y = make_classification(
        n_samples=800,
        n_features=10,
        n_informative=6,
        weights=[0.94, 0.06],
        class_sep=0.7,
        flip_y=0.02,
        random_state=seed,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), pd.Series(y, name="target")


def _fit_settings(**overrides):
    settings = {
        "task": "classification",
        "metric": "f1",
        "estimator_list": ["lgbm"],
        "eval_method": "cv",
        "n_splits": 3,
        "max_iter": 2,
        "time_budget": -1,
        "verbose": 0,
        "n_jobs": 1,
        "retrain_full": False,
    }
    settings.update(overrides)
    return settings


def test_resampler_hook_fires_per_fold_and_balances_train():
    """The per-fold hook must run once per (trial, fold) and grow the training
    partition — asserted directly on the stub's counters, independent of which
    config the search happens to pick."""
    DuplicateMinorityResampler.reset_counters()
    X, y = _imbalanced_dataset(seed=0)

    automl = AutoML()
    automl.fit(X_train=X, y_train=y, resampler=DuplicateMinorityResampler(), seed=42, **_fit_settings())

    n_splits, max_iter = 3, 2
    assert (
        DuplicateMinorityResampler.n_calls == n_splits * max_iter
    ), f"expected {n_splits * max_iter} per-fold resample calls, got {DuplicateMinorityResampler.n_calls}"
    for size_in, size_out in zip(
        DuplicateMinorityResampler.seen_input_sizes, DuplicateMinorityResampler.seen_output_sizes
    ):
        assert size_out > size_in, "resampled training fold did not grow"


def test_resampler_leaves_validation_untouched():
    """Validation folds must keep the raw class distribution. Asserted from
    inside the evaluation loop via a custom metric that inspects y_val."""
    DuplicateMinorityResampler.reset_counters()
    X, y = _imbalanced_dataset(seed=1)
    raw_pos_rate = float(np.mean(y))
    observed_val_rates = []

    def custom_metric(X_val, y_val, estimator, labels, X_train, y_train, *args, **kwargs):
        observed_val_rates.append(float(np.mean(y_val)))
        from sklearn.metrics import f1_score

        y_pred = estimator.predict(X_val)
        loss = 1 - f1_score(y_val, y_pred, zero_division=0)
        return loss, {"val_loss": loss}

    automl = AutoML()
    automl.fit(
        X_train=X,
        y_train=y,
        resampler=DuplicateMinorityResampler(),
        seed=1,
        **_fit_settings(metric=custom_metric),
    )

    assert observed_val_rates, "custom metric was never invoked"
    for rate in observed_val_rates:
        assert rate < 2 * raw_pos_rate, (
            f"validation fold positive rate {rate:.3f} is far above the raw rate "
            f"{raw_pos_rate:.3f} — validation data may have been resampled"
        )


def test_resampler_with_sample_weight_raises():
    X, y = _imbalanced_dataset(seed=2)
    sample_weight = np.where(y == 1, 5.0, 1.0)

    automl = AutoML()
    with pytest.raises(ValueError, match="Cannot combine 'resampler' with 'sample_weight'"):
        automl.fit(
            X_train=X,
            y_train=y,
            resampler=DuplicateMinorityResampler(),
            sample_weight=sample_weight,
            **_fit_settings(),
        )


def test_resampler_with_sample_weight_by_estimator_raises():
    """sample_weight supplied per-estimator via fit_kwargs_by_estimator must be
    rejected just like the top-level kwarg."""
    X, y = _imbalanced_dataset(seed=2)
    sample_weight = np.where(y == 1, 5.0, 1.0)

    automl = AutoML()
    with pytest.raises(ValueError, match="Cannot combine 'resampler' with 'sample_weight'"):
        automl.fit(
            X_train=X,
            y_train=y,
            resampler=DuplicateMinorityResampler(),
            fit_kwargs_by_estimator={"lgbm": {"sample_weight": sample_weight}},
            **_fit_settings(),
        )


def test_resampler_without_fit_resample_raises():
    X, y = _imbalanced_dataset(seed=3)

    class NotAResampler:
        pass

    automl = AutoML()
    with pytest.raises(TypeError, match="fit_resample"):
        automl.fit(X_train=X, y_train=y, resampler=NotAResampler(), **_fit_settings())


def test_resampler_not_cloneable_raises():
    """An object with fit_resample but broken get_params fails at fit() entry
    with a clear message, not mid-fold."""
    X, y = _imbalanced_dataset(seed=3)

    class UncloneableResampler:
        def __init__(self, ratio):
            self.ratio = ratio

        def fit_resample(self, X, y):
            return X, y

        def get_params(self, deep=True):
            return {}  # drops `ratio` — sklearn.clone raises on param mismatch

    automl = AutoML()
    with pytest.raises(TypeError, match="cloneable"):
        automl.fit(X_train=X, y_train=y, resampler=UncloneableResampler(0.5), **_fit_settings())


def test_resampler_none_is_default_and_noop():
    """Explicit `resampler=None` must behave identically to omitting the kwarg."""
    X, y = _imbalanced_dataset(seed=4)

    a = AutoML()
    a.fit(X_train=X, y_train=y, seed=7, **_fit_settings(retrain_full=True))

    b = AutoML()
    b.fit(X_train=X, y_train=y, resampler=None, seed=7, **_fit_settings(retrain_full=True))

    assert a.best_config == b.best_config
    assert a.best_estimator == b.best_estimator
    assert np.array_equal(a.predict(X), b.predict(X))


def test_resampler_does_not_leak_across_fits():
    """A fit without a resampler after a fit with one must not silently reuse
    the earlier resampler (stale task state)."""
    DuplicateMinorityResampler.reset_counters()
    X, y = _imbalanced_dataset(seed=5)

    automl = AutoML()
    automl.fit(X_train=X, y_train=y, resampler=DuplicateMinorityResampler(), seed=7, **_fit_settings())
    calls_after_first = DuplicateMinorityResampler.n_calls
    assert calls_after_first > 0

    automl2 = AutoML()
    automl2.fit(X_train=X, y_train=y, seed=7, **_fit_settings())
    assert (
        DuplicateMinorityResampler.n_calls == calls_after_first
    ), "resampler from a previous fit leaked into a fit that did not request one"


def test_resampler_smote_integration():
    """End-to-end smoke test with a real imblearn SMOTE object; runs only where
    imbalanced-learn is installed."""
    imblearn = pytest.importorskip("imblearn.over_sampling")
    SMOTE = imblearn.SMOTE

    X, y = _imbalanced_dataset(seed=0)
    automl = AutoML()
    automl.fit(
        X_train=X,
        y_train=y,
        resampler=SMOTE(random_state=42, k_neighbors=3),
        seed=42,
        **_fit_settings(retrain_full=True),
    )
    assert automl.model is not None
    assert 0.0 <= automl.best_loss <= 1.0
    pred = automl.predict(X)
    assert len(pred) == len(y)
