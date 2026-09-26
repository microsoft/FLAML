"""SEFR: a linear-time, closed-form classifier.

Reference:
    Keshavarz, Saniee Abadeh, Rawassizadeh (2020).
    "SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices".
    https://arxiv.org/abs/2006.04620

SEFR derives one weight per feature plus a bias from class-conditional feature
means. Fitting is a closed form with no iterative optimization: a constant
number of O(n_samples * n_features) passes (feature range, class means, training
scores). A binary SEFR head is ``n_features + 1`` floats; the fitted classifier
also stores the per-feature scaling parameters, one head per class for
multiclass targets, and one probability scale.

It is implemented here in NumPy rather than pulled in as a dependency: the whole
algorithm is a handful of array operations, and FLAML's only required dependency
is NumPy.
"""

from __future__ import annotations

import inspect

import numpy as np

try:
    from scipy.sparse import issparse
except ImportError:

    def issparse(X):
        return False


try:
    from sklearn.base import BaseEstimator as SKLearnBaseEstimator
    from sklearn.base import ClassifierMixin
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils.multiclass import check_classification_targets
    from sklearn.utils.validation import check_is_fitted

    try:
        from sklearn.utils.validation import validate_data as _validate_data  # scikit-learn >= 1.6
    except ImportError:

        def _validate_data(estimator, **kwargs):
            return estimator._validate_data(**kwargs)

except ImportError as e:
    print(f"scikit-learn is required for SEFREstimator. Please install it; error: {e}")

from flaml import tune
from flaml.automl.model import SKLearnEstimator
from flaml.automl.task import Task

_EPS = 1e-7


def _column_weighted_mean(X, weights, total):
    """Weighted column means of ``X``; works for dense arrays and sparse matrices."""
    return np.asarray(X.T.dot(weights)).ravel() / total


def _to_dense_row(values):
    """Collapse the result of a column-wise reduction to a 1-D ndarray."""
    if issparse(values):
        values = values.toarray()
    return np.asarray(values).ravel()


def _weighted_std(values, weights):
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def _min_value(X):
    if issparse(X):
        return X.data.min() if X.nnz else 0.0
    return X.min() if X.size else 0.0


class SEFRClassifier(ClassifierMixin, SKLearnBaseEstimator):
    """Scalable, Efficient and Fast classifieR (SEFR).

    Binary targets are fit with the closed form of the paper (Eqs. 3-9);
    multiclass targets use the one-vs-rest scheme of Sec. 3.4.

    SEFR's weight formula is only meaningful for non-negative features, so every
    input reaches the SEFR heads in that domain or is rejected.

    Parameters
    ----------
    scaling : {"minmax", "none"}, default="minmax"
        "minmax" maps each feature to [0, 1] using the range seen in training,
        and clips values outside that range at prediction time. Sparse input is
        scaled by the column maximum instead, which preserves sparsity; that is
        only a valid [0, 1] map for non-negative data, so sparse input with
        negative entries is rejected at fit time. "none" uses the features as
        given and requires them to be non-negative.
    class_weight : dict, "balanced" or "none", default="none"
        Multiplies each sample's weight by the weight of its class before the
        means of Eq. 3-4 are taken. "balanced" weights classes inversely to
        their total sample weight. None is accepted as an alias of "none".
    threshold : {"sefr", "balanced"}, default="sefr"
        "sefr" is the class-count-weighted score average of Eq. 9. "balanced"
        is the unweighted midpoint of the two class score means.
    threshold_shift : float, default=0.0
        Shifts the bias by this many standard deviations of the training scores.
    calibration : {"sigmoid", "platt"}, default="sigmoid"
        SEFR produces margins, not probabilities. Both options map a margin m to
        sigmoid(a * m) with one scale a shared by all heads, so `predict`,
        `decision_function` and `predict_proba` always agree. "sigmoid" sets a
        to the inverse standard deviation of the training margins (closed form).
        "platt" fits a by a one-parameter logistic regression on the training
        margins; it is iterative, so it gives up the closed form, but it usually
        gives a better log_loss.
    eps : float, default=1e-7
        Stabilizer for the denominator of Eq. 5.
    """

    def __init__(
        self,
        scaling="minmax",
        class_weight="none",
        threshold="sefr",
        threshold_shift=0.0,
        calibration="sigmoid",
        eps=_EPS,
    ):
        self.scaling = scaling
        self.class_weight = class_weight
        self.threshold = threshold
        self.threshold_shift = threshold_shift
        self.calibration = calibration
        self.eps = eps

    def _more_tags(self):  # scikit-learn < 1.6
        return {"requires_positive_X": self.scaling == "none"}

    def __sklearn_tags__(self):  # scikit-learn >= 1.6
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.input_tags.positive_only = self.scaling == "none"
        return tags

    def _check_params(self):
        if not (self.class_weight in (None, "none", "balanced") or isinstance(self.class_weight, dict)):
            raise ValueError(f"class_weight must be a dict, 'balanced' or 'none', got {self.class_weight!r}")
        for name, allowed in (
            ("scaling", ("minmax", "none")),
            ("threshold", ("sefr", "balanced")),
            ("calibration", ("sigmoid", "platt")),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")

    @staticmethod
    def _check_sample_weight(sample_weight, n_samples):
        if sample_weight is None:
            return np.ones(n_samples, dtype=np.float64)
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.ndim == 0:
            sample_weight = np.full(n_samples, float(sample_weight))
        if sample_weight.shape != (n_samples,):
            raise ValueError(f"sample_weight.shape == {sample_weight.shape}, expected {(n_samples,)}!")
        if not np.all(np.isfinite(sample_weight)):
            raise ValueError("sample_weight must be finite")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must be non-negative")
        return sample_weight

    def _check_non_negative(self, X, reason):
        if _min_value(X) < 0:
            raise ValueError(
                f"Negative values in data passed to SEFRClassifier: SEFR requires non-negative features {reason}. "
                "Use dense input with scaling='minmax', or shift the features to be non-negative."
            )

    def _class_totals(self, y_index, sample_weight):
        """Per-class weight totals; a class with none would divide by zero in Eq. 3-4."""
        totals = np.bincount(y_index, weights=sample_weight, minlength=self.classes_.size)
        if np.any(totals <= 0):
            raise ValueError(
                f"Sample weights sum to zero for classes {self.classes_[totals <= 0].tolist()}; "
                "every class needs a positive total weight"
            )
        return totals

    def _fit_scaler(self, X):
        if issparse(X):
            self._check_non_negative(X, "for sparse input")
        if self.scaling == "none":
            self._check_non_negative(X, "when scaling='none'")
            self.offset_, self.spread_ = None, None
            return
        if issparse(X):
            # the input is non-negative, so dividing by the column maximum maps
            # it into [0, 1]; subtracting a column minimum would densify it
            self.offset_ = None
            spread = _to_dense_row(X.max(axis=0))
        else:
            self.offset_ = X.min(axis=0)
            spread = X.max(axis=0) - self.offset_
        spread[spread == 0] = 1.0
        self.spread_ = spread

    def _scale(self, X):
        if self.spread_ is None:
            self._check_non_negative(X, "when scaling='none'")
            return X
        if issparse(X):
            # multiply() keeps the matrix sparse; dividing by a dense row would not
            X = X.multiply(1.0 / self.spread_).tocsr()
            np.clip(X.data, 0.0, 1.0, out=X.data)
            return X
        if self.offset_ is not None:
            X = X - self.offset_
        return np.clip(X / self.spread_, 0.0, 1.0)

    def _fit_head(self, X, is_positive, sample_weight):
        """Return (coef, bias, margins) for one binary problem."""
        negative = ~is_positive
        w_pos, w_neg = sample_weight[is_positive], sample_weight[negative]
        sum_pos, sum_neg = w_pos.sum(), w_neg.sum()

        avg_pos = _column_weighted_mean(X[is_positive], w_pos, sum_pos)
        avg_neg = _column_weighted_mean(X[negative], w_neg, sum_neg)
        coef = (avg_pos - avg_neg) / (avg_pos + avg_neg + self.eps)

        scores = np.asarray(X @ coef).ravel()
        score_pos = np.average(scores[is_positive], weights=w_pos)
        score_neg = np.average(scores[negative], weights=w_neg)
        if self.threshold == "balanced":
            bias = 0.5 * (score_pos + score_neg)
        else:
            bias = (sum_neg * score_pos + sum_pos * score_neg) / (sum_neg + sum_pos)
        if self.threshold_shift:
            bias += self.threshold_shift * (_weighted_std(scores, sample_weight) or 1.0)
        return coef, bias, scores - bias

    def _fit_calibration(self, margins, targets, sample_weight):
        """Return the scale ``a`` of ``sigmoid(a * margin)``, shared by all heads."""
        margins, targets = margins.ravel(), targets.ravel()
        # margins are (n_samples, n_heads) in row-major order
        weights = np.repeat(sample_weight, margins.size // sample_weight.size)
        scale = 1.0 / (_weighted_std(margins, weights) or 1.0)
        if self.calibration == "platt":
            calibrator = LogisticRegression(fit_intercept=False)
            calibrator.fit(margins.reshape(-1, 1), targets, sample_weight=weights)
            slope = float(calibrator.coef_[0, 0])
            # a non-positive slope would invert predict_proba relative to predict
            if np.isfinite(slope) and slope > 0:
                scale = slope
        return scale

    def fit(self, X, y, sample_weight=None):
        self._check_params()
        X, y = _validate_data(self, X=X, y=y, accept_sparse="csr", dtype=np.float64)
        check_classification_targets(y)
        self.classes_, y_index = np.unique(y, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError(
                "SEFR needs samples of at least 2 classes in the data, "
                f"but the data contains only one class: {self.classes_[0]}"
            )

        sample_weight = self._check_sample_weight(sample_weight, X.shape[0])
        class_totals = self._class_totals(y_index, sample_weight)
        if self.class_weight == "balanced":
            sample_weight = sample_weight * (sample_weight.sum() / (self.classes_.size * class_totals))[y_index]
        elif isinstance(self.class_weight, dict):
            factors = np.array([self.class_weight.get(label, 1.0) for label in self.classes_], dtype=np.float64)
            sample_weight = self._check_sample_weight(sample_weight * factors[y_index], X.shape[0])
            self._class_totals(y_index, sample_weight)

        # zero-weight samples take no part in the fit, including the feature range
        self._fit_scaler(X[sample_weight > 0])
        X = self._scale(X)

        heads = [1] if self.classes_.size == 2 else range(self.classes_.size)
        coefs, biases, margins = [], [], []
        for head in heads:
            coef, bias, head_margins = self._fit_head(X, y_index == head, sample_weight)
            coefs.append(coef)
            biases.append(bias)
            margins.append(head_margins)
        self.coef_ = np.vstack(coefs)
        self.intercept_ = np.array(biases)
        targets = np.column_stack([y_index == head for head in heads])
        self.calibration_scale_ = self._fit_calibration(np.column_stack(margins), targets, sample_weight)
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = _validate_data(self, X=X, accept_sparse="csr", dtype=np.float64, reset=False)
        scores = np.asarray(self._scale(X) @ self.coef_.T) - self.intercept_
        return scores.ravel() if self.coef_.shape[0] == 1 else scores

    def predict(self, X):
        scores = self.decision_function(X)
        if self.coef_.shape[0] == 1:
            return self.classes_[(scores > 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-np.clip(scores * self.calibration_scale_, -30, 30)))
        if self.coef_.shape[0] == 1:
            return np.column_stack([1.0 - proba, proba])
        return proba / proba.sum(axis=1, keepdims=True)


def _adaboost_estimator_param():
    """AdaBoostClassifier's ``base_estimator`` was renamed ``estimator`` in scikit-learn 1.2."""
    params = inspect.signature(AdaBoostClassifier.__init__).parameters
    return "estimator" if "estimator" in params else "base_estimator"


class SEFREstimator(SKLearnEstimator):
    """The class for tuning SEFR."""

    @classmethod
    def search_space(cls, **params) -> dict:
        return {
            "class_weight": {
                "domain": tune.choice(["none", "balanced"]),
                "init_value": "none",
            },
            "threshold": {
                "domain": tune.choice(["sefr", "balanced"]),
                "init_value": "sefr",
            },
            "threshold_shift": {
                "domain": tune.uniform(lower=-1.0, upper=1.0),
                "init_value": 0.0,
            },
            "calibration": {
                "domain": tune.choice(["sigmoid", "platt"]),
                "init_value": "sigmoid",
            },
        }

    @classmethod
    def cost_relative2lgbm(cls) -> float:
        return 1.0

    def config2params(self, config: dict) -> dict:
        params = super().config2params(config)
        params.pop("n_jobs", None)
        params.pop("random_state", None)
        return params

    def __init__(self, task: Task = "binary", **config):
        super().__init__(task, **config)
        assert self._task.is_classification(), "SEFR is for classification tasks only"
        self.estimator_class = SEFRClassifier


class SEFRBoostEstimator(SKLearnEstimator):
    """The class for tuning AdaBoost over SEFR base learners.

    SEFR has no hyperparameters of its own to trade accuracy against cost, so on
    its own it gives the search very little to do. Boosting it keeps the
    closed-form base learner while giving FLAML a real, cheap-to-traverse search
    space.
    """

    ITER_HP = "n_estimators"
    DEFAULT_ITER = 50

    @classmethod
    def search_space(cls, data_size, **params) -> dict:
        upper = max(5, min(1024, int(data_size[0])))
        space = {
            "n_estimators": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "init_value": 4,
                "low_cost_init_value": 4,
            },
            "learning_rate": {
                "domain": tune.loguniform(lower=1 / 1024, upper=1.0),
                "init_value": 0.1,
            },
        }
        space["class_weight"] = SEFREstimator.search_space(**params)["class_weight"]
        return space

    @classmethod
    def cost_relative2lgbm(cls) -> float:
        return 5.0

    def config2params(self, config: dict) -> dict:
        params = super().config2params(config)
        params.pop("n_jobs", None)
        base_params = {key: params.pop(key) for key in ("scaling", "class_weight") if key in params}
        params[_adaboost_estimator_param()] = SEFRClassifier(**base_params)
        if "random_state" not in params:
            params["random_state"] = 24092023
        return params

    def __init__(self, task: Task = "binary", **config):
        super().__init__(task, **config)
        assert self._task.is_classification(), "SEFR is for classification tasks only"
        self.estimator_class = AdaBoostClassifier
