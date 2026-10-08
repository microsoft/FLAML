"""SEFR: a linear-time, closed-form classifier.

Reference:
    Keshavarz, Saniee Abadeh, Rawassizadeh (2020).
    "SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices".
    https://arxiv.org/abs/2006.04620

SEFR derives one weight per feature plus a bias from class-conditional feature
means. Fitting is a closed form with no iterative optimization, and takes three
passes over the data whatever the number of classes: the feature range, the
per-class feature sums (one sparse one-hot matrix product), and the spread of
the training scores (in row chunks). Every one-vs-rest head and its Eq. 9 bias
follow from the per-class sums, so beyond the scaled copy of the input the fit
holds O(n_classes * n_features) memory, not O(n_samples * n_classes).
The optional "platt" calibration adds an iterative fit of n_classes + 1
parameters on at most ``_CALIBRATION_MAX_ENTRIES`` training margins.

A binary SEFR head is ``n_features + 1`` floats; the fitted classifier also
stores the per-feature scaling parameters, one head per class for multiclass
targets, and the calibration (one scale, plus one bias per class).

It is implemented here in NumPy rather than pulled in as a dependency: the whole
algorithm is a handful of array operations, and FLAML's only required dependency
is NumPy.
"""

from __future__ import annotations

import inspect
import numbers

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
# Platt calibration fits n_classes + 1 parameters on the training margins. Past
# this many margin entries it uses a deterministic row subsample, which bounds its
# memory (two float64 arrays of this size) and barely changes the fit.
_CALIBRATION_MAX_ENTRIES = 10_000_000
# Strength of the pull of the Platt slope (relative to the closed form) towards the
# closed form, against the weighted mean log loss; keeps separable data finite.
_CALIBRATION_PENALTY = 1e-3
# rows x heads per chunk when scoring the training data
_CHUNK_ENTRIES = 1_000_000


def _class_sums(X, y_index, sample_weight, n_classes):
    """Weighted per-class feature sums, shape (n_classes, n_features), in one pass over ``X``."""
    from scipy.sparse import csr_matrix

    onehot = csr_matrix((sample_weight, (y_index, np.arange(X.shape[0]))), shape=(n_classes, X.shape[0]))
    sums = onehot @ X
    return sums.toarray() if issparse(sums) else np.asarray(sums)


def _to_dense_row(values):
    """Collapse the result of a column-wise reduction to a 1-D ndarray."""
    if issparse(values):
        values = values.toarray()
    return np.asarray(values).ravel()


def _fit_multinomial(margins, y_index, sample_weight, scale, bias):
    """Fit ``softmax(a * margins + b)`` to the labels by weighted maximum likelihood.

    Optimizes ``(a / scale, b)``, starting from the closed form ``(1, bias)``;
    ``a`` stays positive. The objective is the weighted mean log loss plus
    ``0.5 * _CALIBRATION_PENALTY * (a / scale - 1)^2``. Only the slope can diverge
    (on separable data); with it bounded the biases have a finite optimum, and
    leaving them unpenalized keeps rare classes' biases free. The objective does
    not change when the weights are rescaled (AdaBoost normalizes them to sum to
    1). Each step is
    O(n_samples * n_classes), with ``margins`` and one more array of its size in
    memory.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    rows = np.arange(margins.shape[0])
    total = sample_weight.sum()
    start = np.concatenate([[1.0], bias])

    def loss_and_grad(theta):
        logits = margins * (theta[0] * scale)
        logits += theta[1:]
        norm = logsumexp(logits, axis=1)
        stretch = theta[0] - 1.0
        loss = np.dot(sample_weight, norm - logits[rows, y_index]) / total
        loss += 0.5 * _CALIBRATION_PENALTY * stretch * stretch
        # reuse the buffer for the weighted residuals w * (softmax - onehot)
        residual = logits
        residual -= norm[:, None]
        np.exp(residual, out=residual)
        residual[rows, y_index] -= 1.0
        residual *= sample_weight[:, None]
        grad = np.empty_like(theta)
        grad[0] = scale * np.einsum("ij,ij->", residual, margins)
        grad[1:] = residual.sum(axis=0)
        grad /= total
        grad[0] += _CALIBRATION_PENALTY * stretch
        return loss, grad

    bounds = [(1e-8, None)] + [(None, None)] * bias.size
    theta = minimize(loss_and_grad, start, jac=True, method="L-BFGS-B", bounds=bounds).x
    return float(theta[0] * scale), theta[1:]


def _fit_binary(margins, is_positive, sample_weight, scale, bias):
    """Fit ``sigmoid(a * margins + b)``: the two-class case of ``_fit_multinomial``, same objective."""
    from scipy.optimize import minimize
    from scipy.special import expit

    target = is_positive.astype(np.float64)
    total = sample_weight.sum()

    def loss_and_grad(theta):
        logits = margins * (theta[0] * scale) + theta[1]
        stretch = theta[0] - 1.0
        loss = np.dot(sample_weight, np.logaddexp(0.0, logits) - target * logits) / total
        loss += 0.5 * _CALIBRATION_PENALTY * stretch * stretch
        residual = sample_weight * (expit(logits) - target)
        grad = np.array([scale * np.dot(residual, margins), residual.sum()]) / total
        grad[0] += _CALIBRATION_PENALTY * stretch
        return loss, grad

    theta = minimize(
        loss_and_grad, np.array([1.0, bias]), jac=True, method="L-BFGS-B", bounds=[(1e-8, None), (None, None)]
    ).x
    return float(theta[0] * scale), np.array([theta[1]])


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
        Per-class misclassification costs c. They would cancel in the class
        means of Eq. 3-4, so they act on the calibrated decision instead: the
        closed-form calibration adds log(c) to each class's logit, and "platt"
        weights each sample's likelihood by the cost of its class. "balanced"
        sets c inversely proportional to each class's total sample weight.
        None is accepted as an alias of "none".
    threshold : {"sefr", "balanced"}, default="sefr"
        "sefr" is the class-count-weighted score average of Eq. 9. "balanced"
        is the unweighted midpoint of the two class score means.
    threshold_shift : float, default=0.0
        Shifts the bias by this many standard deviations of the training scores.
    calibration : {"sigmoid", "platt"}, default="sigmoid"
        SEFR produces margins, not probabilities. Margins m map to calibrated
        logits a * m + b, with one scale a > 0 shared by all heads and one bias
        b per head; probabilities are their sigmoid (binary) or softmax
        (multiclass). `decision_function` returns these logits and `predict`
        thresholds them at 0 (binary) or takes their argmax (multiclass), so
        `predict`, `decision_function` and `predict_proba` always agree.
        "sigmoid" is closed form: a is the inverse standard deviation of the
        training margins; b is log(c_1 / c_0) for binary targets, which keeps
        SEFR's Eq. 9 threshold unless class_weight is set, and the log of the
        cost-weighted class prior for multiclass targets.
        "platt" fits a and b by weighted maximum likelihood on the training
        margins, so the fitted intercept replaces the Eq. 9 threshold. It is
        iterative, so it gives up the closed form, but it usually gives a much
        better log_loss.
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
        return {"requires_positive_X": self.scaling == "none", "X_types": ["2darray", "sparse"]}

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
        for name, positive in (("eps", True), ("threshold_shift", False)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
            if positive and value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")

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
        if issparse(X) and self.offset_ is not None and np.any(self.offset_):
            # fitted on dense data with a nonzero minimum: subtracting it densifies
            # the input anyway, and skipping it would score sparse input differently
            X = X.toarray()
        if issparse(X):
            # multiply() keeps the matrix sparse; dividing by a dense row would not
            X = X.multiply(1.0 / self.spread_).tocsr()
            np.clip(X.data, 0.0, 1.0, out=X.data)
            return X
        if self.offset_ is not None:
            X = X - self.offset_
        return np.clip(X / self.spread_, 0.0, 1.0)

    def _class_costs(self, class_totals):
        if self.class_weight == "balanced":
            return class_totals.sum() / (self.classes_.size * class_totals)
        if isinstance(self.class_weight, dict):
            costs = np.array([self.class_weight.get(label, 1.0) for label in self.classes_], dtype=np.float64)
            if not np.all(np.isfinite(costs)) or np.any(costs < 0):
                raise ValueError("class_weight values must be finite and non-negative")
            return costs
        return np.ones(self.classes_.size)

    def _fit_heads(self, X, y_index, sample_weight, class_totals):
        """Closed-form coef and Eq. 9 bias of every one-vs-rest head, from one pass over ``X``."""
        sums = _class_sums(X, y_index, sample_weight, self.classes_.size)
        if self.classes_.size == 2:
            pos_sums, pos_totals = sums[[1]], class_totals[[1]]
            neg_sums, neg_totals = sums[[0]], class_totals[[0]]
        else:
            pos_sums, pos_totals = sums, class_totals
            neg_sums, neg_totals = sums.sum(axis=0) - sums, class_totals.sum() - class_totals
        avg_pos = pos_sums / pos_totals[:, None]
        avg_neg = neg_sums / neg_totals[:, None]
        coef = (avg_pos - avg_neg) / (avg_pos + avg_neg + self.eps)

        # scores are linear in x, so each class's mean score is its mean row times coef
        score_pos = np.einsum("hd,hd->h", avg_pos, coef)
        score_neg = np.einsum("hd,hd->h", avg_neg, coef)
        if self.threshold == "balanced":
            bias = 0.5 * (score_pos + score_neg)
        else:
            bias = (neg_totals * score_pos + pos_totals * score_neg) / (neg_totals + pos_totals)
        score_mean = (sums.sum(axis=0) / class_totals.sum()) @ coef.T
        return coef, bias, score_mean

    def _score_std(self, X, sample_weight, score_mean):
        """Weighted standard deviation of each head's training scores, in row chunks."""
        chunk = max(1, _CHUNK_ENTRIES // self.coef_.shape[0])
        second = np.zeros(self.coef_.shape[0])
        for start in range(0, X.shape[0], chunk):
            centered = np.asarray(X[start : start + chunk] @ self.coef_.T) - score_mean
            second += sample_weight[start : start + chunk] @ (centered * centered)
        return np.sqrt(second / sample_weight.sum())

    def _calibration_rows(self, n_samples, y_index, weights):
        """All rows, or a deterministic subsample that keeps a positive-weight row of every class."""
        n_logits = 2 if self.classes_.size == 2 else self.classes_.size
        if n_samples * n_logits <= _CALIBRATION_MAX_ENTRIES:
            return np.arange(n_samples)
        stride = -(-n_samples * n_logits // _CALIBRATION_MAX_ENTRIES)
        positive = np.flatnonzero(weights > 0)
        first_positive = positive[np.unique(y_index[positive], return_index=True)[1]]
        rows = np.union1d(np.arange(0, n_samples, stride), first_positive)
        # a class without weight in the subsample would drive its bias to -inf
        self._class_totals(y_index[rows], weights[rows])
        return rows

    def _fit_calibration(self, X, y_index, sample_weight, costs, score_mean, score_std):
        """Return the scale ``a`` and per-head biases ``b`` of the calibrated logits."""
        # pooled spread of the margins (scores - intercept) over every head and sample
        margin_mean = score_mean - self.intercept_
        spread = np.sqrt(np.mean(score_std**2 + (margin_mean - margin_mean.mean()) ** 2))
        scale = 1.0 / (spread or 1.0)
        weights = sample_weight * costs[y_index]
        if self.classes_.size == 2:
            bias = np.log(costs[[1]] / costs[[0]])
        else:
            bias = np.log(np.bincount(y_index, weights=weights, minlength=self.classes_.size) / weights.sum())
        if self.calibration == "sigmoid":
            return scale, bias

        rows = self._calibration_rows(X.shape[0], y_index, weights)
        margins = np.asarray(X[rows] @ self.coef_.T) - self.intercept_
        if self.classes_.size == 2:
            return _fit_binary(margins[:, 0], y_index[rows] == 1, weights[rows], scale, bias[0])
        return _fit_multinomial(margins, y_index[rows], weights[rows], scale, bias)

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
        costs = self._class_costs(class_totals)
        self._class_totals(y_index, sample_weight * costs[y_index])

        # zero-weight samples take no part in the fit, including the feature range
        self._fit_scaler(X if np.all(sample_weight > 0) else X[sample_weight > 0])
        X = self._scale(X)

        self.coef_, self.intercept_, score_mean = self._fit_heads(X, y_index, sample_weight, class_totals)
        score_std = self._score_std(X, sample_weight, score_mean)
        if self.threshold_shift:
            self.intercept_ = self.intercept_ + self.threshold_shift * np.where(score_std > 0, score_std, 1.0)
        self.calibration_scale_, self.calibration_bias_ = self._fit_calibration(
            X, y_index, sample_weight, costs, score_mean, score_std
        )
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = _validate_data(self, X=X, accept_sparse="csr", dtype=np.float64, reset=False)
        scores = np.asarray(self._scale(X) @ self.coef_.T) - self.intercept_
        scores = scores * self.calibration_scale_ + self.calibration_bias_
        return scores.ravel() if self.coef_.shape[0] == 1 else scores

    def predict(self, X):
        scores = self.decision_function(X)
        if self.coef_.shape[0] == 1:
            return self.classes_[(scores > 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        if self.coef_.shape[0] == 1:
            proba = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
            return np.column_stack([1.0 - proba, proba])
        proba = np.exp(scores - scores.max(axis=1, keepdims=True))
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
        base_space = SEFREstimator.search_space(**params)
        space["class_weight"] = base_space["class_weight"]
        # AdaBoost's SAMME.R weights the base learners by their probabilities,
        # so fitted calibration pays off more here than for a single SEFR model
        space["calibration"] = dict(base_space["calibration"], init_value="platt")
        return space

    @classmethod
    def cost_relative2lgbm(cls) -> float:
        return 5.0

    def config2params(self, config: dict) -> dict:
        params = super().config2params(config)
        params.pop("n_jobs", None)
        base_params = {key: params.pop(key) for key in ("scaling", "class_weight", "calibration") if key in params}
        params[_adaboost_estimator_param()] = SEFRClassifier(**base_params)
        if "random_state" not in params:
            params["random_state"] = 24092023
        return params

    def __init__(self, task: Task = "binary", **config):
        super().__init__(task, **config)
        assert self._task.is_classification(), "SEFR is for classification tasks only"
        self.estimator_class = AdaBoostClassifier
