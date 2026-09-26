"""SEFR: a single-pass, linear-time classifier.

Reference:
    Keshavarz, Saniee Abadeh, Rawassizadeh (2020).
    "SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices".
    https://arxiv.org/abs/2006.04620

SEFR derives one weight per feature plus a bias from class-conditional feature
means, so fitting is a single O(n_samples * n_features) pass with no iterative
optimization. The trained model is ``n_features + 1`` floats. That makes it the
cheapest learner in the FLAML portfolio and the only one that reliably returns a
model within a very small ``time_budget`` on large data.

It is implemented here in NumPy rather than pulled in as a dependency: the whole
algorithm is a handful of array operations, and FLAML's only required dependency
is NumPy.
"""

from __future__ import annotations

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


class SEFRClassifier(ClassifierMixin, SKLearnBaseEstimator):
    """Scalable, Efficient and Fast classifieR (SEFR).

    Binary targets are fit with the closed form of the paper (Eqs. 3-9);
    multiclass targets use the one-vs-rest scheme of Sec. 3.4.

    Parameters
    ----------
    scaling : {"minmax", "maxabs", "none"}, default="minmax"
        SEFR's weight formula assumes non-negative features, and FLAML does not
        scale features anywhere in its pipeline, so scaling belongs to the
        estimator. Sparse input falls back to "maxabs", which preserves sparsity.
    class_weight : {"none", "balanced"}, default="none"
        "balanced" reweights samples inversely to class frequency before the
        means of Eq. 3-4 are taken.
    threshold : {"sefr", "balanced"}, default="sefr"
        "sefr" is the class-count-weighted score average of Eq. 9. "balanced"
        is the unweighted midpoint of the two class score means.
    threshold_shift : float, default=0.0
        Shifts the bias by this many standard deviations of the training scores.
        Only affects `predict`, not the ranking produced by `decision_function`.
    calibration : {"platt", "sigmoid"}, default="platt"
        SEFR produces margins, not probabilities. "sigmoid" squashes the margin
        by its training standard deviation; "platt" fits a one-dimensional
        logistic regression on the training margins. Both are monotone, so the
        choice does not affect ROC AUC, but it matters a great deal for
        log_loss, which is FLAML's default multiclass metric.
    eps : float, default=1e-7
        Stabilizer for the denominator of Eq. 5.
    """

    def __init__(
        self,
        scaling="minmax",
        class_weight="none",
        threshold="sefr",
        threshold_shift=0.0,
        calibration="platt",
        eps=_EPS,
    ):
        self.scaling = scaling
        self.class_weight = class_weight
        self.threshold = threshold
        self.threshold_shift = threshold_shift
        self.calibration = calibration
        self.eps = eps

    def _fit_scaler(self, X):
        scaling = self.scaling
        if issparse(X) and scaling == "minmax":
            # subtracting a per-column minimum would densify the matrix
            scaling = "maxabs"
        if scaling == "minmax":
            self.offset_ = _to_dense_row(X.min(axis=0))
            spread = _to_dense_row(X.max(axis=0)) - self.offset_
        elif scaling == "maxabs":
            self.offset_ = None
            spread = _to_dense_row(abs(X).max(axis=0))
        else:
            self.offset_ = None
            spread = None
        if spread is not None:
            spread[spread == 0] = 1.0
        self.spread_ = spread

    def _scale(self, X):
        if self.spread_ is None:
            return X
        if issparse(X):
            # multiply() keeps the matrix sparse; dividing by a dense row would not
            return X.multiply(1.0 / self.spread_).tocsr()
        if self.offset_ is None:
            return X / self.spread_
        return (X - self.offset_) / self.spread_

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
            bias += self.threshold_shift * (scores.std() or 1.0)
        return coef, bias, scores - bias

    def _fit_calibrator(self, margins, target):
        if self.calibration == "sigmoid":
            return float(margins.std() or 1.0)
        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(margins.reshape(-1, 1), target)
        return calibrator

    @staticmethod
    def _apply_calibrator(calibrator, margins):
        if isinstance(calibrator, float):
            return 1.0 / (1.0 + np.exp(-np.clip(margins / calibrator, -30, 30)))
        return calibrator.predict_proba(margins.reshape(-1, 1))[:, 1]

    def fit(self, X, y, sample_weight=None):
        if not issparse(X):
            X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        if self.classes_.size < 2:
            raise ValueError("SEFR needs at least two classes in the training data")

        if sample_weight is None:
            sample_weight = np.ones(y.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
        if self.class_weight == "balanced":
            counts = {c: sample_weight[y == c].sum() for c in self.classes_}
            scale = y.shape[0] / (self.classes_.size * np.array([counts[c] for c in self.classes_]))
            lookup = dict(zip(self.classes_, scale))
            sample_weight = sample_weight * np.array([lookup[label] for label in y])

        self._fit_scaler(X)
        X = self._scale(X)

        heads = [self.classes_[1]] if self.classes_.size == 2 else list(self.classes_)
        coefs, biases, self.calibrators_ = [], [], []
        for label in heads:
            is_positive = y == label
            coef, bias, margins = self._fit_head(X, is_positive, sample_weight)
            coefs.append(coef)
            biases.append(bias)
            self.calibrators_.append(self._fit_calibrator(margins, is_positive.astype(int)))
        self.coef_ = np.vstack(coefs)
        self.intercept_ = np.array(biases)
        return self

    def decision_function(self, X):
        if not issparse(X):
            X = np.asarray(X, dtype=np.float64)
        scores = np.asarray(self._scale(X) @ self.coef_.T) - self.intercept_
        return scores.ravel() if self.coef_.shape[0] == 1 else scores

    def predict(self, X):
        scores = self.decision_function(X)
        if self.coef_.shape[0] == 1:
            return self.classes_[(scores > 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        if self.coef_.shape[0] == 1:
            positive = self._apply_calibrator(self.calibrators_[0], scores)
            return np.column_stack([1.0 - positive, positive])
        proba = np.column_stack([self._apply_calibrator(cal, scores[:, i]) for i, cal in enumerate(self.calibrators_)])
        total = proba.sum(axis=1, keepdims=True)
        total[total == 0] = 1.0
        return proba / total


class SEFREstimator(SKLearnEstimator):
    """The class for tuning SEFR."""

    @classmethod
    def search_space(cls, **params) -> dict:
        return {
            "scaling": {
                "domain": tune.choice(["minmax", "maxabs"]),
                "init_value": "minmax",
            },
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
                "domain": tune.choice(["platt", "sigmoid"]),
                "init_value": "platt",
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
    single-pass base learner while giving FLAML a real, cheap-to-traverse search
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
        space.update(
            {
                key: value
                for key, value in SEFREstimator.search_space(**params).items()
                if key in ("scaling", "class_weight")
            }
        )
        return space

    @classmethod
    def cost_relative2lgbm(cls) -> float:
        return 5.0

    def config2params(self, config: dict) -> dict:
        params = super().config2params(config)
        params.pop("n_jobs", None)
        base_params = {key: params.pop(key) for key in ("scaling", "class_weight") if key in params}
        params["estimator"] = SEFRClassifier(**base_params)
        if "random_state" not in params:
            params["random_state"] = 24092023
        return params

    def __init__(self, task: Task = "binary", **config):
        super().__init__(task, **config)
        assert self._task.is_classification(), "SEFR is for classification tasks only"
        self.estimator_class = AdaBoostClassifier
