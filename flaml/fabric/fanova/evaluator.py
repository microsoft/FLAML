import warnings
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Dict, Mapping, Optional

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    DiscreteUniformDistribution,
    IntLogUniformDistribution,
    IntUniformDistribution,
    LogUniformDistribution,
    UniformDistribution,
)
from optuna.exceptions import ExperimentalWarning
from optuna.importance import FanovaImportanceEvaluator as OptunaFanovaImportanceEvaluator
from optuna.importance._fanova._fanova import _Fanova
from optuna.importance._fanova._tree import _FanovaTree
from optuna.trial import create_trial

try:
    from optuna.distributions import FloatDistribution, IntDistribution
except ImportError:
    FloatDistribution = IntDistribution = None


class FanovaImportanceEvaluator:
    """Optuna-backed fANOVA importance evaluator.

    This adapter preserves FLAML's DataFrame-based ``evaluate`` interface while delegating the
    actual importance computation to Optuna's pure-Python
    ``optuna.importance.FanovaImportanceEvaluator``.

    Args:
        n_trees:
            The number of trees in the forest.
        max_depth:
            The maximum depth of the trees in the forest.
        seed:
            Controls the randomness of the forest. For deterministic behavior, specify a value
            other than :obj:`None`.
    """

    def __init__(
        self,
        *,
        n_trees: int = 64,
        max_depth: int = 64,
        seed: Optional[int] = None,
    ) -> None:
        self._evaluator = OptunaFanovaImportanceEvaluator(
            n_trees=n_trees,
            max_depth=max_depth,
            seed=seed,
        )
        self._evaluator._evaluator = _LocalFanova(
            n_trees=n_trees, max_depth=max_depth, min_samples_split=2, min_samples_leaf=1, seed=seed
        )

    def evaluate(
        self,
        hp_df,
        scores,
        search_space: Mapping[str, BaseDistribution],
    ) -> Dict[str, float]:
        param_names = list(search_space)
        if not param_names:
            return {}

        missing_columns = [name for name in param_names if name not in hp_df.columns]
        if missing_columns:
            raise ValueError(f"Missing hyperparameter columns required by search_space: {missing_columns}")

        hp_df = hp_df.reset_index(drop=True)
        score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(hp_df) != len(score_array):
            raise ValueError("`hp_df` and `scores` must have the same number of rows.")

        valid_mask = np.isfinite(score_array)
        for name, distribution in search_space.items():
            present = hp_df[name].notna()
            if isinstance(distribution, CategoricalDistribution):
                present |= hp_df[name].map(lambda value: value is None)
            valid_mask &= present.to_numpy()
        if valid_mask.sum() < 2:
            return {}

        trial_distributions = _expand_distributions_to_observations(search_space, hp_df, valid_mask, param_names)
        study = optuna.create_study(direction="maximize")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            for params, score in zip(hp_df.loc[valid_mask, param_names].to_dict("records"), score_array[valid_mask]):
                study.add_trial(
                    create_trial(
                        value=float(score),
                        params={
                            name: _normalize_param_value(name, params[name], trial_distributions[name])
                            for name in param_names
                        },
                        distributions=trial_distributions,
                    )
                )

        return self._evaluator.evaluate(study, params=param_names)


def _normalize_param_value(name: str, value, distribution: BaseDistribution):
    if isinstance(value, np.generic):
        value = value.item()

    if _is_int_distribution(distribution):
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"Parameter {name!r} has non-integral value {value!r} for integer distribution.")
        return int(value)

    if _is_float_distribution(distribution):
        return float(value)

    if isinstance(distribution, CategoricalDistribution):
        return value

    return value


def _expand_distributions_to_observations(
    search_space: Mapping[str, BaseDistribution],
    hp_df,
    valid_mask: np.ndarray,
    param_names,
) -> Dict[str, BaseDistribution]:
    trial_distributions = {}
    for name in param_names:
        distribution = search_space[name]
        values = [_normalize_param_value(name, value, distribution) for value in hp_df.loc[valid_mask, name]]
        trial_distributions[name] = _expand_distribution(distribution, values)
    return trial_distributions


def _expand_distribution(distribution: BaseDistribution, values) -> BaseDistribution:
    """Widen replay bounds while retaining compatible log scales and step grids."""
    if all(_contains_value(distribution, value) for value in values):
        return distribution

    if _is_int_distribution(distribution):
        low = min([int(getattr(distribution, "low"))] + [int(value) for value in values])
        high = max([int(getattr(distribution, "high"))] + [int(value) for value in values])
        step = getattr(distribution, "step", 1)
        logarithmic = getattr(distribution, "log", isinstance(distribution, IntLogUniformDistribution))
        if any((int(value) - distribution.low) % step != 0 for value in values):
            step = 1
        if IntDistribution is not None and isinstance(distribution, IntDistribution):
            return IntDistribution(low=low, high=high, log=logarithmic and low > 0, step=step)
        if logarithmic and low > 0:
            return IntLogUniformDistribution(low, high)
        return IntUniformDistribution(low, high, step=step)

    if _is_float_distribution(distribution):
        low = min([float(getattr(distribution, "low"))] + [float(value) for value in values])
        high = max([float(getattr(distribution, "high"))] + [float(value) for value in values])
        quantum = getattr(distribution, "step", getattr(distribution, "q", None))
        logarithmic = getattr(distribution, "log", isinstance(distribution, LogUniformDistribution))
        if quantum is not None:
            origin = Decimal(str(distribution.low))
            step = Decimal(str(quantum))
            low_index = ((Decimal(str(low)) - origin) / step).to_integral_value(rounding=ROUND_FLOOR)
            high_index = ((Decimal(str(high)) - origin) / step).to_integral_value(rounding=ROUND_CEILING)
            expanded = _float_distribution(
                distribution, float(origin + low_index * step), float(origin + high_index * step), step=quantum
            )
            if all(_contains_value(expanded, value) for value in values):
                return expanded
        return _float_distribution(distribution, low, high, log=logarithmic and low > 0)

    if isinstance(distribution, CategoricalDistribution):
        choices = list(distribution.choices)
        for value in values:
            if value not in choices:
                choices.append(value)
        return CategoricalDistribution(choices)

    return distribution


def _float_distribution(original, low, high, *, log=False, step=None):
    if FloatDistribution is not None and isinstance(original, FloatDistribution):
        return FloatDistribution(low=low, high=high, log=log, step=step)
    if step is not None:
        return DiscreteUniformDistribution(low, high, q=step)
    if log:
        return LogUniformDistribution(low, high)
    return UniformDistribution(low, high)


def _contains_value(distribution: BaseDistribution, value) -> bool:
    try:
        return distribution._contains(distribution.to_internal_repr(value))
    except ValueError:
        return False


def _is_int_distribution(distribution: BaseDistribution) -> bool:
    if isinstance(distribution, (IntUniformDistribution, IntLogUniformDistribution)):
        return True
    return IntDistribution is not None and isinstance(distribution, IntDistribution)


def _is_float_distribution(distribution: BaseDistribution) -> bool:
    if isinstance(distribution, (UniformDistribution, LogUniformDistribution, DiscreteUniformDistribution)):
        return True
    return FloatDistribution is not None and isinstance(distribution, FloatDistribution)


class _ScalarFanovaTree(_FanovaTree):
    def _get_node_value(self, node_index: int) -> float:
        return float(np.asarray(super()._get_node_value(node_index)).item())


class _LocalFanova(_Fanova):
    """Keep Optuna 2/3's NumPy-2 scalar compatibility local to this evaluator."""

    def fit(self, X, y, search_spaces, column_to_encoded_columns):
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == search_spaces.shape[0]
        assert search_spaces.shape[1] == 2
        self._forest.fit(X, y)
        self._trees = [_ScalarFanovaTree(estimator.tree_, search_spaces) for estimator in self._forest.estimators_]
        self._column_to_encoded_columns = column_to_encoded_columns
        self._variances = {}
        if all(tree.variance == 0 for tree in self._trees):
            raise RuntimeError("Encountered zero total variance in all trees.")
