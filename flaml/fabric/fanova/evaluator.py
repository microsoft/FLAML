import warnings
from typing import Dict, Mapping, Optional

import numpy as np
import optuna
from optuna.distributions import BaseDistribution, CategoricalDistribution, IntUniformDistribution, UniformDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.importance import FanovaImportanceEvaluator as OptunaFanovaImportanceEvaluator
from optuna.importance._fanova import _tree as optuna_fanova_tree
from optuna.trial import create_trial


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
        _patch_optuna_fanova_tree()

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

        valid_mask = hp_df[param_names].notna().all(axis=1).to_numpy() & np.isfinite(score_array)
        if valid_mask.sum() < 2:
            return {}

        trial_distributions = {name: search_space[name] for name in param_names}
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

    if isinstance(distribution, IntUniformDistribution):
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"Parameter {name!r} has non-integral value {value!r} for integer distribution.")
        return int(value)

    if isinstance(distribution, UniformDistribution):
        return float(value)

    if isinstance(distribution, CategoricalDistribution):
        return value

    return value


def _patch_optuna_fanova_tree() -> None:
    original_get_node_value = optuna_fanova_tree._FanovaTree._get_node_value
    if getattr(original_get_node_value, "__name__", "") == "_flaml_get_node_value":
        return

    def _flaml_get_node_value(self, node_index: int) -> float:
        value = original_get_node_value(self, node_index)
        return float(np.ravel(value)[0])

    optuna_fanova_tree._FanovaTree._get_node_value = _flaml_get_node_value
