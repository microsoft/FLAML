# !
#  * Copyright (c) Microsoft Corporation. All rights reserved.
#  * Licensed under the MIT License. See LICENSE file in the
#  * project root for license information.
from typing import Dict, Optional, Union
import numpy as np


try:
    from ray import __version__ as ray_version

    assert ray_version >= "1.10.0"
    if ray_version.startswith("1."):
        from ray.tune.suggest import Searcher
    else:
        from ray.tune.search import Searcher
except (ImportError, AssertionError):
    from .suggestion import Searcher
from .flow2 import FLOW2
from ..space import add_cost_to_space, unflatten_hierarchical
from ..result import TIME_TOTAL_S
from ..utils import get_lexico_bound
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SearchThread:
    """Class of global or local search thread."""

    def __init__(
        self,
        mode: str = "min",
        search_alg: Optional[Searcher] = None,
        cost_attr: Optional[str] = TIME_TOTAL_S,
        eps: Optional[float] = 1.0,
    ):
        """When search_alg is omitted, use local search FLOW2."""
        self._search_alg = search_alg
        self._is_ls = isinstance(search_alg, FLOW2)
        self._mode = mode
        self._metric_op = 1 if mode == "min" else -1
        self.cost_best = self.cost_last = self.cost_total = self.cost_best1 = getattr(search_alg, "cost_incumbent", 0)
        self._eps = eps
        self.cost_best2 = 0
        self.lexico_objectives = getattr(self._search_alg, "lexico_objectives", None)
        self.best_result = None
        # eci: estimated cost for improvement
        self.eci = self.cost_best
        self._init_config = True
        self.running = 0  # the number of running trials from the thread
        self.cost_attr = cost_attr
        if search_alg:
            self.space = self._space = search_alg.space  # unflattened space
            if self.space and not isinstance(search_alg, FLOW2) and isinstance(search_alg._space, dict):
                # remember const config
                self._const = add_cost_to_space(self.space, {}, {})

        if self.lexico_objectives:
            # lexicographic tuning setting
            self.f_best, self.histories = {}, defaultdict(list)  # only use for lexico_comapre.
            self.obj_best1 = self.obj_best2 = {}
            for k_metric in self.lexico_objectives["metrics"]:
                self.obj_best1[k_metric] = self.obj_best2[k_metric] = (
                    np.inf if getattr(search_alg, "best_obj", None) is None else search_alg.best_obj[k_metric]
                )
            self.priority, self.speed = {}, {}
            for k_metric in self.lexico_objectives["metrics"]:
                self.priority[k_metric] = self.speed[k_metric] = 0
        else:
            # normal tuning setting
            self.obj_best1 = self.obj_best2 = getattr(search_alg, "best_obj", np.inf)  # inherently minimize
            self.priority = self.speed = 0

    def suggest(self, trial_id: str) -> Optional[Dict]:
        """Use the suggest() of the underlying search algorithm."""
        if isinstance(self._search_alg, FLOW2):
            config = self._search_alg.suggest(trial_id)
        else:
            try:
                config = self._search_alg.suggest(trial_id)
                if isinstance(self._search_alg._space, dict):
                    config.update(self._const)
                else:
                    # define by run
                    config, self.space = unflatten_hierarchical(config, self._space)
            except FloatingPointError:
                logger.warning("The global search method raises FloatingPointError. " "Ignoring for this iteration.")
                config = None
        if config is not None:
            self.running += 1
        return config

    def update_lexicoPara(self, result):
        # update histories, f_best
        if self.lexico_objectives:
            for k_metric, k_mode in zip(self.lexico_objectives["metrics"], self.lexico_objectives["modes"]):
                self.histories[k_metric].append(result[k_metric]) if k_mode == "min" else self.histories[
                    k_metric
                ].append(result[k_metric] * -1)
            obj_initial = self.lexico_objectives["metrics"][0]
            feasible_index = np.array([*range(len(self.histories[obj_initial]))])
            for k_metric in self.lexico_objectives["metrics"]:
                k_values = np.array(self.histories[k_metric])
                feasible_value = k_values.take(feasible_index)
                self.f_best[k_metric] = np.min(feasible_value)
                if not isinstance(self.lexico_objectives["tolerances"][k_metric], str):
                    tolerance_bound = self.f_best[k_metric] + self.lexico_objectives["tolerances"][k_metric]
                else:
                    assert (
                        self.lexico_objectives["tolerances"][k_metric][-1] == "%"
                    ), "String tolerance of {} should use %% as the suffix".format(k_metric)
                    tolerance_bound = self.f_best[k_metric] * (
                        1 + 0.01 * float(self.lexico_objectives["tolerances"][k_metric].replace("%", ""))
                    )
                feasible_index_filter = np.where(
                    feasible_value
                    <= max(
                        tolerance_bound,
                        self.lexico_objectives["targets"][k_metric],
                    )
                )[0]
                feasible_index = feasible_index.take(feasible_index_filter)

    def update_priority(self, eci: Optional[float] = 0):
        # optimistic projection
        if self.lexico_objectives:
            for k_metric, k_mode in zip(self.lexico_objectives["metrics"], self.lexico_objectives["modes"]):
                self.priority[k_metric] = eci * self.speed[k_metric] - self.obj_best1[k_metric]
        else:
            self.priority = eci * self.speed - self.obj_best1

    def update_eci(self, metric_target: float, max_speed: Optional[float] = np.inf, min_speed: Optional[float] = 1e-9):
        # calculate eci: estimated cost for improvement over metric_target
        if not self.lexico_objectives:
            _metric_op = self._metric_op
            if not self.speed:
                self.speed = max_speed
        else:
            _metric_1st = self.lexico_objectives["metrics"][0]
            _metric_op = 1 if self.lexico_objectives["modes"][0] == "min" else -1
            if self.speed[_metric_1st] == 0:
                self.speed[_metric_1st] = max_speed[_metric_1st]
            elif self.speed[_metric_1st] == -1:
                self.speed[_metric_1st] = min_speed[_metric_1st]
        best_obj = metric_target * _metric_op
        self.eci = max(self.cost_total - self.cost_best1, self.cost_best1 - self.cost_best2)
        obj_best1 = self.obj_best1 if not self.lexico_objectives else self.obj_best1[_metric_1st]
        speed = self.speed if not self.lexico_objectives else self.speed[_metric_1st]
        if obj_best1 > best_obj and speed > 0:
            self.eci = max(self.eci, 2 * (obj_best1 - best_obj) / speed)

    def _better(self, obj_1: Union[dict, float], obj_2: Union[dict, float]):
        if self.lexico_objectives:
            for k_metric, k_mode in zip(self.lexico_objectives["metrics"], self.lexico_objectives["modes"]):
                _f_best = self._search_alg.f_best if self._is_ls else self.f_best
                bound = get_lexico_bound(k_metric, k_mode, self.lexico_objectives, _f_best)
                if (obj_1[k_metric] < bound) and (obj_2[k_metric] < bound):
                    continue
                elif obj_1[k_metric] < obj_2[k_metric]:
                    return True, k_metric
                else:
                    return False, None
            for k_metr in self.lexico_objectives["metrics"]:
                if obj_1[k_metr] == obj_2[k_metr]:
                    continue
                elif obj_1[k_metr] < obj_2[k_metr]:
                    return True, k_metric
                else:
                    return False, None
            return False, None
        else:
            if obj_1 < obj_2:
                return True, None
            else:
                return False, None

    def _update_speed(self):
        # calculate speed; use 0 for invalid speed temporarily
        if not self.lexico_objectives:
            if self.obj_best1 < self.obj_best2:
                self.speed = (
                    (self.obj_best2 - self.obj_best1)
                    / self.running
                    / (max(self.cost_total - self.cost_best2, self._eps))
                )
            else:
                self.speed = 0
        elif (self._is_ls and self._search_alg.histories) or (not self._is_ls and self.histories):
            _is_better, _op_dimension = self._better(self.obj_best1, self.obj_best2)
            if _is_better:
                op_index = self.lexico_objectives["metrics"].index(_op_dimension)
                self.speed[_op_dimension] = (
                    (self.obj_best2[_op_dimension] - self.obj_best1[_op_dimension])
                    / self.running
                    / (max(self.cost_total - self.cost_best2, self._eps))
                )
                for i in range(0, len(self.lexico_objectives["metrics"])):
                    if i < op_index:
                        self.speed[self.lexico_objectives["metrics"][i]] = -1
                    elif i > op_index:
                        self.speed[self.lexico_objectives["metrics"][i]] = 0
            else:
                for k_metric in self.lexico_objectives["metrics"]:
                    self.speed[k_metric] = 0
        else:
            return

    def on_trial_complete(self, trial_id: str, result: Optional[Dict] = None, error: bool = False):
        """Update the statistics of the thread."""
        if not self._search_alg:
            return
        if not hasattr(self._search_alg, "_ot_trials") or (not error and trial_id in self._search_alg._ot_trials):
            # optuna doesn't handle error
            if self._is_ls or not self._init_config:
                try:
                    self._search_alg.on_trial_complete(trial_id, result, error)
                    if not self._is_ls:
                        self.update_lexicoPara(result)
                except RuntimeError as e:
                    # rs is used in place of optuna sometimes
                    if not str(e).endswith("has already finished and can not be updated."):
                        raise e
            else:
                # init config is not proposed by self._search_alg
                # under this thread
                self._init_config = False
        if result:
            self.cost_last = result.get(self.cost_attr, 1)
            self.cost_total += self.cost_last
            _metric_exists = (
                self._search_alg.metric in result
                if not self.lexico_objectives
                else all(x in result for x in self.lexico_objectives["metrics"])
            )
            if _metric_exists:
                if not self.lexico_objectives:
                    obj = result[self._search_alg.metric] * self._metric_op
                else:
                    obj = {}
                    for k, m in zip(
                        self._search_alg.lexico_objectives["metrics"], self._search_alg.lexico_objectives["modes"]
                    ):
                        obj[k] = -1 * result[k] if m == "max" else result[k]
                if self.best_result is None or self._better(obj, self.obj_best1)[0]:
                    self.cost_best2 = self.cost_best1
                    self.cost_best1 = self.cost_total
                    if not self.lexico_objectives:
                        self.obj_best2 = obj if np.isinf(self.obj_best1) else self.obj_best1
                    else:
                        self.obj_best2 = (
                            obj if np.isinf(self.obj_best1[self.lexico_objectives["metrics"][0]]) else self.obj_best1
                        )
                    self.obj_best1 = obj
                    self.cost_best = self.cost_last
                    self.best_result = result
            self._update_speed()
        self.running -= 1
        assert self.running >= 0

    def on_trial_result(self, trial_id: str, result: Dict):
        # TODO update the statistics of the thread with partial result?
        if not self._search_alg:
            return
        if not hasattr(self._search_alg, "_ot_trials") or (trial_id in self._search_alg._ot_trials):
            try:
                self._search_alg.on_trial_result(trial_id, result)
                if not self._is_ls:
                    self.update_lexicoPara(result)
            except RuntimeError as e:
                # rs is used in place of optuna sometimes
                if not str(e).endswith("has already finished and can not be updated."):
                    raise e
        new_cost = result.get(self.cost_attr, 1)
        if self.cost_last < new_cost:
            self.cost_last = new_cost

    @property
    def converged(self) -> bool:
        return self._search_alg.converged

    @property
    def resource(self) -> float:
        return self._search_alg.resource

    def reach(self, thread) -> bool:
        """Whether the incumbent can reach the incumbent of thread."""
        return self._search_alg.reach(thread._search_alg)

    @property
    def can_suggest(self) -> bool:
        """Whether the thread can suggest new configs."""
        return self._search_alg.can_suggest
