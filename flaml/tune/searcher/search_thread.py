# !
#  * Copyright (c) Microsoft Corporation. All rights reserved.
#  * Licensed under the MIT License. See LICENSE file in the
#  * project root for license information.
from typing import Dict, Optional
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
import logging

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

        # get obj 1 and obj 2
        if self._is_ls and self.lexico_objectives:
            self.obj_best1 = self.obj_best2 = {}
            for k in self.lexico_objectives["metrics"]:
                self.obj_best1[k] = self.obj_best2[k] = np.inf if getattr(
                    search_alg, "best_obj", None) is None else search_alg.best_obj[k]
        else:
            self.obj_best1 = self.obj_best2 = getattr(
                search_alg, "best_obj", np.inf
            )  # inherently minimize

        self.best_result = None
        # eci: estimated cost for improvement
        self.eci = self.cost_best
        if self._is_ls and self.lexico_objectives:
            self.priority = self.speed = {}
        else:
            self.priority = self.speed = 0
        self._init_config = True
        self.running = 0  # the number of running trials from the thread
        self.cost_attr = cost_attr
        if search_alg:
            self.space = self._space = search_alg.space  # unflattened space
            if self.space and not isinstance(search_alg, FLOW2) and isinstance(search_alg._space, dict):
                # remember const config
                self._const = add_cost_to_space(self.space, {}, {})

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

    def _get_lexico_bound(self, metric, mode):

        k_target = (
            self.lexico_objectives["targets"][metric]
            if mode == "min"
            else -self.lexico_objectives["targets"][metric]
        )
        if not isinstance(self.lexico_objectives["tolerances"][metric], str):
            tolerance_bound = (
                self.f_best[metric]
                + self.lexico_objectives["tolerances"][metric]
            )
        else:
            assert (
                self.lexico_objectives["tolerances"][metric][-1] == "%"
            ), "String tolerance of {} should use %% as the suffix".format(
                metric
            )
            tolerance_bound = self.f_best[metric] * (
                1
                + 0.01
                * float(
                    self.lexico_objectives["tolerances"][metric].replace(
                        "%", ""
                    )
                )
            )
        bound = max(tolerance_bound, k_target)
        return bound

    def update_priority(self, eci: Optional[float] = 0):
        if self.lexico_objectives and self._is_ls:
            feasible_flag = True
            for k_metric, k_mode in zip(
                self.lexico_objectives["metrics"], self.lexico_objectives["modes"]
            ):
                k_bound = _get_lexico_bound(k_metric, k_mode)
                if self.obj_best1[k_metric] < k_bound:
                    self.priority[k_metric] = self.obj_best1[k_metric]
                    continue
                elif self.obj_best1[k_metric] < self.f_best[k_metric]:
                    raise ValueError(
                        "Bug exists: Performance of a specific search thread is not possible to be better than global best.")
                else:
                    start_point = self.obj_best1[k_metric] if feasible_flag else self.f_worst[k_metric]
                    if start_point <= k_bound:
                        continue
                    elif start_point > k_bound and start_point - eci * self.speed[k_metric] < k_bound:
                        eci -= (start_point - k_bound) / self.speed[k_metric]
                        self.priority[k_metric] = k_bound
                    elif eci == 0:
                        self.priority[k_metric] = self.f_worst[k_metric]
                    else:
                        self.priority[k_metric] = start_point - eci * self.speed[k_metric] 
                        eci = 0
                    feasible_flag = False
        else:
            self.priority = eci * self.speed - self.obj_best1

    def update_eci(self, metric_target: Union[float, dict], max_speed: Optional[Union[float, dict]], f_best: Optional[dict], f_worst: Optional[dict]):
        # calculate eci: estimated cost for improvement over metric_target
        if self.lexico_objectives is None or not self._is_ls:  
            best_obj = metric_target * self._metric_op
            if not self.speed:
                self.speed = max_speed
            self.eci = max(
                self.cost_total - self.cost_best1, self.cost_best1 - self.cost_best2
            )
            if self.obj_best1 > best_obj and self.speed > 0:
                self.eci = max(self.eci, 2 * (self.obj_best1 - best_obj) / self.speed)
        else:  # Optimization lexicographically
            self.eci = max(
                self.cost_total - self.cost_best1, self.cost_best1 - self.cost_best2
            )
            eci_last = 0
            feasible_flag = True
            # TODO: finish 3/31
            for k_metric, k_mode in zip(
                self.lexico_objectives["metrics"], self.lexico_objectives["modes"]
            ):
                if not self.speed[k_metric]:
                    self.speed[k_metric] = max_speed
                k_bound = _get_lexico_bound(k_metric, k_mode)
                if self.obj_best1[k_metric] < k_bound:
                    continue
                elif self.obj_best1[k_metric] < self.f_best[k_metric]:
                    raise ValueError(
                        "Bug exists: Performance of a specific search thread is not possible to be better than global best.")
                else:               
                    if self.speed[k_metric] > 0: # check again in which cases speed[k_metric] <= 0
                        start_point = self.obj_best1[k_metric] if feasible_flag else self.f_worst[k_metric]
                        eci_last + = (start_point - k_bound) / self.speed[k_metric]
                    feasible_flag = False
            self.eci = max(self.eci, 2 * eci_last)

    def _update_speed(self):
        # calculate speed; use 0 for invalid speed temporarily
        if self.lexico_objectives is None and self.obj_best2 > self.obj_best1:
            # discount the speed if there are unfinished trials
            self.speed = (
                (self.obj_best2 - self.obj_best1)
                / self.running
                / (max(self.cost_total - self.cost_best2, self._eps))
            )
            for i in range(0, len(metrics)):
                if i != op_index:
                    self.speed[self.lexico_objectives["metrics"][i]] = 0
        else:
            if self.lexico_objectives is None:
                self.speed = 0
            else:
                for i in range(0, len(metrics)):
                    self.speed[self.lexico_objectives["metrics"][i]] = 0

    def on_trial_complete(self, trial_id: str, result: Optional[Dict] = None, error: bool = False):
        """Update the statistics of the thread."""
        if not self._search_alg:
            return
        if not hasattr(self._search_alg, "_ot_trials") or (not error and trial_id in self._search_alg._ot_trials):
            # optuna doesn't handle error
            if self._is_ls or not self._init_config:
                try:
                    self._search_alg.on_trial_complete(trial_id, result, error)
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
            if self._search_alg.metric in result:
                if self.lexico_objectives is None:
                    obj = result[self._search_alg.metric] * self._metric_op
                else:
                    obj = {}
                    for k, m in zip(self._search_alg.lexico_objectives["metrics"], self._search_alg.lexico_objectives["modes"]):
                        obj[k] = -result[k] if m == "max" else result[k]
                if self.best_result is None or (self.lexico_objectives is None and obj < self.obj_best1) or (self.lexico_objectives is not None and obj == self._search_alg.best_obj):
                    self.cost_best2 = self.cost_best1
                    self.cost_best1 = self.cost_total
                    if self.lexico_objectives is None:
                        self.obj_best2 = obj if np.isinf(self.obj_best1) else self.obj_best1
                    else:
                        self.obj_best2 = obj if np.isinf(
                            self.obj_best1[self.lexico_objectives["metrics"][0]]) else self.obj_best1
                    self.obj_best1 = obj
                    self.cost_best = self.cost_last
                    self.best_result = result
            if self.lexico_objectives is None:
                # TODO: Improve this behavior. When lexico_objectives is provided to CFO,
                # related variables are not callable.
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
            except RuntimeError as e:
                # rs is used in place of optuna sometimes
                if not str(e).endswith("has already finished and can not be updated."):
                    raise e
        new_cost = result.get(self.cost_attr, 1)
        if self.cost_last < new_cost:
            self.cost_last = new_cost
            # self._update_speed()

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