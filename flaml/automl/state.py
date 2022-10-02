import copy
import inspect
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from flaml import tune
from flaml.ml import compute_estimator, train_estimator
from flaml.time_series.ts_data import TimeSeriesDataset

logger = logging.getLogger(__name__)


# parse the strange special spec that built-in estimators use into a flaml.tune spec
def get_tune_domain(space):
    if isinstance(space, dict):
        out = {}
        for k, v in space.items():
            if isinstance(v, dict):
                if isinstance(v["domain"], list):
                    out[k] = tune.choice([get_tune_domain(vv) for vv in v["domain"]])
                else:
                    out[k] = get_tune_domain(v["domain"])
            elif isinstance(v, str):
                out[k] = v
        return out
    else:
        return space


def get_initial_value(space, field_name):
    out = {}
    if isinstance(space, dict):
        for k, v in space.items():
            if isinstance(v, dict):
                if field_name in v:
                    # if a nested config branch has no init_value, ignore it
                    if isinstance(v[field_name], dict):
                        tmp = get_initial_value(v[field_name], field_name)
                        out[k] = tmp
                    else:
                        out[k] = v[field_name]
            else:
                out[k] = v
    else:
        raise ValueError
    return out


class SearchState:
    def __init__(
        self,
        learner_class,
        data_size,
        data,
        task,
        starting_point=None,
        period=None,
        custom_hp=None,
        max_iter=None,
    ):
        self.init_eci = learner_class.cost_relative2lgbm()
        self._search_space_domain = {}
        self.init_config = {}
        self.low_cost_partial_config = {}
        self.cat_hp_cost = {}
        self.data_size = data_size
        self.ls_ever_converged = False
        self.learner_class = learner_class
        if task.is_ts_forecast():
            search_space = learner_class.search_space(
                data=data, task=task, pred_horizon=period
            )
        else:
            search_space = learner_class.search_space(data_size=data_size, task=task)

        if custom_hp is not None:
            search_space.update(custom_hp)

        if (
            isinstance(starting_point, dict)
            and max_iter
            > 1  # If the number of starting point is larger than max iter, avoid the checking
            and not self.valid_starting_point(starting_point, search_space)
        ):
            logger.warning(
                "Starting point {} removed because it is outside of the search space".format(
                    starting_point
                )
            )
            starting_point = None
        elif isinstance(starting_point, list) and max_iter > len(
            starting_point
        ):  # If the number of starting point is larger than max iter, avoid the checking
            starting_point_len = len(starting_point)
            starting_point = [
                x for x in starting_point if self.valid_starting_point(x, search_space)
            ]
            if starting_point_len > len(starting_point):
                logger.warning(
                    "Starting points outside of the search space are removed. "
                    f"Remaining starting points for {learner_class}: {starting_point}"
                )
            starting_point = starting_point or None

        use_new_init = True
        for name, space in search_space.items():
            # assert (
            #     "domain" in space
            # ), f"{name}'s domain is missing in the search space spec {space}"
            # if space["domain"] is None:
            #     # don't search this hp
            #     continue
            # self._search_space_domain[name] = space["domain"]

            # if "low_cost_init_value" in space:
            #     self.low_cost_partial_config[name] = space["low_cost_init_value"]
            if "cat_hp_cost" in space:
                self.cat_hp_cost[name] = space["cat_hp_cost"]
            # if a starting point is provided, set the init config to be
            # the starting point provided
            if (
                isinstance(starting_point, dict)
                and starting_point.get(name) is not None
            ):
                self.init_config[name] = starting_point[name]
            elif (
                not isinstance(starting_point, list)
                and "init_value" in space
                and not use_new_init
                # and self.valid_starting_point_one_dim( # shouldn't tune be doing these checks?
                #     space["init_value"], space["domain"]
                # )
            ):  # If starting point is list, no need to check the validity of self.init_config w.r.t search space
                self.init_config[name] = space[
                    "init_value"
                ]  # If starting_point is list, no need to assign value to self.init_config here

        if isinstance(starting_point, list):
            self.init_config = starting_point

        self._hp_names = list(self._search_space_domain.keys())
        self.search_alg = None
        self.best_config = None
        self.best_result = None
        self.best_loss = self.best_loss_old = np.inf
        self.total_time_used = 0
        self.total_iter = 0
        self.base_eci = None
        self.time_best_found = self.time_best_found_old = 0
        self.time2eval_best = 0
        self.time2eval_best_old = 0
        self.trained_estimator = None
        self.sample_size = None
        self.trial_time = 0

        new_space = get_tune_domain(search_space)
        new_init = get_initial_value(search_space, "init_value")
        # integrate any starting_point values
        if isinstance(self.init_config, dict):
            new_init.update(self.init_config)

        self.init_config = new_init
        self._search_space_domain = new_space
        # TODO:
        # self.low_cost_partial_config = get_initial_value(
        #     search_space, "low_cost_init_value"
        # )

    def compare_dicts(d1, d2):
        for key in sorted(list(set(*d1.keys(), *d2.keys()))):
            print(key, d1.get(key), d2.get(key))

    def update(self, result, time_used):
        if result:
            config = result["config"]
            if config and "FLAML_sample_size" in config:
                self.sample_size = config["FLAML_sample_size"]
            else:
                self.sample_size = self.data_size[0]
            obj = result["val_loss"]
            metric_for_logging = result["metric_for_logging"]
            time2eval = result["time_total_s"]
            trained_estimator = result["trained_estimator"]
            del result["trained_estimator"]  # free up RAM
            n_iter = (
                trained_estimator
                and hasattr(trained_estimator, "ITER_HP")
                and trained_estimator.params[trained_estimator.ITER_HP]
            )
            if n_iter:
                config[trained_estimator.ITER_HP] = n_iter
        else:
            obj, time2eval, trained_estimator = np.inf, 0.0, None
            metric_for_logging = config = None
        self.trial_time = time2eval
        self.total_time_used += time_used
        self.total_iter += 1

        if self.base_eci is None:
            self.base_eci = time_used
        if (obj is not None) and (obj < self.best_loss):
            self.best_loss_old = self.best_loss if self.best_loss < np.inf else 2 * obj
            self.best_loss = obj
            self.best_result = result
            self.time_best_found_old = self.time_best_found
            self.time_best_found = self.total_time_used
            self.iter_best_found = self.total_iter
            self.best_config = config
            self.best_config_sample_size = self.sample_size
            self.best_config_train_time = time_used
            if time2eval:
                self.time2eval_best_old = self.time2eval_best
                self.time2eval_best = time2eval
            if (
                self.trained_estimator
                and trained_estimator
                and self.trained_estimator != trained_estimator
            ):
                self.trained_estimator.cleanup()
            if trained_estimator:
                self.trained_estimator = trained_estimator
        elif trained_estimator:
            trained_estimator.cleanup()
        self.metric_for_logging = metric_for_logging
        self.val_loss, self.config = obj, config

    def get_hist_config_sig(self, sample_size, config):
        config_values = tuple([config[k] for k in self._hp_names])
        config_sig = str(sample_size) + "_" + str(config_values)
        return config_sig

    def est_retrain_time(self, retrain_sample_size):
        assert (
            self.best_config_sample_size is not None
        ), "need to first get best_config_sample_size"
        return self.time2eval_best * retrain_sample_size / self.best_config_sample_size

    @property
    def search_space(self):
        return self._search_space_domain

    @property
    def estimated_cost4improvement(self):
        return max(
            self.time_best_found - self.time_best_found_old,
            self.total_time_used - self.time_best_found,
        )

    def valid_starting_point_one_dim(self, value_one_dim, domain_one_dim):
        from ..tune.space import sample

        """
            For each hp in the starting point, check the following 3 conditions:
            (1) If the type of the starting point does not match the required type in search space, return false
            (2) If the starting point is not in the required search space, return false
            (3) If the search space is a value instead of domain, and the value is not equal to the starting point
            Notice (2) include the case starting point not in user specified search space custom_hp
        """
        if isinstance(domain_one_dim, sample.Domain):
            renamed_type = list(
                inspect.signature(domain_one_dim.is_valid).parameters.values()
            )[0].annotation
            type_match = (
                renamed_type == Any
                or isinstance(value_one_dim, renamed_type)
                or isinstance(value_one_dim, int)
                and renamed_type is float
            )
            if not (type_match and domain_one_dim.is_valid(value_one_dim)):
                return False
        elif value_one_dim != domain_one_dim:
            return False
        return True

    def valid_starting_point(self, starting_point, search_space):
        return all(
            self.valid_starting_point_one_dim(value, search_space[name].get("domain"))
            for name, value in starting_point.items()
            if name != "FLAML_sample_size"
        )


class AutoMLState:
    def __init__(self):
        self.learner_classes = {}

    def _prepare_sample_train_data(self, sample_size):
        # we take the tail, rather than the head, for compatibility with time series
        sampled_weight = groups = None
        if sample_size <= self.data_size[0]:
            if isinstance(self.X_train, pd.DataFrame):
                sampled_X_train = self.X_train.iloc[:sample_size]
                sampled_y_train = self.y_train[:sample_size]
            elif isinstance(self.X_train, TimeSeriesDataset):
                sampled_X_train = copy.copy(self.X_train)
                sampled_X_train.train_data = self.X_train.train_data.iloc[-sample_size:]
                sampled_y_train = None
            else:
                sampled_X_train = self.X_train[:sample_size]
                sampled_y_train = self.y_train[:sample_size]
            weight = self.fit_kwargs.get(
                "sample_weight"
            )  # NOTE: _prepare_sample_train_data is before kwargs is updated to fit_kwargs_by_estimator
            if weight is not None:
                sampled_weight = weight[:sample_size]
            if self.groups is not None:
                groups = self.groups[:sample_size]
        else:
            sampled_X_train = self.X_train_all
            sampled_y_train = self.y_train_all
            if (
                "sample_weight" in self.fit_kwargs
            ):  # NOTE: _prepare_sample_train_data is before kwargs is updated to fit_kwargs_by_estimator
                sampled_weight = self.sample_weight_all
            if self.groups is not None:
                groups = self.groups_all
        return sampled_X_train, sampled_y_train, sampled_weight, groups

    @staticmethod
    def _compute_with_config_base(config_w_resource, state, estimator):
        if "FLAML_sample_size" in config_w_resource:
            sample_size = int(config_w_resource["FLAML_sample_size"])
        else:
            sample_size = state.data_size[0]

        this_estimator_kwargs = state.fit_kwargs_by_estimator.get(
            estimator
        ).copy()  # NOTE: _compute_with_config_base is after kwargs is updated to fit_kwargs_by_estimator
        (
            sampled_X_train,
            sampled_y_train,
            sampled_weight,
            groups,
        ) = state._prepare_sample_train_data(sample_size)
        if sampled_weight is not None:
            weight = this_estimator_kwargs["sample_weight"]
            this_estimator_kwargs["sample_weight"] = sampled_weight
        if groups is not None:
            this_estimator_kwargs["groups"] = groups
        config = config_w_resource.copy()
        if "FLAML_sample_size" in config:
            del config["FLAML_sample_size"]
        budget = (
            None
            if state.time_budget is None
            else state.time_budget - state.time_from_start
            if sample_size == state.data_size[0]
            else (state.time_budget - state.time_from_start)
            / 2
            * sample_size
            / state.data_size[0]
        )

        (
            trained_estimator,
            val_loss,
            metric_for_logging,
            _,
            pred_time,
        ) = compute_estimator(
            sampled_X_train,
            sampled_y_train,
            state.X_val,
            state.y_val,
            state.weight_val,
            state.groups_val,
            state.train_time_limit
            if budget is None
            else min(budget, state.train_time_limit),
            state.kf,
            config,
            state.task,
            estimator,
            state.eval_method,
            state.metric,
            state.best_loss,
            state.n_jobs,
            state.learner_classes.get(estimator),
            state.cv_score_agg_func,
            state.log_training_metric,
            this_estimator_kwargs,
        )
        if state.retrain_final and not state.model_history:
            trained_estimator.cleanup()

        result = {
            "pred_time": pred_time,
            "wall_clock_time": time.time() - state._start_time_flag,
            "metric_for_logging": metric_for_logging,
            "val_loss": val_loss,
            "trained_estimator": trained_estimator,
        }
        if sampled_weight is not None:
            this_estimator_kwargs["sample_weight"] = weight
        tune.report(**result)
        return result

    def sanitize(self, config: dict) -> dict:
        """Make a config ready for passing to estimator."""
        config = config.get("ml", config).copy()
        if "FLAML_sample_size" in config:
            del config["FLAML_sample_size"]
        if "learner" in config:
            del config["learner"]
        return config

    def _train_with_config(
        self,
        estimator,
        config_w_resource,
        sample_size=None,
    ):
        if not sample_size:
            sample_size = config_w_resource.get(
                "FLAML_sample_size", len(self.y_train_all)
            )
        config = self.sanitize(config_w_resource)

        this_estimator_kwargs = self.fit_kwargs_by_estimator.get(
            estimator
        ).copy()  # NOTE: _train_with_config is after kwargs is updated to fit_kwargs_by_estimator

        (
            sampled_X_train,
            sampled_y_train,
            sampled_weight,
            groups,
        ) = self.task._prepare_sample_train_data(self, sample_size)

        if sampled_weight is not None:
            weight = this_estimator_kwargs[
                "sample_weight"
            ]  # NOTE: _train_with_config is after kwargs is updated to fit_kwargs_by_estimator
            this_estimator_kwargs[
                "sample_weight"
            ] = sampled_weight  # NOTE: _train_with_config is after kwargs is updated to fit_kwargs_by_estimator
        if groups is not None:
            this_estimator_kwargs[
                "groups"
            ] = groups  # NOTE: _train_with_config is after kwargs is updated to fit_kwargs_by_estimator

        budget = (
            None
            if self.time_budget is None
            else self.time_budget - self.time_from_start
        )

        estimator, train_time = train_estimator(
            X_train=sampled_X_train,
            y_train=sampled_y_train,
            config_dic=config,
            task=self.task,
            estimator_name=estimator,
            n_jobs=self.n_jobs,
            estimator_class=self.learner_classes.get(estimator),
            budget=budget,
            fit_kwargs=this_estimator_kwargs,  # NOTE: _train_with_config is after kwargs is updated to fit_kwargs_by_estimator
            eval_metric=self.metric if hasattr(self, "metric") else "train_time",
        )

        if sampled_weight is not None:
            this_estimator_kwargs[
                "sample_weight"
            ] = weight  # NOTE: _train_with_config is after kwargs is updated to fit_kwargs_by_estimator

        return estimator, train_time
