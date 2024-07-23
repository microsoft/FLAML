try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
except ImportError:
    pass

from flaml import tune
from flaml.automl.model import SKLearnEstimator
from flaml.automl.task import Task


class HistGradientBoostingEstimator(SKLearnEstimator):
    """The class for tuning Histogram Gradient Boosting."""

    ITER_HP = "max_iter"
    HAS_CALLBACK = False
    DEFAULT_ITER = 100

    @classmethod
    def search_space(cls, data_size: int, task, **params) -> dict:
        upper = max(5, min(32768, int(data_size[0])))  # upper must be larger than lower
        return {
            "n_estimators": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "init_value": 4,
                "low_cost_init_value": 4,
            },
            "max_leaves": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "init_value": 4,
                "low_cost_init_value": 4,
            },
            "min_samples_leaf": {
                "domain": tune.lograndint(lower=2, upper=2**7 + 1),
                "init_value": 20,
            },
            "learning_rate": {
                "domain": tune.loguniform(lower=1 / 1024, upper=1.0),
                "init_value": 0.1,
            },
            "log_max_bin": {  # log transformed with base 2, <= 256
                "domain": tune.lograndint(lower=3, upper=9),
                "init_value": 8,
            },
            "l2_regularization": {
                "domain": tune.loguniform(lower=1 / 1024, upper=1024),
                "init_value": 1.0,
            },
        }

    def config2params(self, config: dict) -> dict:
        params = super().config2params(config)
        if "log_max_bin" in params:
            params["max_bins"] = (1 << params.pop("log_max_bin")) - 1
        if "max_leaves" in params:
            params["max_leaf_nodes"] = params.get("max_leaf_nodes", params.pop("max_leaves"))
        if "n_estimators" in params:
            params["max_iter"] = params.get("max_iter", params.pop("n_estimators"))
        if "random_state" not in params:
            params["random_state"] = 24092023
        if "n_jobs" in params:
            params.pop("n_jobs")
        return params

    def __init__(
        self,
        task: Task,
        **config,
    ):
        super().__init__(task, **config)
        self.params["verbose"] = 0

        if self._task.is_classification():
            self.estimator_class = HistGradientBoostingClassifier
        else:
            self.estimator_class = HistGradientBoostingRegressor
