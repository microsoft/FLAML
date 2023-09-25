try:
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
except ImportError:
    pass

try:
    from pandas import DataFrame
except ImportError:
    DataFrame = None

from flaml import tune
from flaml.automl.model import (
    suppress_stdout_stderr,
    SKLearnEstimator,
    logger,
    LGBMEstimator,
)
from flaml.automl.task import Task


class HistGradientBoostingEstimator(SKLearnEstimator, LGBMEstimator):
    """The class for tuning Histogram Gradient Boosting."""

    ITER_HP = "n_estimators"
    HAS_CALLBACK = False
    DEFAULT_ITER = 100

    @classmethod
    def search_space(cls, data_size, task, **params):
        upper = max(5, min(32768, int(data_size[0])))  # upper must be larger than lower
        return {
            "n_estimators": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "init_value": 4,
                "low_cost_init_value": 4,
            },
            "max_leaf_nodes": {
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
            "log_max_bin": {  # log transformed with base 2
                "domain": tune.lograndint(lower=3, upper=11),
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
        if "random_state" not in params:
            params["random_state"] = 24092023
        if "n_jobs" in params:
            params.pop("n_jobs")
        return params

    def __init__(
        self,
        task: Task,
        **params,
    ):
        super().__init__(task, **params)
        self.params["verbose"] = 0

        if self._task.is_classification():
            self.estimator_class = HistGradientBoostingClassifier
        else:
            self.estimator_class = HistGradientBoostingRegressor

    def _preprocess(self, X):
        if isinstance(X, DataFrame):
            cat_columns = X.select_dtypes(include=["category"]).columns
            if not cat_columns.empty:
                X = X.copy()
                X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
        elif isinstance(X, np.ndarray) and X.dtype.kind not in "buif":
            # numpy array is not of numeric dtype
            X = DataFrame(X)
            for col in X.columns:
                if isinstance(X[col][0], str):
                    X[col] = np.clip(
                        X[col].astype("category").cat.codes.replace(-1, np.nan),
                        0,
                        self.params["max_bins"] #  [0, max_bins - 1]
                    )
            X = X.to_numpy()
        return X

    def fit(self, X_train, y_train, budget=None, free_mem_ratio=0, **kwargs):
        if isinstance(X_train, DataFrame):
            cat_features = list(X_train.select_dtypes(include="category").columns)
        else:
            cat_features = []
        self.params["categorical_features"] = cat_features
        return super().fit(X_train, y_train, budget, free_mem_ratio, **kwargs)
