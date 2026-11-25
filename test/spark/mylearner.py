from flaml import tune
from flaml.automl.model import LGBMEstimator


class MyLargeLGBM(LGBMEstimator):
    @classmethod
    def search_space(cls, **params):
        return {
            "n_estimators": {
                "domain": tune.lograndint(lower=4, upper=32768),
                "init_value": 32768,
                "low_cost_init_value": 4,
            },
            "num_leaves": {
                "domain": tune.lograndint(lower=4, upper=32768),
                "init_value": 32768,
                "low_cost_init_value": 4,
            },
        }


class MinEstimatorsLGBM(LGBMEstimator):
    """LGBM estimator that requires n_estimators >= 16 via config_constraints.

    The low-cost initial n_estimators (4) violates the constraint.
    """

    @classmethod
    def config_constraints(cls):
        return [(lambda config: config["n_estimators"], ">=", 16)]
