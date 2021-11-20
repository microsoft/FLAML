import unittest
import numpy as np
from sklearn.datasets import load_iris
from flaml import AutoML
from flaml.model import LGBMEstimator
from flaml import tune


class TestWarmStart(unittest.TestCase):
    def test_fit_w_freezinghp_starting_point(self, as_frame=True):
        automl_experiment = AutoML()
        automl_settings = {
            "time_budget": 1,
            "metric": "accuracy",
            "task": "classification",
            "estimator_list": ["lgbm"],
            "log_file_name": "test/iris.log",
            "log_training_metric": True,
            "n_jobs": 1,
            "model_history": True,
        }
        X_train, y_train = load_iris(return_X_y=True, as_frame=as_frame)
        if as_frame:
            # test drop column
            X_train.columns = range(X_train.shape[1])
            X_train[X_train.shape[1]] = np.zeros(len(y_train))
        automl_experiment.fit(X_train=X_train, y_train=y_train, **automl_settings)
        automl_val_accuracy = 1.0 - automl_experiment.best_loss
        print("Best ML leaner:", automl_experiment.best_estimator)
        print("Best hyperparmeter config:", automl_experiment.best_config)
        print("Best accuracy on validation data: {0:.4g}".format(automl_val_accuracy))
        print(
            "Training duration of best run: {0:.4g} s".format(
                automl_experiment.best_config_train_time
            )
        )
        # 1. Get starting points from previous experiments.
        starting_points = automl_experiment.best_config_per_estimator
        print("starting_points", starting_points)
        print("loss of the starting_points", automl_experiment.best_loss_per_estimator)

        hps_to_freeze = ["colsample_bytree", "reg_alpha", "reg_lambda", "log_max_bin"]
        # 2. Constrct a new class:
        # a. write the hps you want to freeze as hps with constant 'domain';
        # b. specify the new search space of the other hps accrodingly.

        class MyPartiallyFreezedLargeLGBM(LGBMEstimator):
            @classmethod
            def search_space(cls, **params):
                org_space = LGBMEstimator.search_space(**params)
                # (1) Get the hps in the original search space
                space = org_space.copy()
                # (2) Set up the fixed value from hps from the starting point
                for hp_name in hps_to_freeze:
                    # if an hp is specifed to be freezed, use tine value provided in the starting_points
                    # otherwise use the setting from the original search space
                    if hp_name in starting_points[new_estimator_name]:
                        space[hp_name] = {
                            "domain": starting_points[new_estimator_name][hp_name]
                        }
                # (3.1) Configure the search space for hps that are in the original search space
                #  but you want to change something, for example the range.
                revised_hps_to_search = {
                    "n_estimators": {
                        "domain": tune.lograndint(lower=10, upper=32768),
                        "init_value": starting_points[new_estimator_name].get(
                            "n_estimators"
                        )
                        or org_space["n_estimators"].get("init_value", 10),
                        "low_cost_init_value": org_space["n_estimators"].get(
                            "low_cost_init_value", 10
                        ),
                    },
                    "num_leaves": {
                        "domain": tune.lograndint(lower=10, upper=3276),
                        "init_value": starting_points[new_estimator_name].get(
                            "num_leaves"
                        )
                        or org_space["num_leaves"].get("init_value", 10),
                        "low_cost_init_value": org_space["num_leaves"].get(
                            "low_cost_init_value", 10
                        ),
                    },
                    # (3.2) Add a new hp which is not in the original search space
                    "subsample": {
                        "domain": tune.uniform(lower=0.1, upper=1.0),
                        "init_value": 0.1,
                    },
                }
                space.update(revised_hps_to_search)
                return space

        new_estimator_name = "large_lgbm"
        new_automl_experiment = AutoML()
        new_automl_experiment.add_learner(
            learner_name=new_estimator_name, learner_class=MyPartiallyFreezedLargeLGBM
        )
        starting_points[new_estimator_name] = starting_points["lgbm"]
        automl_settings_resume = {
            "time_budget": 3,
            "metric": "accuracy",
            "task": "classification",
            "estimator_list": ["large_lgbm"],
            "log_file_name": "test/iris_resume.log",
            "log_training_metric": True,
            "n_jobs": 1,
            "model_history": True,
            "log_type": "all",
            "starting_points": starting_points,
        }

        new_automl_experiment.fit(
            X_train=X_train, y_train=y_train, **automl_settings_resume
        )

        new_automl_val_accuracy = 1.0 - new_automl_experiment.best_loss
        print("Best ML leaner:", new_automl_experiment.best_estimator)
        print("Best hyperparmeter config:", new_automl_experiment.best_config)
        print(
            "Best accuracy on validation data: {0:.4g}".format(new_automl_val_accuracy)
        )
        print(
            "Training duration of best run: {0:.4g} s".format(
                new_automl_experiment.best_config_train_time
            )
        )


if __name__ == "__main__":
    unittest.main()
