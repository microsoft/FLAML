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
            "time_budget": 3,
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

        # 2. Constrct a new class:
        # (a) write the hps you want to freeze as hps with constant 'domain';
        # (b) specify the new search space of the other hps accrodingly.
        class MyPartiallyFreezedLargeLGBM(LGBMEstimator):
            @classmethod
            def search_space(cls, **params):
                print(starting_points)
                space = {}
                # get the fixed value from hps from the starting point
                for name, value in starting_points[new_estimator_name].items():
                    space[name] = {"domain": value}

                # specify the search sapce of the hps to want to tune
                hps_to_search = {
                    "n_estimators": {
                        "domain": tune.lograndint(lower=10, upper=32768),
                        "init_value": 32768,
                        "low_cost_init_value": 10,
                    },
                    "num_leaves": {
                        "domain": tune.lograndint(lower=10, upper=3276),
                        "init_value": 3276,
                        "low_cost_init_value": 10,
                    },
                }
                space.update(hps_to_search)
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
