from flaml.tune.space import unflatten_hierarchical
from flaml import AutoML
from sklearn.datasets import fetch_california_housing
import os
import unittest
import logging
import tempfile
import io


class TestLogging(unittest.TestCase):
    def test_logging_level(self):

        from flaml import logger, logger_formatter

        with tempfile.TemporaryDirectory() as d:

            training_log = os.path.join(d, "training.log")

            # Configure logging for the FLAML logger
            # and add a handler that outputs to a buffer.
            logger.setLevel(logging.INFO)
            buf = io.StringIO()
            ch = logging.StreamHandler(buf)
            ch.setFormatter(logger_formatter)
            logger.addHandler(ch)

            # Run a simple job.
            automl = AutoML()
            automl_settings = {
                "time_budget": 1,
                "metric": "rmse",
                "task": "regression",
                "log_file_name": training_log,
                "log_training_metric": True,
                "n_jobs": 1,
                "model_history": True,
                "keep_search_state": True,
                "learner_selector": "roundrobin",
            }
            X_train, y_train = fetch_california_housing(return_X_y=True)
            n = len(y_train) >> 1
            print(automl.model, automl.classes_, automl.predict(X_train))
            automl.fit(
                X_train=X_train[:n],
                y_train=y_train[:n],
                X_val=X_train[n:],
                y_val=y_train[n:],
                **automl_settings
            )
            logger.info(automl.search_space)
            logger.info(automl.low_cost_partial_config)
            logger.info(automl.points_to_evaluate)
            logger.info(automl.cat_hp_cost)
            import optuna as ot

            study = ot.create_study()
            from flaml.tune.space import define_by_run_func, add_cost_to_space

            sample = define_by_run_func(study.ask(), automl.search_space)
            logger.info(sample)
            logger.info(unflatten_hierarchical(sample, automl.search_space))
            add_cost_to_space(
                automl.search_space, automl.low_cost_partial_config, automl.cat_hp_cost
            )
            logger.info(automl.search_space["ml"].categories)
            if automl.best_config:
                config = automl.best_config.copy()
                config["learner"] = automl.best_estimator
                automl.trainable({"ml": config})
            from flaml import tune, BlendSearch
            from flaml.automl import size
            from functools import partial

            low_cost_partial_config = automl.low_cost_partial_config
            search_alg = BlendSearch(
                metric="val_loss",
                mode="min",
                space=automl.search_space,
                low_cost_partial_config=low_cost_partial_config,
                points_to_evaluate=automl.points_to_evaluate,
                cat_hp_cost=automl.cat_hp_cost,
                prune_attr=automl.prune_attr,
                min_resource=automl.min_resource,
                max_resource=automl.max_resource,
                config_constraints=[
                    (partial(size, automl._state), "<=", automl._mem_thres)
                ],
                metric_constraints=automl.metric_constraints,
            )
            analysis = tune.run(
                automl.trainable,
                search_alg=search_alg,  # verbose=2,
                time_budget_s=1,
                num_samples=-1,
            )
            print(min(trial.last_result["val_loss"] for trial in analysis.trials))
            config = analysis.trials[-1].last_result["config"]["ml"]
            automl._state._train_with_config(config["learner"], config)
            for _ in range(3):
                print(
                    search_alg._ls.complete_config(
                        low_cost_partial_config,
                        search_alg._ls_bound_min,
                        search_alg._ls_bound_max,
                    )
                )
            # Check if the log buffer is populated.
            self.assertTrue(len(buf.getvalue()) > 0)

        import pickle

        with open("automl.pkl", "wb") as f:
            pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
        print(automl.__version__)
