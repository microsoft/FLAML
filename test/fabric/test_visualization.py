import os
import unittest
from unittest.mock import PropertyMock, patch

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import flaml
import flaml.visualization as fviz
from flaml import AutoML


class TestVisualization(unittest.TestCase):
    def setUp(self):
        from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

        autologging = patch.dict(
            AUTOLOGGING_INTEGRATIONS,
            {name: {**options, "disable": True} for name, options in AUTOLOGGING_INTEGRATIONS.items()},
            clear=True,
        )
        autologging.start()
        self.addCleanup(autologging.stop)
        fabric_runtime = patch.dict(os.environ, {"FLAML_FABRIC_RUNTIME": "true"})
        fabric_runtime.start()
        self.addCleanup(fabric_runtime.stop)
        x, y = load_iris(return_X_y=True, as_frame=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7654321)

        aml = AutoML()
        automl_settings = {
            "task": "classification",
            "estimator_list": ["rf"],
            "max_iter": 6,
            "n_jobs": 1,
            "seed": 7654321,
            "verbose": 0,
            "custom_hp": {"rf": {"n_estimators": {"domain": flaml.tune.randint(4, 16), "init_value": 4}}},
        }
        aml.fit(X_train=x_train, y_train=y_train, **automl_settings)
        self.aml = aml

        def _sklearn_tune(config):
            X, y = load_iris(return_X_y=True, as_frame=True)
            train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=7654321)
            rf = RandomForestClassifier(**config, n_jobs=1, random_state=7654321)
            rf.fit(train_x, train_y)
            pred = rf.predict(test_x)
            acc = accuracy_score(test_y, pred)
            return {"accuracy": acc}

        params = {
            "n_estimators": flaml.tune.randint(4, 16),
            "min_samples_leaf": flaml.tune.randint(1, 10),
        }
        result = flaml.tune.run(
            _sklearn_tune,
            params,
            metric="accuracy",
            mode="max",
            num_samples=8,
            use_ray=False,
            mlflow_logging=False,
        )
        self.tune_result = result

    def test_plot_optimization_history(self):
        for result in (self.aml, self.tune_result):
            figure = fviz.plot_optimization_history(result)
            self.assertTrue(figure.data)
            self.assertTrue(any(len(trace.y) > 1 for trace in figure.data))

    def test_plot_feature_importance(self):
        figure = fviz.plot_feature_importance(self.aml)
        self.assertEqual(list(figure.data[0].y), list(self.aml.feature_names_in_))
        np.testing.assert_allclose(figure.data[0].x, self.aml.feature_importances_)
        with patch.object(AutoML, "feature_importances_", new_callable=PropertyMock, return_value=None):
            empty = fviz.plot_feature_importance(self.aml)
        self.assertEqual(empty.layout.title.text, "Feature Importance")
        self.assertTrue(empty.layout.annotations)

    def test_plot_parallel_coordinate(self):
        fviz.plot_parallel_coordinate(self.aml)
        fviz.plot_parallel_coordinate(self.tune_result)

    def test_plot_contour(self):
        fviz.plot_contour(self.aml)
        fviz.plot_contour(self.tune_result)

    def test_plot_edf(self):
        for result in (self.aml, self.tune_result):
            figure = fviz.plot_edf(result)
            self.assertTrue(figure.data)
            for trace in figure.data:
                self.assertTrue(np.all(np.diff(trace.y) >= 0))
                self.assertEqual(max(trace.y), 1.0)

    def test_plot_timeline(self):
        for result in (self.aml, self.tune_result):
            figure = fviz.plot_timeline(result)
            self.assertTrue(figure.data)
            self.assertTrue(all(np.all(np.asarray(trace.x) >= 0) for trace in figure.data))

    def test_plot_slice(self):
        fviz.plot_slice(self.aml)
        fviz.plot_slice(self.tune_result)

    def test_plot_param_importance(self):
        fviz.plot_param_importance(self.aml)
        fviz.plot_param_importance(self.tune_result)
