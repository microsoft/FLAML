import unittest
import warnings

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import flaml
import flaml.visualization as fviz
from flaml import AutoML

warnings.filterwarnings("ignore")


class TestVisualization(unittest.TestCase):
    def setUp(self):
        x, y = load_iris(return_X_y=True, as_frame=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7654321)

        aml = AutoML()
        automl_settings = {"time_budget": 10, "task": "classification"}
        aml.fit(X_train=x_train, y_train=y_train, **automl_settings)
        self.aml = aml

        def _sklearn_tune(config):
            X, y = load_iris(return_X_y=True, as_frame=True)
            train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)
            rf = RandomForestClassifier(**config)
            rf.fit(train_x, train_y)
            pred = rf.predict(test_x)
            acc = accuracy_score(test_y, pred)
            return {"accuracy": acc}

        params = {
            "n_estimators": flaml.tune.randint(100, 1000),
            "min_samples_leaf": flaml.tune.randint(1, 10),
        }
        result = flaml.tune.run(
            _sklearn_tune,
            params,
            metric="accuracy",
            mode="max",
            num_samples=50,
        )
        self.tune_result = result

    def test_plot_optimization_history(self):
        fviz.plot_optimization_history(self.aml)
        fviz.plot_optimization_history(self.tune_result)

    def test_plot_feature_importance(self):
        fviz.plot_feature_importance(self.aml)

    def test_plot_parallel_coordinate(self):
        fviz.plot_parallel_coordinate(self.aml)
        fviz.plot_parallel_coordinate(self.tune_result)

    def test_plot_contour(self):
        fviz.plot_contour(self.aml)
        fviz.plot_contour(self.tune_result)

    def test_plot_edf(self):
        fviz.plot_edf(self.aml)
        fviz.plot_edf(self.tune_result)

    def test_plot_timeline(self):
        fviz.plot_timeline(self.aml)
        fviz.plot_timeline(self.tune_result)

    def test_plot_slice(self):
        fviz.plot_slice(self.aml)
        fviz.plot_slice(self.tune_result)

    def test_plot_param_importance(self):
        fviz.plot_param_importance(self.aml)
        fviz.plot_param_importance(self.tune_result)
