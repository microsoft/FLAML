import unittest

import numpy as np
import scipy.sparse
from sklearn.datasets import load_boston, load_iris, load_wine

from flaml import AutoML
from flaml.data import get_output_from_log

from flaml.model import BaseEstimator
from flaml.space import ConfigSearchInfo
from rgf.sklearn import RGFClassifier, RGFRegressor


class MyRegularizedGreedyForest(BaseEstimator):

    # search space
    params_configsearch_info = {
        'max_leaf': ConfigSearchInfo(name = 'max_leaf',
         type = int, lower = 4, init = 4, upper = 10000),
        'n_iter': ConfigSearchInfo(name = 'n_iter', type = int, lower = 1,
         init = 1, upper = 32768),
        'n_tree_search': ConfigSearchInfo(name = 'n_tree_search', type = int,
         lower = 1, init = 1, upper = 32768),
        'opt_interval': ConfigSearchInfo(name = 'opt_interval', type = int,
         lower = 1, init = 100, upper = 10000),
        'learning_rate': ConfigSearchInfo(name = 'learning_rate', type = float,
         lower = 0.01, init = 1.0, upper = 20.0),
        'min_samples_leaf': ConfigSearchInfo(name = 'min_samples_leaf',
         type = int, lower = 1, init = 20, upper = 20)
    }
    
    def __init__(self, objective_name = 'binary:logistic', n_jobs = 1,
     max_leaf = 1000, n_iter = 1, n_tree_search = 1, opt_interval = 1,
      learning_rate = 1.0, min_samples_leaf = 1):

        self.objective_name = objective_name

        if 'regression' in objective_name:
            self.estimator_class = RGFRegressor
        else:
            self.estimator_class = RGFClassifier

        # round integer hyperparameters
        self.params = {
            'max_leaf': int(round(max_leaf)),
            'n_iter': int(round(n_iter)),
            'n_tree_search': int(round(n_tree_search)),
            'opt_interval': int(round(opt_interval)),
            'learning_rate': learning_rate,
            'min_samples_leaf':int(round(min_samples_leaf)),
            "n_jobs": n_jobs,
        }            


def custom_metric(X_test, y_test, estimator, labels, X_train, y_train):
    from sklearn.metrics import log_loss
    y_pred = estimator.predict_proba(X_test)
    test_loss = log_loss(y_test, y_pred, labels=labels)
    y_pred = estimator.predict_proba(X_train)
    train_loss = log_loss(y_train, y_pred, labels=labels)
    alpha = 0.5
    return test_loss * (1 + alpha) - alpha * train_loss, [test_loss, train_loss]


class TestAutoML(unittest.TestCase):

    def test_custom_learner(self):
        automl = AutoML()
        automl.add_learner(learner_name = 'RGF',
            learner_class = MyRegularizedGreedyForest)            
        X_train, y_train = load_wine(return_X_y=True)
        settings = {
            "time_budget": 10, # total running time in seconds
            "estimator_list": ['RGF', 'lgbm', 'rf', 'xgboost'], 
            "task": 'classification', # task type    
            "sample": True, # whether to subsample training data
            "log_file_name": "test/wine.log",
            "log_training_metric": True, # whether to log training metric
        }

        '''The main flaml automl API'''
        automl.fit(X_train = X_train, y_train = y_train, **settings)

    def test_dataframe(self):
        self.test_classification(True)

    def test_custom_metric(self):

        automl_experiment = AutoML()
        automl_settings = {
            "time_budget": 10,
            'eval_method': 'holdout',
            "metric": custom_metric,
            "task": 'classification',
            "log_file_name": "test/iris_custom.log",
            "log_training_metric": True,
            'log_type': 'all',
            "model_history": True
        }
        X_train, y_train = load_iris(return_X_y=True)
        automl_experiment.fit(X_train=X_train, y_train=y_train,
                              **automl_settings)
        print(automl_experiment.classes_)
        print(automl_experiment.predict_proba(X_train))
        print(automl_experiment.model)
        print(automl_experiment.config_history)
        print(automl_experiment.model_history)
        print(automl_experiment.best_iteration)
        print(automl_experiment.best_estimator)
        automl_experiment = AutoML()
        estimator = automl_experiment.get_estimator_from_log(
            automl_settings["log_file_name"], record_id=0,
            objective='multi')
        print(estimator)
        time_history, best_valid_loss_history, valid_loss_history, \
            config_history, train_loss_history = get_output_from_log(
                filename=automl_settings['log_file_name'], time_budget=6)
        print(train_loss_history)

    def test_classification(self, as_frame=False):

        automl_experiment = AutoML()
        automl_settings = {
            "time_budget": 4,
            "metric": 'accuracy',
            "task": 'classification',
            "log_file_name": "test/iris.log",
            "log_training_metric": True,
            "model_history": True
        }
        X_train, y_train = load_iris(return_X_y=True, as_frame=as_frame)
        automl_experiment.fit(X_train=X_train, y_train=y_train,
                              **automl_settings)
        print(automl_experiment.classes_)
        print(automl_experiment.predict_proba(X_train)[:5])
        print(automl_experiment.model)
        print(automl_experiment.config_history)
        print(automl_experiment.model_history)
        print(automl_experiment.best_iteration)
        print(automl_experiment.best_estimator)
        del automl_settings["metric"]
        del automl_settings["model_history"]
        del automl_settings["log_training_metric"]
        automl_experiment = AutoML()
        duration = automl_experiment.retrain_from_log(
            log_file_name=automl_settings["log_file_name"],
            X_train=X_train, y_train=y_train,
            train_full=True, record_id=0)
        print(duration)
        print(automl_experiment.model)
        print(automl_experiment.predict_proba(X_train)[:5])

    def test_regression(self):

        automl_experiment = AutoML()
        automl_settings = {
            "time_budget": 2,
            "metric": 'mse',
            "task": 'regression',
            "log_file_name": "test/boston.log",
            "log_training_metric": True,
            "model_history": True
        }
        X_train, y_train = load_boston(return_X_y=True)
        n = int(len(y_train)*9//10)
        automl_experiment.fit(X_train=X_train[:n], y_train=y_train[:n],
                              X_val=X_train[n:], y_val=y_train[n:],
                              **automl_settings)
        assert automl_experiment.eval_method == 'holdout'
        print(automl_experiment.predict(X_train))
        print(automl_experiment.model)
        print(automl_experiment.config_history)
        print(automl_experiment.model_history)
        print(automl_experiment.best_iteration)
        print(automl_experiment.best_estimator)
        print(get_output_from_log(automl_settings["log_file_name"], 1))

    def test_sparse_matrix_classification(self):

        automl_experiment = AutoML()
        automl_settings = {
            "time_budget": 2,
            "metric": 'auto',
            "task": 'classification',
            "log_file_name": "test/sparse_classification.log",
            "split_type": "uniform",
            "model_history": True
        }
        X_train = scipy.sparse.random(1554, 21, dtype=int)
        y_train = np.random.randint(3, size=1554)
        automl_experiment.fit(X_train=X_train, y_train=y_train,
                              **automl_settings)
        print(automl_experiment.classes_)
        print(automl_experiment.predict_proba(X_train))
        print(automl_experiment.model)
        print(automl_experiment.config_history)
        print(automl_experiment.model_history)
        print(automl_experiment.best_iteration)
        print(automl_experiment.best_estimator)

    def test_sparse_matrix_regression(self):

        automl_experiment = AutoML()
        automl_settings = {
            "time_budget": 2,
            "metric": 'mae',
            "task": 'regression',
            "log_file_name": "test/sparse_regression.log",
            "model_history": True
        }
        X_train = scipy.sparse.random(300, 900, density=0.0001)
        y_train = np.random.uniform(size=300)
        X_val = scipy.sparse.random(100, 900, density=0.0001)
        y_val = np.random.uniform(size=100)
        automl_experiment.fit(X_train=X_train, y_train=y_train,
                              X_val=X_val, y_val=y_val,
                              **automl_settings)
        assert automl_experiment.X_val.shape == X_val.shape
        print(automl_experiment.predict(X_train))
        print(automl_experiment.model)
        print(automl_experiment.config_history)
        print(automl_experiment.model_history)
        print(automl_experiment.best_iteration)
        print(automl_experiment.best_estimator)
        print(automl_experiment.best_config)
        print(automl_experiment.best_loss)
        print(automl_experiment.best_config_train_time)

    def test_sparse_matrix_xgboost(self):

        automl_experiment = AutoML()
        automl_settings = {
            "time_budget": 2,
            "metric": 'ap',
            "task": 'classification',
            "log_file_name": "test/sparse_classification.log",
            "estimator_list": ["xgboost"],
            "log_type": "all",
        }
        X_train = scipy.sparse.eye(900000)
        y_train = np.random.randint(2, size=900000)
        automl_experiment.fit(X_train=X_train, y_train=y_train,
                              **automl_settings)
        print(automl_experiment.predict(X_train))
        print(automl_experiment.model)
        print(automl_experiment.config_history)
        print(automl_experiment.model_history)
        print(automl_experiment.best_iteration)
        print(automl_experiment.best_estimator)

    def test_sparse_matrix_lr(self):

        automl_experiment = AutoML()
        automl_settings = {
            "time_budget": 2,
            "metric": 'f1',
            "task": 'classification',
            "log_file_name": "test/sparse_classification.log",
            "estimator_list": ["lrl1", "lrl2"],
            "log_type": "all",
        }
        X_train = scipy.sparse.random(3000, 900, density=0.1)
        y_train = np.random.randint(2, size=3000)
        automl_experiment.fit(X_train=X_train, y_train=y_train,
                              **automl_settings)
        print(automl_experiment.predict(X_train))
        print(automl_experiment.model)
        print(automl_experiment.config_history)
        print(automl_experiment.model_history)
        print(automl_experiment.best_iteration)
        print(automl_experiment.best_estimator)

    def test_sparse_matrix_regression_cv(self):

        automl_experiment = AutoML()
        automl_settings = {
            "time_budget": 2,
            'eval_method': 'cv',
            "task": 'regression',
            "log_file_name": "test/sparse_regression.log",
            "model_history": True
        }
        X_train = scipy.sparse.random(100, 100)
        y_train = np.random.uniform(size=100)
        automl_experiment.fit(X_train=X_train, y_train=y_train,
                              **automl_settings)
        print(automl_experiment.predict(X_train))
        print(automl_experiment.model)
        print(automl_experiment.config_history)
        print(automl_experiment.model_history)
        print(automl_experiment.best_iteration)
        print(automl_experiment.best_estimator)


if __name__ == "__main__":
    unittest.main()
