import unittest
from sklearn.datasets import fetch_openml
from flaml.automl import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from flaml.data import load_openml_task

dataset = "kc1"
task_id = 3917
fold = 0


class TestEnsemble(unittest.TestCase):


    def test_ensemble(self):
        automl = AutoML()

        automl_settings = {
            "time_budget": 10,
            # "metric": 'roc_auc',
            "objective_name": 'classification',
            "log_file_name": "test/{}.log".format(dataset),
            "model_history": True,
            "ensemble": True,
            # "n_jobs": 1,
            # "estimator_list": ['lgbm','nn'],
        }
        # X_train, X_test, y_train, y_test = load_openml_task(task_id, "test")

        X, y = fetch_openml(name=dataset, return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
         random_state=42)
        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

        pred = automl.predict(X_test)
        pred_prob = automl.predict_proba(X_test)
        acc = accuracy_score(y_test, pred)

        print(acc)
        automl.__del__()

if __name__ == "__main__":
    unittest.main()
