import numpy as np
from sklearn.datasets import fetch_openml, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from flaml.automl import AutoML

dataset = "credit-g"


def _test(split_type):
    from sklearn.externals._arff import ArffException

    automl = AutoML()

    automl_settings = {
        "time_budget": 2,
        # "metric": 'accuracy',
        "task": "classification",
        "log_file_name": f"test/{dataset}.log",
        "model_history": True,
        "log_training_metric": True,
        "split_type": split_type,
    }

    try:
        X, y = fetch_openml(name=dataset, return_X_y=True)
    except (ArffException, ValueError):
        from sklearn.datasets import load_wine

        X, y = load_wine(return_X_y=True)
    if split_type != "time":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

    pred = automl.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(acc)


def _test_uniform():
    _test(split_type="uniform")


def test_time():
    _test(split_type="time")


def test_groups_for_classification_task():
    from sklearn.externals._arff import ArffException

    try:
        X, y = fetch_openml(name=dataset, return_X_y=True)
    except (ArffException, ValueError):
        from sklearn.datasets import load_wine

        X, y = load_wine(return_X_y=True)

    import numpy as np

    automl = AutoML()
    automl_settings = {
        "time_budget": 2,
        "task": "classification",
        "log_file_name": f"test/{dataset}.log",
        "model_history": True,
        "eval_method": "cv",
        "groups": np.random.randint(low=0, high=10, size=len(y)),
        "estimator_list": ["lgbm", "rf", "xgboost", "kneighbor"],
        "learner_selector": "roundrobin",
    }
    automl.fit(X, y, **automl_settings)

    automl_settings["eval_method"] = "holdout"
    automl.fit(X, y, **automl_settings)

    automl_settings["split_type"] = GroupKFold(n_splits=3)
    try:
        automl.fit(X, y, **automl_settings)
        raise RuntimeError("GroupKFold object as split_type should fail when eval_method is holdout")
    except AssertionError:
        # eval_method must be 'auto' or 'cv' for custom data splitter.
        pass

    automl_settings["eval_method"] = "cv"
    automl.fit(X, y, **automl_settings)


def test_groups_for_regression_task():
    """Append nonsensical groups to iris dataset and use it to test that GroupKFold works for regression tasks"""
    iris_dict_data = load_iris(as_frame=True)  # numpy arrays
    iris_data = iris_dict_data["frame"]  # pandas dataframe data + target

    rng = np.random.default_rng(42)
    iris_data["cluster"] = rng.integers(
        low=0, high=5, size=iris_data.shape[0]
    )  # np.random.randint(0, 5, iris_data.shape[0])

    automl = AutoML()
    X = iris_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]].to_numpy()
    y = iris_data["petal width (cm)"]
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, iris_data["cluster"], random_state=42
    )
    automl_settings = {
        "max_iter": 5,
        "time_budget": -1,
        "metric": "r2",
        "task": "regression",
        "estimator_list": ["lgbm", "rf", "xgboost", "kneighbor"],
        "eval_method": "cv",
        "split_type": "uniform",
        "groups": groups_train,
    }
    automl.fit(X_train, y_train, **automl_settings)


def test_stratified_groupkfold():
    from minio.error import ServerError
    from sklearn.model_selection import StratifiedGroupKFold

    from flaml.automl.data import load_openml_dataset

    try:
        X_train, _, y_train, _ = load_openml_dataset(dataset_id=1169, data_dir="test/")
    except (ServerError, Exception):
        return
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    automl = AutoML()
    settings = {
        "time_budget": 6,
        "metric": "ap",
        "eval_method": "cv",
        "split_type": splitter,
        "groups": X_train["Airline"],
        "estimator_list": [
            "lgbm",
            "rf",
            "xgboost",
            "extra_tree",
            "xgb_limitdepth",
            "lrl1",
        ],
    }

    automl.fit(X_train=X_train, y_train=y_train, **settings)


def test_rank():
    from sklearn.externals._arff import ArffException

    try:
        X, y = fetch_openml(name=dataset, return_X_y=True)
        y = y.cat.codes
    except (ArffException, ValueError):
        from sklearn.datasets import load_wine

        X, y = load_wine(return_X_y=True)
    import numpy as np

    automl = AutoML()
    automl_settings = {
        "time_budget": 2,
        "task": "rank",
        "log_file_name": f"test/{dataset}.log",
        "model_history": True,
        "eval_method": "cv",
        "groups": np.array([0] * 200 + [1] * 200 + [2] * 200 + [3] * 200 + [4] * 100 + [5] * 100),  # group labels
        "learner_selector": "roundrobin",
    }
    automl.fit(X, y, **automl_settings)

    automl = AutoML()
    automl_settings = {
        "time_budget": 2,
        "task": "rank",
        "metric": "ndcg@5",  # 5 can be replaced by any number
        "log_file_name": f"test/{dataset}.log",
        "model_history": True,
        "groups": [200] * 4 + [100] * 2,  # alternative way: group counts
        # "estimator_list": ['lgbm', 'xgboost'],  # list of ML learners
        "learner_selector": "roundrobin",
    }
    automl.fit(X, y, **automl_settings)


def test_object():
    from sklearn.externals._arff import ArffException

    try:
        X, y = fetch_openml(name=dataset, return_X_y=True)
    except (ArffException, ValueError):
        from sklearn.datasets import load_wine

        X, y = load_wine(return_X_y=True)

    import numpy as np

    class TestKFold(KFold):
        def __init__(self, n_splits):
            self.n_splits = int(n_splits)

        def split(self, X):
            rng = np.random.default_rng()
            train_num = int(len(X) * 0.8)
            for _ in range(self.n_splits):
                permu_idx = rng.permutation(len(X))
                yield permu_idx[:train_num], permu_idx[train_num:]

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

    automl = AutoML()
    automl_settings = {
        "time_budget": 2,
        "task": "classification",
        "log_file_name": f"test/{dataset}.log",
        "model_history": True,
        "log_training_metric": True,
        "split_type": TestKFold(5),
    }
    automl.fit(X, y, **automl_settings)
    assert automl._state.eval_method == "cv", "eval_method must be 'cv' for custom data splitter"

    kf = TestKFold(5)
    kf.shuffle = True
    automl_settings["split_type"] = kf
    automl.fit(X, y, **automl_settings)


if __name__ == "__main__":
    test_groups_for_classification_task()
