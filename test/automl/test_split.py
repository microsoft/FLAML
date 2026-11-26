import platform
import sys
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest
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
    except (ArffException, ValueError, URLError):
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
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


def test_time_split_with_sample_weight():
    """Regression for #887 and #1553: split_type='time' + sample_weight + holdout."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
    sample_weight = np.ones(len(y))
    automl = AutoML()
    automl.fit(
        X_train=X,
        y_train=y,
        sample_weight=sample_weight,
        split_type="time",
        retrain_full=True,
        eval_method="holdout",
        task="classification",
        time_budget=-1,
        max_iter=2,
        estimator_list=["lgbm"],
    )
    assert automl.model is not None


def test_group_split_with_sample_weight_alignment():
    """Regression for group holdout: sample_weight must be split by the group
    train/val indices, not left full-length and sliced positionally downstream."""
    from unittest.mock import patch

    from sklearn.ensemble import RandomForestClassifier

    n = 200
    rng = np.random.default_rng(42)
    # column 0 is a row id, so each row's expected weight is recoverable from X
    X = np.column_stack([np.arange(n, dtype=float), rng.normal(size=n)])
    groups = np.repeat(np.arange(20), 10)
    y = (groups % 2).astype(int)
    sample_weight = 1000.0 + np.arange(n, dtype=float)

    captured = []
    original_fit = RandomForestClassifier.fit

    def spy_fit(self, X, y, sample_weight=None, **kwargs):
        captured.append(
            (
                np.asarray(X).copy(),
                None if sample_weight is None else np.asarray(sample_weight, dtype=float).copy(),
            )
        )
        return original_fit(self, X, y, sample_weight=sample_weight, **kwargs)

    automl = AutoML()
    with patch.object(RandomForestClassifier, "fit", spy_fit):
        automl.fit(
            X_train=X,
            y_train=y,
            sample_weight=sample_weight,
            groups=groups,
            split_type="group",
            eval_method="holdout",
            task="classification",
            time_budget=-1,
            max_iter=2,
            estimator_list=["rf"],
        )
    assert automl.model is not None
    assert captured
    for X_fit, weight_fit in captured:
        assert weight_fit is not None
        # each row trains with its own weight (1000 + row id) after the split
        np.testing.assert_array_equal(weight_fit, 1000.0 + np.asarray(X_fit)[:, 0])
    # the validation loss must be weighted as well
    assert automl._state.weight_val is not None


def test_group_split_with_sample_weight_series_alignment():
    """Same regression as above, but sample_weight is a pd.Series with a
    non-default index, to verify the group branch slices it positionally
    (.iloc) rather than by label."""
    from unittest.mock import patch

    from sklearn.ensemble import RandomForestClassifier

    n = 200
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        {"id": np.arange(n, dtype=float), "feat": rng.normal(size=n)},
    )
    groups = np.repeat(np.arange(20), 10)
    y = pd.Series((groups % 2).astype(int))
    # non-default index: label != positional order, so .loc and .iloc would disagree
    weight_index = np.arange(n) + 500
    sample_weight = pd.Series(1000.0 + np.arange(n, dtype=float), index=weight_index)

    captured = []
    original_fit = RandomForestClassifier.fit

    def spy_fit(self, X, y, sample_weight=None, **kwargs):
        captured.append(
            (
                np.asarray(X).copy(),
                None if sample_weight is None else np.asarray(sample_weight, dtype=float).copy(),
            )
        )
        return original_fit(self, X, y, sample_weight=sample_weight, **kwargs)

    automl = AutoML()
    with patch.object(RandomForestClassifier, "fit", spy_fit):
        automl.fit(
            X_train=X,
            y_train=y,
            sample_weight=sample_weight,
            groups=groups,
            split_type="group",
            eval_method="holdout",
            task="classification",
            time_budget=-1,
            max_iter=2,
            estimator_list=["rf"],
        )
    assert automl.model is not None
    assert captured
    for X_fit, weight_fit in captured:
        assert weight_fit is not None
        # each row trains with its own weight (1000 + row id) after the split,
        # regardless of the Series' original (non-default) index
        np.testing.assert_array_equal(weight_fit, 1000.0 + np.asarray(X_fit)[:, 0])
    assert automl._state.weight_val is not None


@pytest.mark.parametrize("eval_method", ["holdout", "cv"])
@pytest.mark.parametrize("kind", ["list", "tuple", "series_offset_index", "group_kfold_list"])
def test_group_split_with_non_array_groups(kind, eval_method):
    """groups may be any array-like: a list, a tuple or a pd.Series whose index is not 0..n-1."""
    from unittest.mock import patch

    from sklearn.ensemble import RandomForestClassifier

    if kind == "group_kfold_list" and eval_method != "cv":
        pytest.skip("a splitter object requires eval_method='cv'")
    n = 200
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"id": np.arange(n, dtype=float), "feat": rng.normal(size=n)})
    group_labels = np.repeat(np.arange(20), 10)
    y = pd.Series((group_labels % 2).astype(int))
    groups = {
        "list": group_labels.tolist(),
        "tuple": tuple(group_labels.tolist()),
        "series_offset_index": pd.Series(group_labels, index=np.arange(n) + 500),
        "group_kfold_list": group_labels.tolist(),
    }[kind]
    split_type = GroupKFold(3) if kind == "group_kfold_list" else "group"

    captured = []
    original_fit = RandomForestClassifier.fit

    def spy_fit(self, X, y, **kwargs):
        captured.append(np.asarray(X)[:, 0].copy())
        return original_fit(self, X, y, **kwargs)

    automl = AutoML()
    with patch.object(RandomForestClassifier, "fit", spy_fit):
        automl.fit(
            X_train=X,
            y_train=y,
            groups=groups,
            split_type=split_type,
            eval_method=eval_method,
            n_splits=3,
            task="classification",
            time_budget=-1,
            max_iter=2,
            estimator_list=["rf"],
            verbose=0,
        )
    assert automl.model is not None
    assert captured
    all_counts = [np.bincount((ids // 10).astype(int), minlength=20) for ids in captured]
    for counts in all_counts:
        # every fit sees whole groups only: a group is never split between train and validation
        assert set(counts) <= {0, 10}
    # at least one fit actually holds some groups out
    assert any(0 in counts for counts in all_counts)


def _spy_val_loss_calls():
    """record the rows, the training group labels and the validation group labels of every evaluated fold"""
    from flaml.automl import ml

    calls = []
    original = ml.get_val_loss  # the CV path calls the same function through generic_task

    def spy(config, estimator, X_train, y_train, X_val, y_val, weight_val, groups_val, *args, **kwargs):
        fit_kwargs = kwargs.get("fit_kwargs") or (args[-1] if args and isinstance(args[-1], dict) else {})
        calls.append(
            {
                "train_ids": np.asarray(X_train)[:, 0].astype(int),
                "val_ids": np.asarray(X_val)[:, 0].astype(int),
                "groups": fit_kwargs.get("groups"),
                "groups_val": groups_val,
            }
        )
        return original(config, estimator, X_train, y_train, X_val, y_val, weight_val, groups_val, *args, **kwargs)

    return calls, spy


@pytest.mark.parametrize("eval_method", ["holdout", "cv"])
@pytest.mark.parametrize("kind", ["list", "series_offset_index", "series_permuted_index"])
def test_group_labels_follow_rows_in_every_fold(kind, eval_method):
    """the group labels handed to each fold are the labels of the rows in that fold"""
    from unittest.mock import patch

    n = 200
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"id": np.arange(n, dtype=float), "feat": rng.normal(size=n)})
    labels = np.repeat(np.arange(20), 10)
    y = pd.Series((labels % 2).astype(int))
    groups = {
        "list": labels.tolist(),
        "series_offset_index": pd.Series(labels, index=np.arange(n) + 500),
        "series_permuted_index": pd.Series(labels, index=rng.permutation(n)),
    }[kind]
    calls, spy = _spy_val_loss_calls()
    with patch("flaml.automl.ml.get_val_loss", spy), patch("flaml.automl.task.generic_task.get_val_loss", spy):
        AutoML().fit(
            X_train=X,
            y_train=y,
            groups=groups,
            split_type="group",
            eval_method=eval_method,
            n_splits=3,
            task="classification",
            time_budget=-1,
            max_iter=2,
            estimator_list=["rf"],
            verbose=0,
        )
    assert calls
    held_out = False
    for call in calls:
        assert call["groups_val"] is not None
        np.testing.assert_array_equal(np.asarray(call["groups_val"]), labels[call["val_ids"]])
        if call["groups"] is not None:
            np.testing.assert_array_equal(np.asarray(call["groups"]), labels[call["train_ids"]])
        train_groups, val_groups = set(labels[call["train_ids"]]), set(labels[call["val_ids"]])
        assert not train_groups & val_groups
        held_out |= bool(val_groups)
    assert held_out


@pytest.mark.parametrize("kind", ["list", "tuple", "series_offset_index"])
def test_group_labels_with_caller_supplied_validation_data(kind):
    """X_val / y_val with non-array groups and groups_val: labels stay aligned with their rows"""
    from unittest.mock import patch

    n_train = 160
    rng = np.random.default_rng(5)
    X_all = pd.DataFrame({"id": np.arange(200, dtype=float), "feat": rng.normal(size=200)})
    labels_all = np.repeat(np.arange(20), 10)
    y_all = pd.Series((labels_all % 2).astype(int))
    X_train, X_val = X_all.iloc[:n_train], X_all.iloc[n_train:].reset_index(drop=True)
    y_train, y_val = y_all.iloc[:n_train], y_all.iloc[n_train:].reset_index(drop=True)
    labels_train, labels_val = labels_all[:n_train], labels_all[n_train:]
    wrap = {
        "list": lambda a: a.tolist(),
        "tuple": lambda a: tuple(a.tolist()),
        "series_offset_index": lambda a: pd.Series(a, index=np.arange(len(a)) + 500),
    }[kind]
    calls, spy = _spy_val_loss_calls()
    with patch("flaml.automl.ml.get_val_loss", spy), patch("flaml.automl.task.generic_task.get_val_loss", spy):
        automl = AutoML()
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            groups=wrap(labels_train),
            groups_val=wrap(labels_val),
            split_type="group",
            eval_method="holdout",
            task="classification",
            time_budget=-1,
            max_iter=2,
            estimator_list=["rf"],
            verbose=0,
        )
    assert automl.model is not None
    assert calls
    for call in calls:
        np.testing.assert_array_equal(call["val_ids"], X_val["id"].to_numpy().astype(int))
        np.testing.assert_array_equal(np.asarray(call["groups_val"]), labels_val)
        if call["groups"] is not None:
            np.testing.assert_array_equal(np.asarray(call["groups"]), labels_train[call["train_ids"]])


def test_group_holdout_with_permuted_index_series_keeps_group_alignment():
    """A Series whose index is a permutation of 0..n-1 does not raise, but was indexed by label."""
    from unittest.mock import patch

    from flaml.automl.task.generic_task import GenericTask

    snapshot = {}
    original = GenericTask.prepare_data

    def spy(self, state, *args, **kwargs):
        out = original(self, state, *args, **kwargs)
        snapshot.update(
            train_ids=np.asarray(state.X_train)[:, 0].astype(int),
            val_ids=np.asarray(state.X_val)[:, 0].astype(int),
            groups=np.asarray(state.groups),
            groups_val=np.asarray(state.groups_val),
        )
        return out

    n = 200
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"id": np.arange(n, dtype=float), "feat": rng.normal(size=n)})
    labels = np.repeat(np.arange(20), 10)
    y = pd.Series((labels % 2).astype(int))
    groups = pd.Series(labels, index=rng.permutation(n))  # values in row order, shuffled index labels
    with patch.object(GenericTask, "prepare_data", spy):
        AutoML().fit(
            X_train=X,
            y_train=y,
            groups=groups,
            split_type="group",
            eval_method="holdout",
            task="classification",
            time_budget=-1,
            max_iter=2,
            estimator_list=["rf"],
            verbose=0,
        )
    np.testing.assert_array_equal(snapshot["groups"], labels[snapshot["train_ids"]])
    np.testing.assert_array_equal(snapshot["groups_val"], labels[snapshot["val_ids"]])


@pytest.mark.skipif(
    sys.platform == "win32" and platform.machine() == "ARM64", reason="catboost is not available on win-arm64 machine"
)
def test_groups_for_classification_task():
    from sklearn.externals._arff import ArffException

    try:
        X, y = fetch_openml(name=dataset, return_X_y=True)
    except (ArffException, ValueError, URLError):
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)

    automl = AutoML()
    automl_settings = {
        "time_budget": 2,
        "task": "classification",
        "log_file_name": f"test/{dataset}.log",
        "model_history": True,
        "eval_method": "cv",
        "groups": np.random.randint(low=0, high=10, size=len(y)),
        "estimator_list": ["catboost", "lgbm", "rf", "xgboost", "kneighbor"],
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


def test_groups_with_sample_weights():
    """Verifies that sample weights can be used with group splits i.e. that https://github.com/microsoft/FLAML/issues/1396 remains fixed"""
    iris_dict_data = load_iris(as_frame=True)  # numpy arrays
    iris_data = iris_dict_data["frame"]  # pandas dataframe data + target
    iris_data["cluster"] = np.random.randint(0, 5, iris_data.shape[0])
    automl = AutoML()

    X = iris_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]].to_numpy()
    y = iris_data["petal width (cm)"]
    sample_weight = pd.Series(np.random.rand(X.shape[0]))
    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        sample_weight_train,
        sample_weight_test,
    ) = train_test_split(X, y, iris_data["cluster"], sample_weight, random_state=42)
    automl_settings = {
        "max_iter": 5,
        "time_budget": -1,
        "metric": "r2",
        "task": "regression",
        "log_file_name": "error.log",
        "log_type": "all",
        "estimator_list": ["lgbm"],
        "eval_method": "cv",
        "split_type": "group",
        "groups": groups_train,
        "sample_weight": sample_weight_train,
    }
    automl.fit(X_train, y_train, **automl_settings)
    assert automl.model is not None


@pytest.mark.skipif(
    sys.platform == "win32" and platform.machine() == "ARM64",
    reason="minio and catboost are not available on win-arm64 machine",
)
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
            "catboost",
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
    except (ArffException, ValueError, URLError):
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
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
    except (ArffException, ValueError, URLError):
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)

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
