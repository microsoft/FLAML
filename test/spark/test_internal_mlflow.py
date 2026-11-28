import importlib
from test._artifact_paths import artifact_path
from uuid import uuid4

import pytest
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import flaml
from flaml.automl.spark import disable_spark_ansi_mode, restore_spark_ansi_mode
from flaml.automl.spark.utils import to_pandas_on_spark
from flaml.fabric import is_fabric_runtime

mlflow = pytest.importorskip("mlflow")
pyspark = pytest.importorskip("pyspark")
VectorAssembler = pytest.importorskip("pyspark.ml.feature").VectorAssembler

pytestmark = pytest.mark.spark

skip_spark = importlib.util.find_spec("pyspark") is None


@pytest.fixture(autouse=True, params=["false", "true"], ids=["public", "fabric"])
def _runtime_namespace(request, monkeypatch):
    monkeypatch.setenv("FLAML_FABRIC_RUNTIME", request.param)
    yield
    mlflow.autolog(disable=True)
    while mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture
def _isolated_autologging():
    from unittest.mock import patch

    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    with patch.dict(AUTOLOGGING_INTEGRATIONS, {}, clear=True):
        yield


def _sklearn_tune(config):
    config = config.copy()
    is_autolog = config.pop("is_autolog")
    is_parent_run = config.pop("is_parent_run")
    is_parallel = config.pop("is_parallel")
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=41)
    rf = RandomForestRegressor(**config, n_jobs=1, random_state=41)
    rf.fit(train_x, train_y)
    pred = rf.predict(test_x)
    r2 = r2_score(test_y, pred)
    if not is_autolog and not is_parent_run and not is_parallel:
        with mlflow.start_run(nested=True):
            mlflow.log_metric("r2", r2)
    return {"r2": r2}


def _test_tune(is_autolog, is_parent_run, is_parallel):
    mlflow_exp_name = f"test_mlflow_integration_{uuid4().hex}"
    mlflow_experiment = mlflow.set_experiment(mlflow_exp_name)
    params = {
        "n_estimators": flaml.tune.randint(4, 16),
        "min_samples_leaf": flaml.tune.randint(1, 10),
        "is_autolog": is_autolog,
        "is_parent_run": is_parent_run,
        "is_parallel": is_parallel,
    }
    if is_autolog:
        mlflow.autolog()
    else:
        mlflow.autolog(disable=True)
    if is_parent_run:
        mlflow.start_run(run_name=f"tune_autolog_{is_autolog}_sparktrial_{is_parallel}")
    flaml.tune.run(
        _sklearn_tune,
        params,
        metric="r2",
        mode="max",
        num_samples=3,
        use_spark=True if is_parallel else False,
        n_concurrent_trials=2 if is_parallel else 1,
        mlflow_exp_name=mlflow_exp_name,
    )
    mlflow.end_run()  # end current run
    mlflow.autolog(disable=True)
    return mlflow_experiment.experiment_id


def _check_mlflow_logging(possible_num_runs, metric, is_parent_run, experiment_id, is_automl=False, skip_tags=False):
    client = mlflow.tracking.MlflowClient()
    if isinstance(possible_num_runs, int):
        possible_num_runs = [possible_num_runs]
    if is_parent_run:
        parent_run = mlflow.last_active_run()
        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
        )
    else:
        child_runs = client.search_runs(experiment_ids=[experiment_id])
    experiment_name = client.get_experiment(experiment_id).name
    metrics = [metric in run.data.metrics for run in child_runs]
    namespace = "synapseml.flaml" if is_fabric_runtime() else "flaml"
    tags = [f"{namespace}.version" in run.data.tags for run in child_runs]
    params = ["learner" in run.data.params for run in child_runs]
    assert (
        len(child_runs) in possible_num_runs
    ), f"The number of child runs is not correct on experiment {experiment_name}."
    if possible_num_runs[0] > 0:
        assert all(metrics), f"The metrics are not logged correctly on experiment {experiment_name}."
        assert (
            all(tags) if not skip_tags else True
        ), f"The tags are not logged correctly on experiment {experiment_name}."
        assert (
            all(params) if is_automl else True
        ), f"The params are not logged correctly on experiment {experiment_name}."
    # mlflow.delete_experiment(experiment_id)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_nonsparkdata_noautolog_noparentrun():
    experiment_id = _test_automl_nonsparkdata(is_autolog=False, is_parent_run=False)
    _check_mlflow_logging(0, "r2", False, experiment_id, is_automl=True)  # no logging


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_sparkdata_noautolog_noparentrun():
    experiment_id = _test_automl_sparkdata(is_autolog=False, is_parent_run=False)
    _check_mlflow_logging(0, "mse", False, experiment_id, is_automl=True)  # no logging


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_tune_noautolog_noparentrun_parallel():
    experiment_id = _test_tune(is_autolog=False, is_parent_run=False, is_parallel=True)
    _check_mlflow_logging(0, "r2", False, experiment_id)


def test_tune_noautolog_noparentrun_nonparallel():
    experiment_id = _test_tune(is_autolog=False, is_parent_run=False, is_parallel=False)
    _check_mlflow_logging(3, "r2", False, experiment_id, skip_tags=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_tune_autolog_parentrun_parallel():
    experiment_id = _test_tune(is_autolog=True, is_parent_run=True, is_parallel=True)
    _check_mlflow_logging([4, 3], "r2", True, experiment_id)


def test_tune_autolog_parentrun_nonparallel():
    experiment_id = _test_tune(is_autolog=True, is_parent_run=True, is_parallel=False)
    _check_mlflow_logging(3, "r2", True, experiment_id)


def test_tune_autolog_noparentrun_nonparallel():
    experiment_id = _test_tune(is_autolog=True, is_parent_run=False, is_parallel=False)
    _check_mlflow_logging(3, "r2", False, experiment_id)


def test_tune_noautolog_parentrun_nonparallel():
    experiment_id = _test_tune(is_autolog=False, is_parent_run=True, is_parallel=False)
    _check_mlflow_logging(3, "r2", True, experiment_id)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_tune_autolog_noparentrun_parallel():
    experiment_id = _test_tune(is_autolog=True, is_parent_run=False, is_parallel=True)
    _check_mlflow_logging([4, 3], "r2", False, experiment_id)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_tune_noautolog_parentrun_parallel():
    experiment_id = _test_tune(is_autolog=False, is_parent_run=True, is_parallel=True)
    _check_mlflow_logging([4, 3], "r2", True, experiment_id)


@pytest.mark.parametrize("autolog", [False, True])
def test_reported_spark_objectives_keep_latest_metric_and_trial_identity(autolog, monkeypatch):
    from flaml.fabric.mlflow import MLflowIntegration

    accepted_run_ids = []
    record_trial = MLflowIntegration.record_trial

    def record(self, *args, **kwargs):
        accepted_run_ids.append(kwargs.get("run_id"))
        return record_trial(self, *args, **kwargs)

    monkeypatch.setattr(MLflowIntegration, "record_trial", record)

    def objective(config):
        from pyspark import TaskContext

        from flaml import tune

        assert TaskContext.get() is not None
        tune.report(loss=config["x"] + 100)
        tune.report(loss=float(config["x"]))

    experiment = mlflow.set_experiment(f"reported-spark-{uuid4().hex}")
    mlflow.autolog(disable=not autolog, log_models=False, silent=True)
    with mlflow.start_run() as parent:
        result = flaml.tune.run(
            objective,
            config={"x": flaml.tune.choice([1, 2])},
            points_to_evaluate=[{"x": 1}, {"x": 2}],
            metric="loss",
            mode="min",
            num_samples=2,
            use_spark=True,
            n_concurrent_trials=2,
            verbose=0,
        )
        assert len(result.trials) == 2
        assert result.best_result["loss"] == 1.0
        client = mlflow.MlflowClient()
        children = client.search_runs(
            [experiment.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'"
        )
        assert len(children) == len(result.trials)
        by_trial = {run.data.tags["flaml.trial_id"]: run for run in children}
        for trial in result.trials:
            assert by_trial[trial.trial_id].data.metrics == {"loss": float(trial.config["x"])}
        assert result.best_run_id == by_trial[result.best_trial.trial_id].info.run_id
        if autolog:
            assert set(accepted_run_ids) == {run.info.run_id for run in children}


def test_automl_spark_workers_use_the_driver_store_and_isolate_autologging(_isolated_autologging, monkeypatch):
    import copy

    from joblib import wrap_non_picklable_objects
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    def logged_metric(X_val, y_val, estimator, labels, *args, **kwargs):
        from mlflow.utils.autologging_utils import autologging_is_disabled
        from pyspark import TaskContext

        assert TaskContext.get() is not None
        assert autologging_is_disabled("sklearn")
        assert mlflow.active_run() is None
        mlflow.log_metric("automl_spark_worker_probe", 73.0)
        return float(estimator.params["n_estimators"]), {}

    monkeypatch.setenv("FLAML_MAX_CONCURRENT", "2")
    experiment = mlflow.set_experiment(f"automl-spark-workers-{uuid4().hex}")
    mlflow.sklearn.autolog(log_models=False, silent=True)
    configurations = copy.deepcopy(AUTOLOGGING_INTEGRATIONS)
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    with mlflow.start_run() as parent:
        automl = flaml.AutoML()
        automl.fit(
            X,
            y,
            task="regression",
            estimator_list=["rf"],
            max_iter=2,
            time_budget=90,
            n_jobs=1,
            n_concurrent_trials=2,
            use_spark=True,
            metric=wrap_non_picklable_objects(logged_metric),
            retrain_full=False,
            model_history=False,
            eval_method="holdout",
            verbose=0,
            starting_points={
                "rf": [
                    {"n_estimators": 4, "max_leaves": 4, "max_features": 1.0},
                    {"n_estimators": 8, "max_leaves": 4, "max_features": 1.0},
                ]
            },
        )
        assert automl.model is not None and automl.best_config["n_estimators"] == 4
        client = mlflow.MlflowClient()
        diagnostics = [
            run
            for run in client.search_runs([experiment.experiment_id])
            if run.data.metrics.get("automl_spark_worker_probe") == 73.0
        ]
        assert len(diagnostics) == 2
        assert all(run.info.status == "FINISHED" for run in diagnostics)
        assert all(not client.list_artifacts(run.info.run_id) for run in diagnostics)
        assert mlflow.active_run().info.run_id == parent.info.run_id
        assert AUTOLOGGING_INTEGRATIONS == configurations


def _test_automl_sparkdata(is_autolog, is_parent_run):
    estimator_list = ["rf_spark"]

    mlflow_exp_name = f"test_mlflow_integration_{uuid4().hex}"
    mlflow_experiment = mlflow.set_experiment(mlflow_exp_name)
    if is_autolog:
        mlflow.autolog()
    else:
        mlflow.autolog(disable=True)
    if is_parent_run:
        mlflow.start_run(run_name=f"automl_sparkdata_autolog_{is_autolog}")
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark, ansi_conf, adjusted = disable_spark_ansi_mode()
    cached = None
    try:
        pd_df = load_diabetes(as_frame=True).frame
        cached = spark.createDataFrame(pd_df).repartition(4).cache()
        train, _ = cached.randomSplit([0.8, 0.2], seed=1)
        featurizer = VectorAssembler(inputCols=pd_df.columns[:-1].tolist(), outputCol="features")
        train_data = featurizer.transform(train)["target", "features"]
        automl = flaml.AutoML()
        settings = {
            "max_iter": 3,
            "metric": "mse",
            "task": "regression",
            "log_file_name": artifact_path(__file__, "flaml_experiment.log"),
            "mlflow_exp_name": mlflow_exp_name,
            "log_type": "all",
            "n_splits": 2,
            "model_history": True,
            "estimator_list": estimator_list,
            "custom_hp": {"rf_spark": {"numTrees": {"domain": flaml.tune.randint(2, 5), "init_value": 2}}},
        }
        df = to_pandas_on_spark(to_pandas_on_spark(train_data).to_spark(index_col="index"))
        automl.fit(dataframe=df, label="target", **settings)
    finally:
        if cached is not None:
            cached.unpersist()
        restore_spark_ansi_mode(spark, ansi_conf, adjusted)
    mlflow.end_run()  # end current run
    mlflow.autolog(disable=True)
    return mlflow_experiment.experiment_id


def _test_automl_nonsparkdata(is_autolog, is_parent_run):
    mlflow_exp_name = f"test_mlflow_integration_{uuid4().hex}"
    mlflow_experiment = mlflow.set_experiment(mlflow_exp_name)
    if is_autolog:
        mlflow.autolog()
    else:
        mlflow.autolog(disable=True)
    if is_parent_run:
        mlflow.start_run(run_name=f"automl_nonsparkdata_autolog_{is_autolog}")
    automl_experiment = flaml.AutoML()
    automl_settings = {
        "max_iter": 3,
        "metric": "r2",
        "task": "regression",
        "n_concurrent_trials": 2,
        "use_spark": True,
        "mlflow_exp_name": None if is_parent_run else mlflow_exp_name,
        "log_type": "all",
        "n_splits": 2,
        "model_history": True,
        "estimator_list": ["rf"],
        "n_jobs": 1,
        "custom_hp": {"rf": {"n_estimators": {"domain": flaml.tune.randint(4, 8), "init_value": 4}}},
    }
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    train_x, _, train_y, _ = train_test_split(X, y, test_size=0.25, random_state=41)
    automl_experiment.fit(X_train=train_x, y_train=train_y, **automl_settings)
    mlflow.end_run()  # end current run
    mlflow.autolog(disable=True)
    return mlflow_experiment.experiment_id


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_sparkdata_autolog_parentrun():
    experiment_id = _test_automl_sparkdata(is_autolog=True, is_parent_run=True)
    _check_mlflow_logging(3, "mse", True, experiment_id, is_automl=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_sparkdata_autolog_noparentrun():
    experiment_id = _test_automl_sparkdata(is_autolog=True, is_parent_run=False)
    _check_mlflow_logging(3, "mse", False, experiment_id, is_automl=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_sparkdata_noautolog_parentrun():
    experiment_id = _test_automl_sparkdata(is_autolog=False, is_parent_run=True)
    _check_mlflow_logging(3, "mse", True, experiment_id, is_automl=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_nonsparkdata_autolog_parentrun():
    experiment_id = _test_automl_nonsparkdata(is_autolog=True, is_parent_run=True)
    _check_mlflow_logging([4, 3], "r2", True, experiment_id, is_automl=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_nonsparkdata_autolog_noparentrun():
    experiment_id = _test_automl_nonsparkdata(is_autolog=True, is_parent_run=False)
    _check_mlflow_logging([4, 3], "r2", False, experiment_id, is_automl=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_automl_nonsparkdata_noautolog_parentrun():
    experiment_id = _test_automl_nonsparkdata(is_autolog=False, is_parent_run=True)
    _check_mlflow_logging([4, 3], "r2", True, experiment_id, is_automl=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def _stop_pyspark_autolog():
    import pyspark

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark.sparkContext._gateway.shutdown_callback_server()  # this is to avoid stucking
    mlflow.autolog(disable=True)


def _init_spark_for_main():
    from test.spark._init_spark import setup_spark_for_tests

    setup_spark_for_tests("MyApp")


if __name__ == "__main__":
    _init_spark_for_main()

    test_tune_autolog_parentrun_parallel()
    # test_tune_autolog_parentrun_nonparallel()
    # test_tune_autolog_noparentrun_parallel()  # TODO: runs not removed
    # test_tune_noautolog_parentrun_parallel()
    # test_tune_autolog_noparentrun_nonparallel()
    # test_tune_noautolog_parentrun_nonparallel()
    # test_tune_noautolog_noparentrun_parallel()
    # test_tune_noautolog_noparentrun_nonparallel()
    # test_automl_sparkdata_autolog_parentrun()
    # test_automl_sparkdata_autolog_noparentrun()
    # test_automl_sparkdata_noautolog_parentrun()
    # test_automl_sparkdata_noautolog_noparentrun()
    # test_automl_nonsparkdata_autolog_parentrun()
    # test_automl_nonsparkdata_autolog_noparentrun()  # TODO: runs not removed
    # test_automl_nonsparkdata_noautolog_parentrun()
    # test_automl_nonsparkdata_noautolog_noparentrun()

    _stop_pyspark_autolog()
