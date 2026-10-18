"""Compare models on held-out data and retain explicit dataset-download failure statuses."""

import gc
import json
import logging
import math
import os
import random
import time
import warnings
from numbers import Real

import numpy
import pmlb
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

GLOBAL_CONF = {
    "time_budget": 60,
    # "num_iter": 10,
    "seed": 7654321,
}
GLOABL_CONF = GLOBAL_CONF

logger = logging.getLogger(__name__)


def _download_failure(conf, error):
    logger.warning("Could not fetch benchmark dataset %s: %s", conf["dataset"], error)
    return {
        "score": None,
        "duration": 0.0,
        "metric": conf["metric"],
        "status": "download_failed",
        "error": str(error),
    }


def flatten_results(results):
    """Build report rows without treating missing evaluation scores as measurements."""
    flattened = {}
    for dataset, runners in results.items():
        row = {}
        for runner, metrics in runners.items():
            if metrics["status"] == "ok" and (
                isinstance(metrics["score"], (bool, numpy.bool_))
                or not isinstance(metrics["score"], Real)
                or not math.isfinite(metrics["score"])
            ):
                raise ValueError(f"Successful benchmark {dataset}/{runner} must have a finite scalar score.")
            row["metric"] = metrics["metric"]
            for field in ("score", "duration", "status", "error"):
                row[f"{runner}_{field}"] = metrics[field]
        flattened[dataset] = row
    return flattened


def runner_flaml(conf, use_spark=False):
    try:
        X, y = pmlb.fetch_data(conf["dataset"], return_X_y=True)
        print(f"Running on dataset: {conf['dataset']}")
    except (OSError, ValueError) as exc:
        return _download_failure(conf, exc)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=GLOBAL_CONF.get("seed", None)
    )
    import flaml

    aml = flaml.AutoML()

    automl_settings = {
        "max_iter": GLOBAL_CONF.get("num_iter", None),
        "time_budget": GLOBAL_CONF.get("time_budget", None),
        "metric": conf["metric"],
        "task": conf["task"],
        "estimator_list": ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"],
        "seed": GLOBAL_CONF.get("seed", None),
    }

    if use_spark:
        automl_settings["use_spark"] = True
        automl_settings["n_concurrent_trials"] = 2

    if conf["task"] == "classification":
        automl_settings["estimator_list"] += ["lrl1"]

    start = time.time()
    aml.fit(X_train=X_train, y_train=y_train, **automl_settings)

    pred = aml.predict(X_test)

    if conf["task"] == "classification":
        score = accuracy_score(y_test, pred)
    else:
        score = r2_score(y_test, pred)
    del X, y, X_train, y_train, X_test, y_test, pred, aml
    end = time.time()
    return {"score": score, "duration": end - start, "metric": conf["metric"], "status": "ok", "error": None}


def runner_h2o(conf):
    try:
        df = pmlb.fetch_data(conf["dataset"])
        print(f"Running on dataset: {conf['dataset']}")
    except (OSError, ValueError) as exc:
        return _download_failure(conf, exc)

    import h2o

    h2o.init()

    if conf["task"] == "classification":
        df["target"] = df["target"].astype("int")

    hf = h2o.H2OFrame(df)
    if conf["task"] == "classification":
        hf["target"] = hf["target"].asfactor()
    train, test = hf.split_frame(ratios=[0.75], seed=GLOBAL_CONF.get("seed", None))

    aml = h2o.automl.H2OAutoML(
        max_runtime_secs=GLOBAL_CONF.get("time_budget", None),
        max_models=GLOBAL_CONF.get("num_iter", None),
        seed=GLOBAL_CONF.get("seed", None),
    )
    start = time.time()
    # H2O selects its leader using training CV, never the final test partition.
    aml.train(y="target", training_frame=train)
    if conf["metric"] == "r2":
        score = aml.leader.model_performance(test).r2()
    else:
        predictions = aml.leader.predict(test).as_data_frame()["predict"]
        actual = test["target"].as_data_frame()["target"]
        score = float(accuracy_score(actual, predictions))
    end = time.time()
    return {"score": score, "duration": end - start, "metric": conf["metric"], "status": "ok", "error": None}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    random.seed(GLOBAL_CONF.get("seed", None))
    numpy.random.seed(GLOBAL_CONF.get("seed", None))
    with open("benchmark/pmlb/choice_tasks.json", encoding="utf-8") as task_file:
        BENCHMARK_TASKS = json.load(task_file)
    runners = {"h2o": runner_h2o, "flaml": runner_flaml}
    result = {}
    os.makedirs("benchmark_results", exist_ok=True)

    for runner, runner_fn in runners.items():
        for task in BENCHMARK_TASKS:
            if task["task"] == "regression" and task["size"] not in ["xl", "l"]:
                if task["dataset"] not in result.keys():
                    result[task["dataset"]] = {}
                result[task["dataset"]][runner] = runner_fn(task)
                with open("benchmark_results/temp_result.json", "w", encoding="utf-8") as result_file:
                    json.dump(result, result_file)
                gc.collect()

    flatten_result = flatten_results(result)

    import pandas as pd

    df = pd.DataFrame(flatten_result).T
    print(df)
    df.to_csv(f"benchmark_results/{time.time()}.csv")
