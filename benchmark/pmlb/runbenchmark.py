import gc
import json
import os
import random
import time
import warnings

import numpy
import pmlb
import sklearn
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
# spark = build_test_spark()

BENCHMARK_TASKS = json.load(open("benchmark/pmlb/choice_tasks.json"))
GLOABL_CONF = {
    "time_budget": 60,
    # "num_iter": 10,
    "seed": 7654321,
}

random.seed(GLOABL_CONF.get("seed", None))
numpy.random.seed(GLOABL_CONF.get("seed", None))


def runner_flaml(conf, use_spark=False):
    try:
        X, y = pmlb.fetch_data(conf["dataset"], return_X_y=True)
        print(f"Running on dataset: {conf['dataset']}")
    except Exception:
        print(f"{conf['dataset']} is not in pmlb database")
        return {"score": 0, "duration": 0}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=GLOABL_CONF.get("seed", None)
    )
    import flaml

    aml = flaml.AutoML()

    automl_settings = {
        "max_iter": GLOABL_CONF.get("num_iter", None),
        "time_budget": GLOABL_CONF.get("time_budget", None),
        "metric": conf["metric"],
        "task": conf["task"],
        "estimator_list": ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"],
        "seed": GLOABL_CONF.get("seed", None),
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
    return {"score": score, "duration": end - start, "metric": conf["metric"]}


def runner_h2o(conf):
    import h2o

    h2o.init()

    try:
        df = pmlb.fetch_data(conf["dataset"])
        print(f"Running on dataset: {conf['dataset']}")
    except Exception:
        print(f"{conf['dataset']} is not in pmlb database")
        return {"score": 0, "duration": 0}

    if conf["task"] == "classification":
        df["target"] = df["target"].astype("int")

    hf = h2o.H2OFrame(df)
    train, test = hf.split_frame(ratios=[0.75], seed=GLOABL_CONF.get("seed", None))

    aml = h2o.automl.H2OAutoML(
        max_runtime_secs=GLOABL_CONF.get("time_budget", None),
        max_models=GLOABL_CONF.get("num_iter", None),
        seed=GLOABL_CONF.get("seed", None),
    )
    start = time.time()
    aml.train(y="target", training_frame=train, leaderboard_frame=test)

    perf = aml.leader.model_performance(test)
    # del df, hf, train, test, aml
    end = time.time()
    if conf["metric"] == "r2":
        score = perf.r2()
    else:
        score = perf.accuracy()
    return {"score": score, "duration": end - start, "metric": conf["metric"]}


if __name__ == "__main__":
    runners = {"h2o": runner_h2o, "flaml": runner_flaml}
    result = {}
    os.makedirs("benchmark_results", exist_ok=True)

    for runner, runner_fn in runners.items():
        for task in BENCHMARK_TASKS:
            if task["task"] == "regression" and task["size"] not in ["xl", "l"]:
                if task["dataset"] not in result.keys():
                    result[task["dataset"]] = {}
                result[task["dataset"]][runner] = runner_fn(task)
                json.dump(result, open("benchmark_results/temp_result.json", "w"))
                gc.collect()

    flatten_result = {}
    for dataset, runner_result in result.items():
        flatten_result[dataset] = {}
        for runner, metrics in runner_result.items():
            flatten_result[dataset]["metric"] = metrics["metric"]
            flatten_result[dataset][f"{runner}_score"] = metrics["score"]
        for runner, metrics in runner_result.items():
            flatten_result[dataset][f"{runner}_duration"] = metrics["duration"]

    import pandas as pd

    df = pd.DataFrame(flatten_result).T
    print(df)
    df.to_csv(f"benchmark_results/{time.time()}.csv")
