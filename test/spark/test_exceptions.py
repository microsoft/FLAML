from flaml.automl.data import load_openml_dataset
from flaml import AutoML
import flaml
from flaml.utils import check_spark
import os
import pytest

try:
    check_spark()
    skip_spark = False
except Exception:
    print("Spark is not installed. Skip all spark tests.")
    skip_spark = True

os.environ["FLAML_MAX_CONCURRENT"] = "2"


def base_automl(n_concurrent_trials=1, use_ray=False, use_spark=False, verbose=0):
    X_train, X_test, y_train, y_test = load_openml_dataset(
        dataset_id=537, data_dir="./"
    )
    automl = AutoML()
    settings = {
        "time_budget": 10,  # total running time in seconds
        "metric": "r2",  # primary metrics for regression can be chosen from: ['mae','mse','r2','rmse','mape']
        "estimator_list": ["lgbm", "rf", "xgboost"],  # list of ML learners
        "task": "regression",  # task type
        "log_file_name": "houses_experiment.log",  # flaml log file
        "seed": 7654321,  # random seed
        "n_concurrent_trials": n_concurrent_trials,  # the maximum number of concurrent learners
        "use_ray": use_ray,  # whether to use Ray for distributed training
        "use_spark": use_spark,  # whether to use Spark for distributed training
        "verbose": verbose,
    }

    automl.fit(X_train=X_train, y_train=y_train, **settings)

    print("Best ML leaner:", automl.best_estimator)
    print("Best hyperparmeter config:", automl.best_config)
    print("Best accuracy on validation data: {0:.4g}".format(1 - automl.best_loss))
    print(
        "Training duration of best run: {0:.4g} s".format(automl.best_config_train_time)
    )


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_both_ray_spark():
    with pytest.raises(ValueError):
        base_automl(n_concurrent_trials=2, use_ray=True, use_spark=True)


@pytest.mark.skipif(skip_spark, reason="Spark is not installed. Skip all spark tests.")
def test_verboses():
    for verbose in [1, 3, 5]:
        base_automl(verbose=verbose)


if __name__ == "__main__":
    base_automl()
