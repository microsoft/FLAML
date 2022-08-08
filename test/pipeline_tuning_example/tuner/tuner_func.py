import time
import flaml
import azureml.core
from azureml.core import Run
import submit_train_pipeline
from functools import partial
import os
import logging
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

logger = logging.getLogger(__name__)


def wait_for_completion(run):
    """Wait for the run to complete
    """
    status = 'Preparing'
    while status not in ['Failed', 'Completed']:
        status = run.status
        print(f'status: {status}')
        time.sleep(1)

    print("The run is terminated.")
    print(status)
    return status


def get_all_metrics(ml_client, run, metrics_name):
    # get all the metrics.
    exp_name = run.experiment_name
    run_id = run.name

    import mlflow
    from mlflow.tracking import MlflowClient

    # need to connect with workspace for local run.
    # TODO: remove duplication.

    track_uri = ml_client.workspaces.get().mlflow_tracking_uri
    mlflow.set_tracking_uri(track_uri)

    mlflow_client = MlflowClient()
    mlflow.set_experiment(exp_name)

    query = f"tags.mlflow.parentRunId = '{run_id}'"
    df = mlflow.search_runs(filter_string=query)
    # drop NAN
    # TODO: polish
    complete_df = df.dropna(subset=["metrics.test_accuracy_score"])
    if len(complete_df) == 0:
        return None
    elif len(complete_df) > 1:
        raise Exception(f"Found more than one metric with the same run id: {run_id}.")

    metric = complete_df.iloc[0]["metrics.test_accuracy_score"]

    return metric


def run_with_config(config: dict):
    """Run the pipeline with a given config dict
    """

    overrides = [f"{key}={value}" for key, value in config.items()]
    # # overwrite the path to deep speed configuration.
    # if isinstance(Run.get_context(), azureml.core.run._OfflineRun):
    #     config_searchpath = os.path.abspath(os.path.join(deepspeed_wd, "..\\.."))
    # else:
    #     config_searchpath = deepspeed_wd
    # overrides += [f'+script_args.deepspeed_wd={deepspeed_wd}', f'hydra.searchpath=[{config_searchpath}]']

    print(overrides)
    run = submit_train_pipeline.build_and_submit_aml_pipeline(overrides)

    print(run.get_portal_url())

    status = 'Preparing'
    while status not in ['Failed', 'Completed']:
        print(f'status: {status}')
        # get metrics
        metrics = run._core_run.get_metrics(recursive=True)
        if metrics == {}:
            pass
        else:
            run_metrics = list(metrics.values())

            new_metric = run_metrics[0]['eval_binary_error']
            
            if type(new_metric) == list:
                new_metric = new_metric[-1]
                
            print(f'eval_binary_error: {new_metric}')
            if 'old_metric' not in locals() or new_metric != old_metric:
                old_metric = new_metric
                tune.report(eval_binary_error=old_metric)
            else:
                pass
        time.sleep(2)
        status = run._core_run.get_status()

    print("The run is terminated.")
    print(status)

    return


def tune_pipeline(concurrent_run=1):
    start_time = time.time()
    # hyperparameter search space
    search_space = {
        "train_config.n_estimators": flaml.tune.randint(50, 200),
        "train_config.learning_rate": flaml.tune.uniform(0.01, 0.5),
    }

    hp_metric = "eval_binary_error"
    mode = "max"
    if concurrent_run > 1:
        import ray  # For parallel tuning

        ray.init(num_cpus=concurrent_run)
        use_ray = True
    else:
        use_ray = False

    # the working directory of the current AML job is '/mnt/azureml/cr/j/somerandomnumber/exe/wd/'
    # the wd contains the file included in the snapshot/code folder.
    # however the implementation in the run_with_config has the working direction as
    # local_dir + 'ray_results/trail_folder/'
    # need to pass the deepspeed_wd to find the correct file of deepspeed config.
    # tune_wd = os.getcwd()
    analysis = flaml.tune.run(
        run_with_config,
        config=search_space,
        metric=hp_metric,
        mode=mode,
        num_samples=2,  # number of trials
        use_ray=use_ray,
    )
    best_trial = analysis.get_best_trial(hp_metric, mode, "all")
    metric = best_trial.metric_analysis[hp_metric][mode]
    print(f"n_trials={len(analysis.trials)}")
    print(f"time={time.time()-start_time}")
    print(f"Best {hp_metric}: {metric:.4f}")
    print(f"Best coonfiguration: {best_trial.config}")


if __name__ == "__main__":
    tune_pipeline(concurrent_run=2)
    # for parallel tuning, pass concurrent_run > 1
