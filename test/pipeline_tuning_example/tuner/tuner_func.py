import time
import flaml
import submit_train_pipeline
import logging
from ray import tune

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


def run_with_config(config: dict):
    """Run the pipeline with a given config dict
    """

    overrides = [f"{key}={value}" for key, value in config.items()]

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
