"""Require: pip install torch transformers datasets wandb flaml[blendsearch,ray]
"""
global azure_log_path
global azure_key

""" Notice ray is required by flaml/nlp. The try except before each test function
is for telling user to install flaml[nlp]. In future, if flaml/nlp contains a module that
 does not require ray, need to remove the try...except before the test functions and address
  import errors in the library code accordingly. """


def get_autohf_settings():
    autohf_settings = {
        "output_dir": "data/input/",
        "max_seq_length": 10,
        "load_config_mode": "args",
        "source_fold": ["train", "validation"],
        "split_portion": [0, 0.001, 0.001, 0.002, 0.002, 0.003],
        "resources_per_trial": {"cpu": 1},
        "sample_num": 1,
        "time_budget": 100000,
        "ckpt_per_epoch": 1,
        "fp16": False,
    }
    return autohf_settings


def get_autohf_settings_cv():
    autohf_settings = {
        "output_dir": "data/input/",
        "max_seq_length": 10,
        "load_config_mode": "args",
        "source_fold": ["train", "validation"],
        "split_portion": [0, 0.001, 0.001, 0.002, 0.002, 0.003],
        "resources_per_trial": {"cpu": 1},
        "sample_num": 1,
        "time_budget": 100000,
        "ckpt_per_epoch": 1,
        "fp16": False,
        "cv_k": 2,
    }
    return autohf_settings


def get_autohf_settings_grid():
    autohf_settings = {
        "output_dir": "data/input/",
        "max_seq_length": 10,
        "load_config_mode": "args",
        "source_fold": ["train", "validation"],
        "split_portion": [0, 0.001, 0.001, 0.002, 0.002, 0.003],
        "resources_per_trial": {"cpu": 1},
        "sample_num": 1,
        "time_budget": 100000,
        "ckpt_per_epoch": 1,
        "fp16": False,
        "grid_space_model_type": "bert_test",
    }
    return autohf_settings


def get_autohf_settings_mnli():
    autohf_settings = {
        "output_dir": "data/input/",
        "max_seq_length": 5,
        "load_config_mode": "args",
        "source_fold": ["train", "validation"],
        "fold_names": ["train", "validation_matched", "test_matched"],
        "split_portion": [0, 0.0001, 0.0001, 0.00011, 0.00011, 0.00012],
        "resources_per_trial": {"cpu": 1},
        "sample_num": 1,
        "time_budget": 100000,
        "ckpt_per_epoch": 1,
        "fp16": False,
    }
    return autohf_settings


def test_hpo_grid():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return

    from flaml.nlp import AutoTransformers
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp.utils import HPOArgs

    """
        test grid search
    """
    autohf = AutoTransformers()
    autohf_settings = get_autohf_settings_grid()
    unittest_setting = HPOArgs._get_unittest_config()
    for key, val in unittest_setting.items():
        autohf_settings[key] = val
    autohf_settings["dataset_config"] = ["glue", "stsb"]
    autohf_settings["algo_mode"] = "grid"
    autohf_settings["algo_name"] = "grid"
    autohf_settings["space_mode"] = "grid"
    autohf_settings["resplit_mode"] = "rspt"

    validation_metric, analysis = autohf.fit(**autohf_settings)
    autohf._load_model()


def test_foldname():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return

    from flaml.nlp import AutoTransformers
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp.utils import HPOArgs

    """
        test fold_name
    """
    autohf = AutoTransformers()

    autohf_settings = get_autohf_settings_mnli()
    unittest_setting = HPOArgs._get_unittest_config()
    for key, val in unittest_setting.items():
        autohf_settings[key] = val
    autohf_settings["model_path"] = "google/electra-small-discriminator"
    autohf_settings["model_size"] = "small"
    autohf_settings["dataset_config"] = ["glue", "mnli"]

    validation_metric, analysis = autohf.fit(**autohf_settings)
    autohf._load_model()


def test_hpo_ori():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return
    from flaml.nlp import AutoTransformers
    from flaml.nlp.result_analysis.azure_utils import JobID, AzureUtils
    from flaml.nlp.utils import HPOArgs

    autohf = AutoTransformers()
    autohf_settings = get_autohf_settings()
    unittest_setting = HPOArgs._get_unittest_config()
    for key, val in unittest_setting.items():
        autohf_settings[key] = val
    autohf_settings["resplit_mode"] = "ori"
    autohf_settings["dataset_config"] = ["glue", "wnli"]
    autohf_settings["space_mode"] = "gnr_test"
    autohf_settings["points_to_evaluate"] = [
        {
            "learning_rate": 2e-5,
            "num_train_epochs": 0.005,
            "per_device_train_batch_size": 1,
        }
    ]
    validation_metric, analysis = autohf.fit(**autohf_settings)

    if validation_metric is not None:

        predictions, test_metric = autohf.predict()
        if test_metric:
            validation_metric.update({"test": test_metric})

        azure_utils = AzureUtils(
            root_log_path="logs_test/", output_dir="data/", autohf=autohf
        )
        azure_utils._azure_key = "test"
        azure_utils._container_name = "test"

        configscore_list = azure_utils.extract_configscore_list_from_analysis(analysis)
        azure_utils.write_autohf_output(
            configscore_list=configscore_list,
            valid_metric=validation_metric,
            predictions=predictions,
            duration=autohf.last_run_duration,
        )


def test_hpo():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return
    from flaml.nlp import AutoTransformers
    from flaml.nlp.result_analysis.azure_utils import JobID, AzureUtils
    from flaml.nlp.utils import HPOArgs
    import os

    autohf = AutoTransformers()

    autohf_settings = get_autohf_settings()
    unittest_setting = HPOArgs._get_unittest_config()
    for key, val in unittest_setting.items():
        autohf_settings[key] = val

    autohf_settings["points_to_evaluate"] = [
        {"learning_rate": 2e-5, "per_device_train_batch_size": 1}
    ]
    # autohf_settings["is_wandb_on"] = True
    # autohf_settings["key_path"] = "."
    # os.environ["WANDB_MODE"] = "online"

    validation_metric, analysis = autohf.fit(**autohf_settings)

    if validation_metric is not None:
        predictions, test_metric = autohf.predict()
        if test_metric:
            validation_metric.update({"test": test_metric})

        azure_utils = AzureUtils(
            root_log_path="logs_test/", output_dir="data/", autohf=autohf
        )
        azure_utils._azure_key = "test"
        azure_utils._container_name = "test"

        configscore_list = azure_utils.extract_configscore_list_from_analysis(analysis)
        azure_utils.write_autohf_output(
            configscore_list=configscore_list,
            valid_metric=validation_metric,
            predictions=predictions,
            duration=autohf.last_run_duration,
        )


def test_transformers_verbosity():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return
    import transformers
    from flaml.nlp import AutoTransformers

    autohf = AutoTransformers()

    for verbose in [
        transformers.logging.ERROR,
        transformers.logging.WARNING,
        transformers.logging.INFO,
        transformers.logging.DEBUG,
    ]:
        autohf._set_transformers_verbosity(verbose)


def test_one_sentence_key():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return
    from flaml.nlp import AutoTransformers
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp.utils import HPOArgs

    """
        test fold_name
    """
    autohf = AutoTransformers()
    autohf_settings = get_autohf_settings()
    unittest_setting = HPOArgs._get_unittest_config()
    for key, val in unittest_setting.items():
        autohf_settings[key] = val
    autohf_settings["model_path"] = "google/electra-small-discriminator"
    autohf_settings["model_size"] = "small"
    autohf_settings["dataset_config"] = ["glue", "cola"]

    validation_metric, analysis = autohf.fit(**autohf_settings)


def test_cv():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return
    from flaml.nlp import AutoTransformers
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp.utils import HPOArgs

    """
        test cv
    """
    autohf = AutoTransformers()
    autohf_settings = get_autohf_settings_cv()
    unittest_setting = HPOArgs._get_unittest_config()
    for key, val in unittest_setting.items():
        autohf_settings[key] = val
    autohf_settings["resplit_mode"] = "cv"

    validation_metric, analysis = autohf.fit(**autohf_settings)


if __name__ == "__main__":
    test_hpo()
