"""Require: pip install torch transformers datasets wandb flaml[blendsearch,ray]
"""
global azure_log_path
global azure_key

""" Notice ray is required by flaml/nlp. The try except before each test function
is for telling user to install flaml[nlp]. In future, if flaml/nlp contains a module that
 does not require ray, need to remove the try...except before the test functions and address
  import errors in the library code accordingly. """


def get_preparedata_setting(jobid_config):
    preparedata_setting = {
        "server_name": "tmdev",
        "data_root_path": "data/",
        "max_seq_length": 10,
        "jobid_config": jobid_config,
        "load_config_mode": "jobid",
        "resplit_portion": {
            "source": ["train", "validation"],
            "train": [0, 0.001],
            "validation": [0.001, 0.002],
            "test": [0.002, 0.003],
        },
    }
    return preparedata_setting


def get_preparedata_setting_cv(jobid_config):
    preparedata_setting = {
        "server_name": "tmdev",
        "data_root_path": "data/",
        "max_seq_length": 10,
        "jobid_config": jobid_config,
        "load_config_mode": "jobid",
        "resplit_portion": {
            "source": ["train", "validation"],
            "train": [0, 0.001],
            "validation": [0.001, 0.002],
            "test": [0.002, 0.003],
        },
        "foldnum": 2,
    }
    return preparedata_setting


def get_preparedata_setting_mnli(jobid_config):
    preparedata_setting = {
        "server_name": "tmdev",
        "data_root_path": "data/",
        "max_seq_length": 5,
        "load_config_mode": "jobid",
        "jobid_config": jobid_config,
        "resplit_portion": {
            "source": ["train", "validation"],
            "train": [0, 0.0001],
            "validation": [0.0001, 0.00011],
            "test": [0.00011, 0.00012],
        },
        "fold_name": [
            "train",
            "validation_matched",
            "test_matched",
        ],
    }
    return preparedata_setting


def get_autohf_settings():
    autohf_settings = {
        "resources_per_trial": {"cpu": 1},
        "num_samples": 1,
        "time_budget": 100000,
        "ckpt_per_epoch": 1,
        "fp16": False,
    }
    return autohf_settings


def get_autohf_settings_grid():
    autohf_settings = {
        "resources_per_trial": {"cpu": 1},
        "num_samples": 1,
        "time_budget": 100000,
        "ckpt_per_epoch": 1,
        "fp16": False,
        "grid_search_space": "bert_test",
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
    from flaml.nlp import JobID

    """
        test grid search
    """
    jobid_config = JobID()
    jobid_config.set_unittest_config()
    jobid_config.subdat = "stsb"
    autohf = AutoTransformers()
    jobid_config.mod = "grid"
    jobid_config.alg = "grid"
    jobid_config.spa = "grid"
    jobid_config.spt = "rspt"
    autohf = AutoTransformers()

    preparedata_setting = get_preparedata_setting(jobid_config)
    autohf.prepare_data(**preparedata_setting)

    autohf_settings = get_autohf_settings_grid()
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
    from flaml.nlp import JobID
    from flaml.nlp import AzureUtils

    """
        test fold_name
    """
    jobid_config = JobID()
    jobid_config.set_dataset_and_model(
        dat=["glue"],
        subdat="mrpc",
        pre_full="google/electra-small-discriminator",
        pre="electra",
        presz="small",
    )
    autohf = AutoTransformers()
    jobid_config.subdat = "mnli"
    preparedata_setting = get_preparedata_setting_mnli(jobid_config)
    autohf.prepare_data(**preparedata_setting)

    autohf_settings = get_autohf_settings()
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
    from flaml.nlp import JobID
    from flaml.nlp import AzureUtils

    jobid_config = JobID()
    jobid_config.set_unittest_config()
    jobid_config.spt = "ori"
    jobid_config.subdat = "wnli"
    jobid_config.spa = "gnr_test"
    autohf = AutoTransformers()

    preparedata_setting = get_preparedata_setting(jobid_config)
    autohf.prepare_data(**preparedata_setting)

    autohf_settings = get_autohf_settings()
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
            root_log_path="logs_test/", data_root_dir="data/", autohf=autohf
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
    from flaml.nlp import JobID
    from flaml.nlp import AzureUtils

    jobid_config = JobID()
    jobid_config.set_unittest_config()
    autohf = AutoTransformers()

    preparedata_setting = get_preparedata_setting(jobid_config)
    autohf.prepare_data(**preparedata_setting)

    autohf_settings = get_autohf_settings()
    autohf_settings["points_to_evaluate"] = [
        {"learning_rate": 2e-5, "per_device_train_batch_size": 1}
    ]
    validation_metric, analysis = autohf.fit(**autohf_settings)

    if validation_metric is not None:
        predictions, test_metric = autohf.predict()
        if test_metric:
            validation_metric.update({"test": test_metric})

        azure_utils = AzureUtils(
            root_log_path="logs_test/", data_root_dir="data/", autohf=autohf
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
    from flaml.nlp import JobID

    jobid_config = JobID()
    jobid_config.set_unittest_config()
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
    from flaml.nlp import JobID
    from flaml.nlp import AzureUtils

    """
        test fold_name
    """
    jobid_config = JobID()
    jobid_config.set_unittest_config()
    jobid_config.reset_pre_full("google/electra-small-discriminator")
    autohf = AutoTransformers()
    jobid_config.subdat = "cola"
    preparedata_setting = get_preparedata_setting(jobid_config)
    autohf.prepare_data(**preparedata_setting)

    autohf_settings = get_autohf_settings()
    validation_metric, analysis = autohf.fit(**autohf_settings)


def test_cv():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return
    from flaml.nlp import AutoTransformers
    from flaml.nlp import JobID
    from flaml.nlp import AzureUtils

    """
        test cv
    """
    jobid_config = JobID()
    jobid_config.set_unittest_config()
    # jobid_config.dat = ["hyperpartisan_news_detection"]
    # jobid_config.subdat = "bypublisher"
    autohf = AutoTransformers()
    jobid_config.spt = "cv"
    preparedata_setting = get_preparedata_setting_cv(jobid_config)
    autohf.prepare_data(**preparedata_setting)

    autohf.eval_dataset = autohf.eval_datasets[0]
    autohf.train_dataset = autohf.train_datasets[0]

    autohf_settings = get_autohf_settings()
    validation_metric, analysis = autohf.fit(**autohf_settings)
