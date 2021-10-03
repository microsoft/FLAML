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
        "max_seq_length": 5,
        "jobid_config": jobid_config,
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
        "max_seq_length": 5,
        "jobid_config": jobid_config,
        "resplit_portion": {
            "source": ["train", "validation"],
            "train": [0, 0.00001],
            "validation": [0.00001, 0.00002],
            "test": [0.00002, 0.00003],
        },
        "foldnum": 2,
    }
    return preparedata_setting


def get_preparedata_setting_mnli(jobid_config):
    preparedata_setting = {
        "server_name": "tmdev",
        "data_root_path": "data/",
        "max_seq_length": 5,
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
    jobid_config.subdat = "mnli"
    preparedata_setting = get_preparedata_setting_mnli(jobid_config)
    autohf.prepare_data(**preparedata_setting)

    autohf_settings = get_autohf_settings()
    validation_metric, analysis = autohf.fit(**autohf_settings)
    autohf._load_model()
