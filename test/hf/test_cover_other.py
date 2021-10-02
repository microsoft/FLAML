"""
    test suites for covering other functions
"""
""" Notice ray is required by flaml/nlp. The try except before each test function
 is for telling user to install flaml[nlp]. In future, if flaml/nlp contains a module that
 does not require ray, need to remove the try...except before the test functions and address
  import errors in the library code accordingly. """


def get_console_args():
    from flaml.nlp.utils import load_dft_args

    args = load_dft_args()
    args.dataset_subdataset_name = "glue:mrpc"
    args.algo_mode = "hpo"
    args.space_mode = "uni"
    args.search_alg_args_mode = "dft"
    args.algo_name = "bs"
    args.pruner = "None"
    args.pretrained_model_size = ["google/electra-base-discriminator", "base"]
    args.resplit_mode = "rspt"
    args.rep_id = 0
    args.seed_data = 43
    args.seed_transformers = 42
    return args


def model_init():
    from flaml.nlp import JobID

    jobid_config = JobID()
    jobid_config.set_unittest_config()
    from flaml.nlp import AutoTransformers

    autohf = AutoTransformers()

    preparedata_setting = get_preparedata_setting(jobid_config)
    autohf.prepare_data(**preparedata_setting)
    return autohf._load_model()


def get_preparedata_setting(jobid_config):
    preparedata_setting = {
        "server_name": "tmdev",
        "data_root_path": "data/",
        "max_seq_length": 128,
        "jobid_config": jobid_config,
        "resplit_portion": {
            "source": ["train", "validation"],
            "train": [0, 0.1],
            "validation": [0.1, 0.11],
            "test": [0.11, 0.12],
        },
    }
    return preparedata_setting


def test_dataprocess():
    """
    test to increase the coverage for flaml.nlp.dataprocess_auto
    """
    try:
        import ray
    except ImportError:
        return

    from flaml.nlp import AutoTransformers
    from flaml.nlp import JobID
    from flaml.nlp.dataset.dataprocess_auto import TOKENIZER_MAPPING

    jobid_config = JobID()
    jobid_config.set_unittest_config()
    autohf = AutoTransformers()

    dataset_name = JobID.dataset_list_to_str(jobid_config.dat)
    default_func = TOKENIZER_MAPPING[(dataset_name, jobid_config.subdat)]

    funcs_to_eval = set(
        [
            (dat, subdat)
            for (dat, subdat) in TOKENIZER_MAPPING.keys()
            if TOKENIZER_MAPPING[(dat, subdat)] != default_func
        ]
    )

    for (dat, subdat) in funcs_to_eval:
        print("loading dataset for {}, {}".format(dat, subdat))
        jobid_config.dat = dat.split(",")
        jobid_config.subdat = subdat

        preparedata_setting = get_preparedata_setting(jobid_config)
        autohf.prepare_data(**preparedata_setting)

        if subdat == "wic":
            jobid_config.pre_full = "xlnet-base-cased"
            jobid_config.pre = "xlnet"

            preparedata_setting = get_preparedata_setting(jobid_config)
            autohf.prepare_data(**preparedata_setting)


def test_gridsearch_space():
    try:
        import ray
    except ImportError:
        return

    from flaml.nlp.hpo.grid_searchspace_auto import (
        GRID_SEARCH_SPACE_MAPPING,
        AutoGridSearchSpace,
    )
    from flaml.nlp import JobID

    jobid_config = JobID()
    jobid_config.set_unittest_config()

    for each_model_type in GRID_SEARCH_SPACE_MAPPING.keys():
        AutoGridSearchSpace.from_model_and_dataset_name(
            each_model_type, "base", jobid_config.dat, jobid_config.subdat, "hpo"
        )


def test_hpo_space():
    try:
        import ray
    except ImportError:
        return

    from flaml.nlp.hpo.hpo_searchspace import (
        AutoHPOSearchSpace,
        HPO_SEARCH_SPACE_MAPPING,
    )
    from flaml.nlp import JobID

    jobid_config = JobID()
    jobid_config.set_unittest_config()

    for spa in HPO_SEARCH_SPACE_MAPPING.keys():
        jobid_config.spa = spa
        if jobid_config.spa == "cus":
            custom_hpo_args = {"hpo_space": {"learning_rate": [1e-5]}}
        else:
            custom_hpo_args = {}

        AutoHPOSearchSpace.from_model_and_dataset_name(
            jobid_config.spa,
            jobid_config.pre,
            jobid_config.presz,
            jobid_config.dat,
            jobid_config.subdat,
            **custom_hpo_args
        )


def test_trainer():
    try:
        import ray
    except ImportError:
        return
    from flaml.nlp.huggingface.trainer import TrainerForAutoTransformers

    num_train_epochs = 3
    num_train_examples = 100
    per_device_train_batch_size = 32
    device_count = 1
    max_steps = 1000
    warmup_steps = 100
    warmup_ratio = 0.1
    trainer = TrainerForAutoTransformers(model_init=model_init)
    trainer.convert_num_train_epochs_to_max_steps(
        num_train_epochs, num_train_examples, per_device_train_batch_size, device_count
    )
    trainer.convert_max_steps_to_num_train_epochs(
        max_steps, num_train_examples, per_device_train_batch_size, device_count
    )
    trainer.convert_warmup_ratio_to_warmup_steps(
        warmup_ratio,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        num_train_examples=num_train_examples,
        per_device_train_batch_size=per_device_train_batch_size,
        device_count=device_count,
    )
    trainer.convert_warmup_ratio_to_warmup_steps(
        warmup_ratio,
        max_steps=None,
        num_train_epochs=num_train_epochs,
        num_train_examples=num_train_examples,
        per_device_train_batch_size=per_device_train_batch_size,
        device_count=device_count,
    )
    trainer.convert_warmup_steps_to_warmup_ratio(
        warmup_steps,
        num_train_epochs,
        num_train_examples,
        per_device_train_batch_size,
        device_count,
    )


def test_switch_head():
    try:
        import ray
    except ImportError:
        return
    from transformers import AutoConfig

    from flaml.nlp.huggingface.switch_head_auto import (
        AutoSeqClassificationHead,
        MODEL_CLASSIFICATION_HEAD_MAPPING,
    )
    from flaml.nlp import JobID

    jobid_config = JobID()
    jobid_config.set_unittest_config()
    checkpoint_path = jobid_config.pre_full

    model_config = AutoConfig.from_pretrained(
        checkpoint_path,
        num_labels=AutoConfig.from_pretrained(checkpoint_path).num_labels,
    )

    for model in list(MODEL_CLASSIFICATION_HEAD_MAPPING.keys()):
        jobid_config.pre = model
        AutoSeqClassificationHead.from_model_type_and_config(
            jobid_config.pre, model_config
        )


def test_wandb_utils():
    try:
        import ray
    except ImportError:
        return

    from flaml.nlp.result_analysis.wandb_utils import WandbUtils
    from flaml.nlp import JobID
    import os

    args = get_console_args()
    args.key_path = "."
    jobid_config = JobID(args)

    wandb_utils = WandbUtils(
        is_wandb_on=True, wandb_key_path=args.key_path, jobid_config=jobid_config
    )
    os.environ["WANDB_MODE"] = "online"
    wandb_utils.wandb_group_name = "test"
    wandb_utils._get_next_trial_ids()
    wandb_utils.set_wandb_per_run()
    wandb_utils.set_wandb_per_trial()

    wandb_utils = WandbUtils(
        is_wandb_on=False, wandb_key_path=args.key_path, jobid_config=jobid_config
    )


def test_objective():
    try:
        import ray
    except ImportError:
        return

    from flaml.nlp import AutoTransformers
    from flaml.nlp import JobID
    from flaml.nlp import AzureUtils

    """
        test grid search
    """
    jobid_config = JobID()
    jobid_config.set_unittest_config()
    autohf = AutoTransformers()
    jobid_config.mod = "hpo"
    jobid_config.alg = "bs"
    jobid_config.spa = "gnr"
    jobid_config.pre = "bert"
    jobid_config.arg = "cus"
    autohf = AutoTransformers()

    preparedata_setting = get_preparedata_setting(jobid_config)
    autohf.prepare_data(**preparedata_setting)
    autohf._transformers_verbose = 10
    autohf._resources_per_trial = {"cpu": 1}
    autohf.ckpt_per_epoch = 1
    autohf._fp16 = True
    autohf.task_name = "seq-classification"

    autohf._objective(
        config={
            "learning_rate": 1e-5,
            "num_train_epochs": 0.01,
            "per_device_train_batch_size": 1,
            "warmup_ratio": 0,
            "weight_decay": 0,
            "seed": 42,
        },
    )


def test_search_algo_auto():
    try:
        import ray
    except ImportError:
        return

    from flaml.nlp import AutoTransformers
    from flaml.nlp import JobID
    from flaml.nlp.hpo.searchalgo_auto import AutoSearchAlgorithm

    jobid_config = JobID()
    jobid_config.set_unittest_config()

    _search_space_hpo = AutoTransformers._convert_dict_to_ray_tune_space(
        {
            "learning_rate": {"l": 1e-6, "u": 1e-3, "space": "log"},
            "num_train_epochs": {"l": 1.0, "u": 10.0, "space": "log"},
            "per_device_train_batch_size": [4, 8, 16, 32],
            "warmup_ratio": {"l": 0.0, "u": 0.3, "space": "linear"},
            "weight_decay": {"l": 0.0, "u": 0.3, "space": "linear"},
            "adam_epsilon": {"l": 1e-8, "u": 1e-6, "space": "linear"},
            "seed": [x for x in range(40, 45)],
        },
        mode="hpo",
    )

    AutoSearchAlgorithm.from_method_name(
        "bs",
        "dft",
        _search_space_hpo,
        1,
        "accuracy",
        "max",
        42,
    )

    AutoSearchAlgorithm.from_method_name(
        None,
        "dft",
        _search_space_hpo,
        1,
        "accuracy",
        "max",
        42,
    )

    AutoSearchAlgorithm.from_method_name(
        "optuna",
        "cus",
        _search_space_hpo,
        1,
        "accuracy",
        "max",
        42,
    )

    AutoSearchAlgorithm.from_method_name(
        "grid",
        "dft",
        _search_space_hpo,
        1,
        "accuracy",
        "max",
        42,
    )


# if __name__ == "__main__":
#     test_objective()
