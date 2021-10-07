"""
    test suites for covering other functions
"""
""" Notice ray is required by flaml/nlp. The try except before each test function
 is for telling user to install flaml[nlp]. In future, if flaml/nlp contains a module that
 does not require ray, need to remove the try...except before the test functions and address
  import errors in the library code accordingly. """


def get_autohf_setting():
    autohf_settings = {
        "output_dir": "data/input/",
        "dataset_config": ["glue", "mrpc"],
        "space_mode": "uni",
        "search_alg_args_mode": "dft",
        "model_path": "google/electra-base-discriminator",
        "model_size": "base",
        "key_path": ".",
        "resplit_portion": [0, 0.1, 0.1, 0.11, 0.11, 0.12],
    }
    return autohf_settings


def model_init():
    from flaml.nlp.utils import HPOArgs
    from flaml.nlp import AutoTransformers

    autohf = AutoTransformers()
    args = get_autohf_setting()
    HPOArgs()

    set_autohf_setting(autohf, args)
    autohf._num_labels = 2
    return autohf._load_model()


def test_dataprocess():
    """
    test to increase the coverage for flaml.nlp.dataprocess_auto
    """
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return

    from flaml.nlp import AutoTransformers
    from flaml.nlp.dataset.dataprocess_auto import TOKENIZER_MAPPING

    autohf = AutoTransformers()
    default_func = TOKENIZER_MAPPING[("glue", "mrpc")]

    funcs_to_eval = set(
        [
            (dat, subdat)
            for (dat, subdat) in TOKENIZER_MAPPING.keys()
            if TOKENIZER_MAPPING[(dat, subdat)] != default_func
        ]
    )

    for (dat, subdat) in funcs_to_eval:
        args = get_autohf_setting()
        args["dataset_config"] = [dat, subdat]
        set_autohf_setting(autohf, args)

        if subdat == "wic":
            args["model_path"] = "xlnet-base-cased"
            set_autohf_setting(autohf, args)


def test_gridsearch_space():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return

    from flaml.nlp.hpo.grid_searchspace_auto import (
        GRID_SEARCH_SPACE_MAPPING,
        AutoGridSearchSpace,
    )

    for each_model_type in GRID_SEARCH_SPACE_MAPPING.keys():
        AutoGridSearchSpace.from_model_and_dataset_name(
            each_model_type, "base", ["glue"], "mrpc", "hpo"
        )


def test_hpo_space():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return

    from flaml.nlp.hpo.hpo_searchspace import (
        AutoHPOSearchSpace,
        HPO_SEARCH_SPACE_MAPPING,
    )
    from flaml.nlp.utils import HPOArgs
    from flaml.nlp import AutoTransformers

    autohf = AutoTransformers()

    args = get_autohf_setting()
    HPOArgs()

    set_autohf_setting(autohf, args)

    for spa in HPO_SEARCH_SPACE_MAPPING.keys():
        args["space_mode"] = spa
        if spa == "cus":
            custom_search_space = {"learning_rate": [1e-5]}
        else:
            custom_search_space = {}

        AutoHPOSearchSpace.from_model_and_dataset_name(
            autohf.jobid_config.spa,
            autohf.jobid_config.pre,
            autohf.jobid_config.presz,
            autohf.jobid_config.dat,
            autohf.jobid_config.subdat,
            autohf.jobid_config.mod,
            custom_search_space=custom_search_space,
        )


def test_trainer():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
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

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return
    from transformers import AutoConfig

    from flaml.nlp.huggingface.switch_head_auto import (
        AutoSeqClassificationHead,
        MODEL_CLASSIFICATION_HEAD_MAPPING,
    )
    from flaml.nlp import AutoTransformers
    from flaml.nlp.utils import HPOArgs

    autohf = AutoTransformers()
    args = get_autohf_setting()
    HPOArgs()

    set_autohf_setting(autohf, args)
    checkpoint_path = autohf.jobid_config.pre_full

    model_config = AutoConfig.from_pretrained(
        checkpoint_path,
        num_labels=AutoConfig.from_pretrained(checkpoint_path).num_labels,
    )

    for model in list(MODEL_CLASSIFICATION_HEAD_MAPPING.keys()):
        autohf.jobid_config.pre = model
        AutoSeqClassificationHead.from_model_type_and_config(
            autohf.jobid_config.pre, model_config
        )


def test_wandb_utils():
    try:
        import ray
        import wandb

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return

    from flaml.nlp.result_analysis.wandb_utils import WandbUtils
    from flaml.nlp.result_analysis.azure_utils import JobID
    import os

    args = get_autohf_setting()
    jobid_config = JobID(args)

    wandb_utils = WandbUtils(
        is_wandb_on=True, wandb_key_path=args["key_path"], jobid_config=jobid_config
    )
    os.environ["WANDB_MODE"] = "online"
    wandb_utils.wandb_group_name = "test"
    wandb_utils._get_next_trial_ids()
    try:
        wandb_utils.set_wandb_per_run()
        wandb_utils.set_wandb_per_trial()
    except wandb.errors.UsageError:
        print(
            "wandb.errors.UsageError: api_key not configured (no-tty).  Run wandb login"
        )

    WandbUtils(
        is_wandb_on=False, wandb_key_path=args["key_path"], jobid_config=jobid_config
    )


def set_autohf_setting(autohf, args):
    from flaml.nlp.utils import HPOArgs
    from flaml.nlp.result_analysis.azure_utils import JobID

    hpo_args = HPOArgs()
    autohf.custom_hpo_args = hpo_args.load_args("args", **args)
    autohf.jobid_config = JobID()
    autohf.jobid_config.set_jobid_from_console_args(console_args=autohf.custom_hpo_args)
    autohf._prepare_data()


def test_objective():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return

    from flaml.nlp import AutoTransformers
    from flaml.nlp.utils import HPOArgs

    """
        test grid search
    """
    autohf = AutoTransformers()
    args = get_autohf_setting()
    HPOArgs()

    set_autohf_setting(autohf, args)
    autohf._num_labels = 2

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

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return

    from flaml.nlp import AutoTransformers
    from flaml.nlp.hpo.searchalgo_auto import AutoSearchAlgorithm

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


if __name__ == "__main__":
    test_search_algo_auto()
