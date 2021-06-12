from transformers import AutoConfig

from flaml.nlp.huggingface.trainer import TrainerForAutoTransformers


def get_console_args():
    from flaml.nlp.utils import load_console_args
    args = load_console_args()
    args.dataset_subdataset_name = "glue:mrpc"
    args.algo_mode = "hpo"
    args.space_mode = "uni"
    args.search_alg_args_mode = "dft"
    args.algo_name = "bs"
    args.pruner = "None"
    args.pretrained_model_size = "google/electra-base-discriminator:base"
    args.resplit_mode = "rspt"
    args.rep_id = 0
    args.seed_data = 43
    args.seed_transformers = 42
    return args


def model_init():
    from flaml.nlp.result_analysis.azure_utils import JobID, AzureUtils
    jobid_config = JobID()
    jobid_config.set_unittest_config()
    from flaml.nlp import AutoTransformers
    autohf = AutoTransformers()
    AzureUtils(root_log_path="logs_test/",
                             jobid=jobid_config, autohf=autohf)

    preparedata_setting = get_preparedata_setting(jobid_config)
    autohf.prepare_data(**preparedata_setting)
    return autohf._load_model()


def get_preparedata_setting(jobid_config):
    preparedata_setting = {
        "server_name": "tmdev",
        "data_root_path": "data/",
        "max_seq_length": 128,
        "jobid_config": jobid_config,
        "resplit_portion": {"source": ["train", "validation"],
                            "train": [0, 0.8],
                            "validation": [0.8, 0.9],
                            "test": [0.9, 1.0]}
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
    from flaml.nlp import AzureUtils
    from flaml.nlp.dataset.dataprocess_auto import TOKENIZER_MAPPING

    jobid_config = JobID()
    jobid_config.set_unittest_config()
    autohf = AutoTransformers()

    dataset_name = JobID.dataset_list_to_str(jobid_config.dat)
    default_func = TOKENIZER_MAPPING[(dataset_name, jobid_config.subdat)]

    funcs_to_eval = set([(dat, subdat) for (dat, subdat) in TOKENIZER_MAPPING.keys()
                         if TOKENIZER_MAPPING[(dat, subdat)] != default_func])

    for (dat, subdat) in funcs_to_eval:
        print("loading dataset for {}, {}".format(dat, subdat))
        jobid_config.dat = dat.split(",")
        jobid_config.subdat = subdat

        try:
            AzureUtils(root_log_path="logs_test/",
                                     jobid=jobid_config, autohf=autohf)

            preparedata_setting = get_preparedata_setting(jobid_config)
            autohf.prepare_data(**preparedata_setting)
        except Exception as err:
            print(err)
            pass


def test_gridsearch_space():
    from flaml.nlp.hpo.grid_searchspace_auto import GRID_SEARCH_SPACE_MAPPING, AutoGridSearchSpace
    from flaml.nlp.result_analysis.azure_utils import JobID
    jobid_config = JobID()
    jobid_config.set_unittest_config()

    for each_model_type in GRID_SEARCH_SPACE_MAPPING.keys():
        AutoGridSearchSpace.from_model_and_dataset_name(
            each_model_type,
            "base",
            jobid_config.dat,
            jobid_config.subdat, "hpo")


def test_hpo_space():
    from flaml.nlp.hpo.hpo_searchspace import AutoHPOSearchSpace, HPO_SEARCH_SPACE_MAPPING
    from flaml.nlp.result_analysis.azure_utils import JobID
    jobid_config = JobID()
    jobid_config.set_unittest_config()

    for spa in HPO_SEARCH_SPACE_MAPPING.keys():
        jobid_config.spa = spa
        if jobid_config.spa == "cus":
            custom_hpo_args = {"hpo_space": {"learning_rate": [1e-5]}}
        elif jobid_config.spa == "buni":
            best_config = {"learning_rate": 1e-5}
            custom_hpo_args = {"points_to_evaluate": [best_config],
                               "bound": {"learning_rate":
                                             {"u": best_config["learning_rate"]}}}
        else:
            custom_hpo_args = {}

        AutoHPOSearchSpace.from_model_and_dataset_name(jobid_config.spa,
                                                             jobid_config.pre,
                                                             jobid_config.presz,
                                                             jobid_config.dat,
                                                             jobid_config.subdat,
                                                             **custom_hpo_args)


def test_trainer():
    num_train_epochs = 3
    num_train_examples = 100
    per_device_train_batch_size = 32
    device_count = 1
    max_steps = 1000
    warmup_steps = 100
    warmup_ratio = 0.1
    trainer = TrainerForAutoTransformers(model_init=model_init)
    trainer.convert_num_train_epochs_to_max_steps(num_train_epochs,
                                                  num_train_examples,
                                                  per_device_train_batch_size,
                                                  device_count)
    trainer.convert_max_steps_to_num_train_epochs(max_steps,
                                                  num_train_examples,
                                                  per_device_train_batch_size,
                                                  device_count)
    trainer.convert_warmup_ratio_to_warmup_steps(warmup_ratio,
                                                 max_steps=max_steps,
                                                 num_train_epochs=num_train_epochs,
                                                 num_train_examples=num_train_examples,
                                                 per_device_train_batch_size=per_device_train_batch_size,
                                                 device_count=device_count)
    trainer.convert_warmup_steps_to_warmup_ratio(warmup_steps,
                                                 num_train_epochs,
                                                 num_train_examples,
                                                 per_device_train_batch_size,
                                                 device_count)


def test_switch_head():
    from flaml.nlp.huggingface.switch_head_auto import AutoSeqClassificationHead, MODEL_CLASSIFICATION_HEAD_MAPPING
    from flaml.nlp.result_analysis.azure_utils import JobID
    jobid_config = JobID()
    jobid_config.set_unittest_config()
    checkpoint_path = jobid_config.pre_full

    model_config = AutoConfig.from_pretrained(
        checkpoint_path,
        num_labels=AutoConfig.from_pretrained(checkpoint_path).num_labels)

    for model in list(MODEL_CLASSIFICATION_HEAD_MAPPING.keys()) + ["bert"]:
        try:
            jobid_config.pre = model
            AutoSeqClassificationHead \
                .from_model_type_and_config(jobid_config.pre,
                                            model_config)
        except ValueError:
            pass


def test_wandb_utils():
    from flaml.nlp.result_analysis.wandb_utils import WandbUtils
    from flaml.nlp.result_analysis.azure_utils import JobID
    import os

    args = get_console_args()
    args.key_path = "."
    jobid_config = JobID(args)

    try:
        wandb_utils = WandbUtils(is_wandb_on=True, console_args=args, jobid_config=jobid_config)
    except FileNotFoundError:
        wandb_utils = WandbUtils(is_wandb_on=False, console_args=args, jobid_config=jobid_config)
        pass
    os.environ["WANDB_MODE"] = "online"
    wandb_utils.wandb_group_name = "test"
    wandb_utils.set_wandb_per_trial()
    wandb_utils._get_next_trial_ids()
    wandb_utils.set_wandb_per_run()


if __name__ == "__main__":
    try:
        import ray
    except ImportError:
        import sys
        sys.exit(1)

    test_wandb_utils()
    test_dataprocess()
    test_gridsearch_space()
    test_hpo_space()
    test_trainer()
    test_switch_head()
