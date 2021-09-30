"""
    test suites for covering azure_utils.py
"""


def get_preparedata_setting(jobid_config):
    preparedata_setting = {
        "server_name": "tmdev",
        "data_root_path": "data/",
        "max_seq_length": 128,
        "jobid_config": jobid_config,
        "resplit_portion": {
            "source": ["train", "validation"],
            "train": [0, 0.01],
            "validation": [0.01, 0.011],
            "test": [0.011, 0.012],
        },
    }
    return preparedata_setting


def get_preparedata_setting_mnli_1(jobid_config):
    preparedata_setting = {
        "server_name": "tmdev",
        "data_root_path": "data/",
        "max_seq_length": 128,
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


def get_preparedata_setting_mnli_2(jobid_config):
    preparedata_setting = {
        "server_name": "tmdev",
        "data_root_path": "data/",
        "max_seq_length": 128,
        "jobid_config": jobid_config,
        "resplit_portion": {
            "source": ["train", "validation"],
            "train": [0, 0.0001],
            "validation": [0.0001, 0.00011],
            "test": [0.00011, 0.00012],
        },
        "fold_name": [
            "train",
            "validation_mismatched",
            "test_mismatched",
        ],
    }
    return preparedata_setting


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


def test_get_configblob_from_partial_jobid():
    try:
        import ray
    except ImportError:
        return

    from flaml.nlp import JobID

    each_blob_name = (
        "dat=glue_subdat=cola_mod=grid_spa=cus_arg=dft_alg=grid"
        "_pru=None_pre_full=roberta-base_presz=large_spt=rspt_rep=0_sddt=43"
        "_sdhf=42_sdbs=20.json"
    )

    partial_jobid = JobID()
    partial_jobid.pre = "deberta"
    partial_jobid.mod = "grid"
    partial_jobid.spa = "cus"
    partial_jobid.presz = "large"

    each_jobconfig = JobID.convert_blobname_to_jobid(each_blob_name)
    each_jobconfig.is_match(partial_jobid)

    partial_jobid = JobID()
    partial_jobid.pre = "deberta"
    partial_jobid.mod = "hpo"
    partial_jobid.spa = "cus"
    partial_jobid.presz = "large"
    partial_jobid.sddt = None

    each_jobconfig = JobID.convert_blobname_to_jobid(each_blob_name)
    each_jobconfig.is_match(partial_jobid)


def test_jobid():
    try:
        import ray
    except ImportError:
        return

    from flaml.nlp import JobID

    args = get_console_args()

    jobid_config = JobID(args)
    jobid_config.to_partial_jobid_string()
    JobID.convert_blobname_to_jobid("test")
    JobID.dataset_list_to_str("glue")
    JobID.get_full_data_name(["glue"], "mrpc")
    JobID._extract_model_type_with_keywords_match(
        "google/electra-base-discriminator:base"
    )

    jobid_config.to_wandb_string()


def test_azureutils():
    try:
        import ray
    except ImportError:
        return

    from flaml.nlp import AzureUtils, JobID
    from flaml.nlp.result_analysis.azure_utils import ConfigScore, ConfigScoreList
    from flaml.nlp import AutoTransformers
    from datetime import datetime

    mnli_count = 0

    for subdat in ["mrpc", "rte", "stsb", "cola", "mnli", "mnli"]:
        if subdat == "mnli":
            mnli_count += 1
        args = get_console_args()
        args.key_path = "."
        jobid_config = JobID(args)
        jobid_config.subdat = subdat

        autohf = AutoTransformers()
        autohf.jobid_config = jobid_config

        if subdat == "mnli":
            if mnli_count == 1:
                preparedata_setting = get_preparedata_setting_mnli_1(jobid_config)
            else:
                preparedata_setting = get_preparedata_setting_mnli_2(jobid_config)
        else:
            preparedata_setting = get_preparedata_setting(jobid_config)
        autohf.prepare_data(**preparedata_setting)

        each_configscore = ConfigScore(
            trial_id="test",
            start_time=0.0,
            last_update_time=0.0,
            config={},
            metric_score={"max": 0.0},
            time_stamp=0.0,
        )
        configscore_list = ConfigScoreList([each_configscore])
        for each_method in ["unsorted", "sort_time", "sort_accuracy"]:
            configscore_list.sorted(each_method)
        configscore_list.get_best_config()

        azureutils = AzureUtils(
            azure_key_path=args.key_path,
            data_root_dir=args.data_root_dir,
            autohf=autohf,
        )
        azureutils.download_azure_blob(
            "test/dat=glue_subdat=cola_mod=grid_spa=cus_arg=dft_alg=grid"
            "_pru=None_pre_full=roberta-base_presz=large_spt=rspt_rep=0_sddt=43"
            "_sdhf=42_sdbs=20.json"
        )
        azureutils.autohf = autohf
        azureutils.root_log_path = "logs_azure/"

        if subdat == "stsb":
            predictions = [5.1]
        elif subdat in ("rte", "qnli", "mnli"):
            predictions = [0]
        elif subdat == "cola":
            predictions = [0.1]
        else:
            predictions = ["0"]

        azureutils.write_autohf_output(
            configscore_list=[each_configscore],
            valid_metric={},
            predictions=predictions,
            duration=0,
        )
        azureutils.data_root_dir = None
        azureutils.write_autohf_output(
            configscore_list=[each_configscore],
            valid_metric={},
            predictions=predictions,
            duration=0,
        )

        azureutils.get_config_and_score_from_partial_jobid(
            root_log_path="data/", partial_jobid=jobid_config
        )

        this_blob = type("", (), {})()
        import pytz

        utc = pytz.UTC
        setattr(this_blob, "last_modified", utc.localize(datetime.now()))
        azureutils.is_after_earliest_time(this_blob, (2008, 8, 8))

        setattr(
            this_blob,
            "name",
            "test/dat=glue_subdat=cola_mod=grid_spa=cus_arg=dft_alg=grid"
            "_pru=None_pre_full=roberta-base_presz=large_spt=rspt_rep=0_sddt=43_sdhf=42_sdbs=20.json",
        )

        matched_blob_list = [(None, this_blob)]
        azureutils.get_config_and_score_from_matched_blob_list(matched_blob_list)

        from flaml.nlp.utils import load_dft_args

        jobid_config.mod = "grid"
        console_args = load_dft_args()
        console_args.algo_mode = "grid"
        jobid_config.set_jobid_from_console_args(console_args)

        azureutils = AzureUtils(
            jobid_config_rename=None, autohf=None, jobid_config=jobid_config
        )

        setattr(
            this_blob,
            "name",
            "test/dat=glue_subdat=cola_mod=grid_spa=cus_arg=dft_alg=grid"
            "_pru=None_pre_full=roberta-base_presz=large_spt=rspt_rep=0_sddt=43_sdhf=42_sdbs=20.json",
        )

        each_jobconfig = JobID.convert_blobname_to_jobid(this_blob.name)
        jobid_config = each_jobconfig
        AzureUtils.append_blob_list(
            this_blob,
            root_log_path="",
            partial_jobid=jobid_config,
            earliest_time=(2088, 8, 8),
            blob_list=[],
        )


if __name__ == "__main__":
    test_azureutils()
