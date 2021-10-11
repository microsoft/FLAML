"""
    test suites for covering azure_utils.py
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
        "key_path": ".",
    }
    return autohf_settings


def test_get_configblob_from_partial_jobid():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return

    from flaml.nlp.result_analysis.azure_utils import JobID

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

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return

    from flaml.nlp.result_analysis.azure_utils import JobID

    args = get_autohf_setting()

    jobid_config = JobID(args)
    jobid_config.to_partial_jobid_string()
    JobID.convert_blobname_to_jobid("test")
    JobID.dataset_list_to_str("glue")
    JobID.get_full_data_name(["glue"], "mrpc")
    JobID._extract_model_type_with_keywords_match(
        "google/electra-base-discriminator:base"
    )

    jobid_config.to_wandb_string()


def set_autohf_setting(autohf, args):
    from flaml.nlp.utils import HPOArgs
    from flaml.nlp.result_analysis.azure_utils import JobID

    hpo_args = HPOArgs()
    autohf.custom_hpo_args = hpo_args.load_args("args", **args)
    autohf.jobid_config = JobID()
    autohf.jobid_config.set_jobid_from_console_args(console_args=autohf.custom_hpo_args)
    autohf._prepare_data()


def test_azureutils():
    try:
        import ray

        ray.shutdown()
        ray.init(local_mode=False)
    except ImportError:
        return

    from flaml.nlp.result_analysis.azure_utils import JobID, AzureUtils
    from flaml.nlp.result_analysis.azure_utils import ConfigScore, ConfigScoreList
    from flaml.nlp import AutoTransformers
    from datetime import datetime
    from flaml.nlp.utils import HPOArgs

    mnli_count = 0

    for subdat in ["mrpc", "rte", "stsb", "cola", "mnli", "mnli"]:
        if subdat == "mnli":
            mnli_count += 1

        autohf = AutoTransformers()
        args = get_autohf_setting()
        HPOArgs()
        set_autohf_setting(autohf, args)

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
            azure_key_path=args["key_path"],
            output_dir=args["output_dir"],
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
        azureutils.output_dir = None
        azureutils.write_autohf_output(
            configscore_list=[each_configscore],
            valid_metric={},
            predictions=predictions,
            duration=0,
        )

        azureutils.get_config_and_score_from_partial_jobid(
            root_log_path="data/", partial_jobid=autohf.jobid_config
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

        azureutils = AzureUtils(
            jobid_config_rename=None, autohf=None, jobid_config=autohf.jobid_config
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
