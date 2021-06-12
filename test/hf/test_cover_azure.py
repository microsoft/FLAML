
def test_get_configblob_from_partial_jobid():
    from flaml.nlp.result_analysis.azure_utils import JobID
    each_blob_name = "dat=glue_subdat=cola_mod=grid_spa=cus_arg=dft_alg=grid" \
                     "_pru=None_pre=deberta_presz=large_spt=rspt_rep=0_sddt=43" \
                     "_sdhf=42_var1=1e-05_var2=0.0.json"
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
    from flaml.nlp.result_analysis.azure_utils import JobID
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

    jobid_config = JobID(args)
    jobid_config.to_partial_jobid_string()
    JobID.convert_blobname_to_jobid("test")
    JobID.dataset_list_to_str("glue")
    JobID.get_full_data_name(["glue"], "mrpc")
    JobID._extract_model_type_with_keywords_match("google/electra-base-discriminator:base")

if __name__ == "__main__":
    test_get_configblob_from_partial_jobid()
    test_jobid()