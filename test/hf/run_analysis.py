'''Require: pip install torch transformers datasets wandb flaml[blendsearch,ray]
'''
#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
import argparse,os
import subprocess
from flaml.nlp.result_analysis.azure_utils import JobID

def create_partial_config_bestnn():
    jobid_config = JobID()
    jobid_config.mod = "bestnn"
    jobid_config.spa = "cus"
    return jobid_config

def create_partial_config_list():
    jobid_config = JobID()
    jobid_config.mod = "list"
    jobid_config.spa = "uni"
    jobid_config.presz = "xlarge"
    return jobid_config

def create_partial_config_hpo():
    jobid_config = JobID()
    jobid_config.mod = "hpo"
    jobid_config.spa = "uni"
    return jobid_config

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', type=str, help='analysis mode', required=True, choices=["summary", "analysis", "plot", "extract"])
    arg_parser.add_argument('--key_path', type=str, help='key path', required=False, default = "../../")
    args = arg_parser.parse_args()

    if args.mode == "extract":
        partial_config_list = {"hpo":  create_partial_config_hpo(),
                               "list": create_partial_config_list(),
                               "bestnn": create_partial_config_bestnn()}
        from flaml.nlp.result_analysis.generate_result_summary import extract_ranked_config_score
        extract_ranked_config_score(args, partial_config_list)