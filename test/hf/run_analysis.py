'''Require: pip install torch transformers datasets wandb flaml[blendsearch,ray]
'''
#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
import argparse,os
import subprocess

from flaml.nlp import generate_result_csv
from flaml.nlp import plot_walltime_curve
from flaml.nlp import get_all_azure_uploaded_files
from utils import get_wandb_azure_key
from run_autohf import dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes, resplit_modes

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--server_name', type=str, help='server name', required=True,
                            choices=["tmdev", "dgx", "azureml"])
    arg_parser.add_argument('--azure_key', type=str, help='azure key', required=False)
    arg_parser.add_argument('--mode', type=str, help='analysis mode', required=True, choices=["summary", "analysis", "plot"])
    arg_parser.add_argument('--key_path', type=str, help='path for storing key.json', required=False, default = os.path.abspath("../../"))
    args = arg_parser.parse_args()

    args.wandb_key, args.azure_key = get_wandb_azure_key(args.key_path)
    task2blobs, bloblist = get_all_azure_uploaded_files(args)

    if args.mode == "analysis":
        from flaml.nlp import analysis_model_size
        analysis_model_size(args, task2blobs, dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes, resplit_modes)
    elif args.mode == "summary":
        generate_result_csv(args, bloblist, dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes, resplit_modes)
    elif args.mode == "plot":
        plot_walltime_curve(args)