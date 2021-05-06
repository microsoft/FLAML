'''Require: pip install torch transformers datasets wandb flaml[blendsearch,ray]
'''
#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
import argparse,json

from flaml.nlp.wandbazure.generate_result_summary import generate_result_csv
from flaml.nlp.wandbazure.utils import get_all_runs
from utils import get_wandb_azure_key
from run_autohf import dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--server_name', type=str, help='server name', required=True,
                            choices=["tmdev", "dgx", "azureml"])
    arg_parser.add_argument('--azure_key', type=str, help='azure key', required=False)
    args = arg_parser.parse_args()

    wandb_key, args.azure_key = get_wandb_azure_key()
    task2blobs, tasklist = get_all_runs(args)

    generate_result_csv(args, task2blobs, dataset_names, subdataset_names, search_algos, pretrained_models, scheduler_names, hpo_searchspace_modes, search_algo_args_modes)