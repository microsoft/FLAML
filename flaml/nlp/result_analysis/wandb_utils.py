import os, shutil
from ..utils import get_wandb_azure_key
import subprocess, wandb
import hashlib
from time import time

class WandbUtils:
    def __init__(self,
                 autohf,
                 current_path):
        if os.path.exists(current_path + "/wandb"):
            shutil.rmtree(current_path + "/wandb")

        self.autohf = autohf
        wandb_key, azure_key, container_name = get_wandb_azure_key(current_path)
        subprocess.run(["wandb", "login", "--relogin", wandb_key])
        os.environ["WANDB_API_KEY"] = wandb_key

    def restart_wandb_group(self):
        wandb_group_name_str = self.autohf.full_dataset_name.lower() \
                               + "_" + self.autohf.model_type.lower() + "_" \
                               + self.autohf.model_size_type.lower()
        if hasattr(self.autohf, "search_algo_name"):
            wandb_group_name_str += "_" + self.autohf.search_algo_name.lower()
        if hasattr(self.autohf, "scheduler_name"):
            wandb_group_name_str += "_" + self.autohf.scheduler_name.lower()
        if hasattr(self.autohf, "hpo_searchspace_mode"):
            wandb_group_name_str += "_" + self.autohf.hpo_searchspace_mode.lower()
        wandb_group_name_str += "_" + self.autohf.path_utils.group_hash_id
        self.wandb_group_name = wandb_group_name_str

    def set_wandb_group_name(self,
                             parse_args):
        if parse_args.algo_mode == "eval_config_list":
            self.retrieve_wandb_group_name(parse_args)
        elif parse_args.algo_mode in ("hpo", "hpo_hf", "grid_search", "grid_search_bert"):
            self.restart_wandb_group()

    def retrieve_wandb_group_name(self, parse_args):


    def set_wandb_per_trial(self):
        print("before wandb.init\n\n\n")
        os.environ["WANDB_RUN_GROUP"] = self.wandb_group_name
        os.environ["WANDB_SILENT"] = "false"
        os.environ["WANDB_MODE"] = "online"
        return wandb.init(project = self.autohf.full_dataset_name,
                   group=self.wandb_group_name,
                   name= str(self._get_next_trial_ids()),
                   settings=wandb.Settings(
                   _disable_stats=True),
                   reinit=False)

    def _get_next_trial_ids(self):
        hash = hashlib.sha1()
        hash.update(str(time()).encode('utf-8'))
        return "trial_" + hash.hexdigest()[:3]

    def set_wandb_per_run(self):
        self.autohf.path_utils.group_hash_id = wandb.util.generate_id()
        # os.environ["WANDB_IGNORE_GLOBS"] = "*.json,*.csv,*.tmdev,*.pkl"
        os.environ["WANDB_RUN_GROUP"] = self.wandb_group_name
        os.environ["WANDB_SILENT"] = "false"
        os.environ["WANDB_MODE"] = "online"
        return wandb.init(project=self.autohf.full_dataset_name,
                   group= self.wandb_group_name,
                   settings=wandb.Settings(
                       _disable_stats=True),
                   reinit=False)

    def log_wandb(self, config, key_list):
        for each_hp in key_list:
            wandb.log({each_hp: config[each_hp]})

    def get_config_to_score(self):
        import wandb
        api = wandb.Api()
        full_dataset_name = self.autohf.full_dataset_name
        runs = api.runs('liususan/' + full_dataset_name, filters={"group": self.wandb_group_name})
        config2score = []
        for idx in range(0, len(runs)):
            run = runs[idx]
            try:
                this_eval_acc = run.summary['eval/' + self.autohf.metric_name]
                config2score.append((run.summary, self.autohf.metric_name, this_eval_acc, run.name))
            except KeyError:
                pass
        return config2score