import os, shutil
from ..utils import get_wandb_azure_key
import subprocess, wandb
import hashlib
from time import time

class WandbUtils:
    def __init__(self,
                 is_wandb_on = None,
                 console_args = None,
                 jobid_config = None):
        wandb_key, azure_key, container_name = get_wandb_azure_key(console_args.key_path)
        if is_wandb_on == True:
            subprocess.run(["wandb", "login", "--relogin", wandb_key])
            os.environ["WANDB_API_KEY"] = wandb_key
            os.environ["WANDB_MODE"] = "online"
        else:
            os.environ["WANDB_MODE"] = "offline"
        self.jobid_config = jobid_config

    def set_wandb_per_trial(self):
        print("before wandb.init\n\n\n")
        if os.environ["WANDB_MODE"] == "online":
            os.environ["WANDB_SILENT"] = "false"
            return wandb.init(project = self.jobid_config.get_jobid_full_data_name(),
                       group= self.wandb_group_name,
                       name= str(self._get_next_trial_ids()),
                       settings=wandb.Settings(
                       _disable_stats=True),
                       reinit=False)
        else:
            return None

    def _get_next_trial_ids(self):
        hash = hashlib.sha1()
        hash.update(str(time()).encode('utf-8'))
        return "trial_" + hash.hexdigest()[:3]

    def set_wandb_per_run(self):
        os.environ["WANDB_RUN_GROUP"] = self.jobid_config.to_wandb_string() + wandb.util.generate_id()
        self.wandb_group_name = os.environ["WANDB_RUN_GROUP"]
        if os.environ["WANDB_MODE"] == "online":
            os.environ["WANDB_SILENT"] = "false"
            return wandb.init(project= self.jobid_config.get_jobid_full_data_name(),
                       group= os.environ["WANDB_RUN_GROUP"],
                       settings=wandb.Settings(
                           _disable_stats=True),
                       reinit=False)
        else:
            return None