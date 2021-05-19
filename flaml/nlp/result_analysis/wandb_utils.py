import os, shutil
from ..utils import get_wandb_azure_key
import subprocess, wandb
import hashlib
from time import time

class WandbUtils:
    def __init__(self,
                 console_args,
                 jobid_config):
        wandb_key, azure_key, container_name = get_wandb_azure_key(console_args.key_path)
        subprocess.run(["wandb", "login", "--relogin", wandb_key])
        os.environ["WANDB_API_KEY"] = wandb_key
        self.jobid_config = jobid_config

    def set_wandb_per_trial(self):
        print("before wandb.init\n\n\n")
        os.environ["WANDB_RUN_GROUP"] = self.jobid_config.to_wandb_string()
        os.environ["WANDB_SILENT"] = "false"
        os.environ["WANDB_MODE"] = "online"
        return wandb.init(project = self.jobid_config.get_jobid_full_data_name(),
                   group= os.environ["WANDB_RUN_GROUP"],
                   name= str(self._get_next_trial_ids()),
                   settings=wandb.Settings(
                   _disable_stats=True),
                   reinit=False)

    def _get_next_trial_ids(self):
        hash = hashlib.sha1()
        hash.update(str(time()).encode('utf-8'))
        return "trial_" + hash.hexdigest()[:3]

    def set_wandb_per_run(self):
        os.environ["WANDB_RUN_GROUP"] = self.jobid_config.to_wandb_string() + wandb.util.generate_id()
        os.environ["WANDB_SILENT"] = "false"
        os.environ["WANDB_MODE"] = "online"
        return wandb.init(project= self.jobid_config.get_jobid_full_data_name(),
                   group= os.environ["WANDB_RUN_GROUP"],
                   settings=wandb.Settings(
                       _disable_stats=True),
                   reinit=False)