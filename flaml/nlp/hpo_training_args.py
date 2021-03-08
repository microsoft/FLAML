from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os, json

@dataclass
class HPOTrainingArguments(TrainingArguments):

    hpo_method: str = field(default= "", metadata={"help": "The hpo method."})
    task_name: str = field(default= "", metadata={"help": "The task name."})
    model_name_short: str = field(default="", metadata={"help": "The short model name."})
    scheduler_name: str = field(default="", metadata={"help": "The scheduler name."})
    submit_mode: str = field(default="", metadata={"help": "The submit mode."})
    folder_name: str = field(default="", metadata={"help": "The folder name for hpo."})
    CODE_PATH_REL: str = field(default="../../../", metadata={"help": "The folder name for hpo."})
    DATA_PATH_REL: str = field(default="../../../../data/", metadata={"help": "The folder name for hpo."})

    @property
    def folder_name(self):
        return self.hpo_method + "_" + self.scheduler_name + self.model_name_short + "_" + self.submit_mode



    def _init_path(self,
                 abs_data_path):

        self._search_space_dir = os.path.abspath(os.path.join(self.CODE_PATH_REL, "flaml/nlp/search_space/", self.task_name))
        self._output_dir = os.path.join(self.DATA_PATH_REL + "output/glue/", self.task_name)
        self._output_dir_abs = os.path.join(abs_data_path + "output/glue/", self.task_name)

        self._model_dir_abs = os.path.join(abs_data_path, "model", json.load(open(os.path.join(self._search_space_dir, "model_path.json"), "r"))[self.model_name_short])
        self._ckpt_dir_abs = os.path.join(self._output_dir_abs, "checkpoints/", self.folder_name + "/")