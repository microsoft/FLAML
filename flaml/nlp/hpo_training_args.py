from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os, json
from datasets import Dataset

@dataclass
class AutoHFArguments:

    CODE_PATH_REL: str = field(default="../../../", metadata={"help": "The folder name for hpo."})
    DATA_PATH_REL: str = field(default="../../../../data/", metadata={"help": "The folder name for hpo."})

    output_path: str = field(default ="", metadata={"output data path"})

    dataset_name: str = field(default="glue", metadata={"dataset name"})
    dataset_config: tuple = field(default=None, metadata={"dataset config"})

    #task_name: Optional[str] = field(default= "rte", metadata={"help": "The task name."})
    model_name: str = field(default="google/electra-base-discriminator", metadata={"help": "The short model name."})
    submit_mode: str = field(default="resplit", metadata={"help": "The submit mode."})

    scheduler_name: str = field(default="", metadata={"help": "The scheduler name."})
    #hpo_method: str = field(default= "", metadata={"help": "The hpo method."})

    num_labels: Optional[str] = field(default= 2, metadata={"help": "number of labels"})

    train_dataset: Dataset = field(default= None, metadata={"help": "train dataset"})
    eval_dataset: Dataset = field(default=None, metadata={"help": "dev dataset"})
    test_dataset: Dataset = field(default=None, metadata={"help": "test dataset"})

    @property
    def folder_name(self):
        return self.hpo_method + "_" + self.scheduler_name + "_" + self.model_name.replace("/", "_") + "_" + self.submit_mode

    @property
    def model_checkpoint(self):
        return os.path.join(self.output_path, "model", self.model_name)

    def init_path(self):
        search_space_dir = os.path.abspath(os.path.join(self.CODE_PATH_REL, "flaml/nlp/search_space/", self.task_name))
        self.search_space_grid = json.load(open(os.path.join(search_space_dir, self.task_name + "_grid.json", "r")))

        # self._output_dir = os.path.join(self.DATA_PATH_REL + "output/glue/", self.task_name)
        # self._output_dir_abs = os.path.join(abs_data_path + "output/glue/", self.task_name)
        #
        # self._model_dir_abs = os.path.join(abs_data_path, "model", json.load(open(os.path.join(self._search_space_dir, "model_path.json"), "r"))[self.model_name_short])
        # self._ckpt_dir_abs = os.path.join(self._output_dir_abs, "checkpoints/", self.folder_name + "/")