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
    dataset_config: dict = field(default=None, metadata={"dataset config"})
    task_name: str = field(default="text-classification", metadata={"task name"})

    #task_name: Optional[str] = field(default= "rte", metadata={"help": "The task name."})
    model_name: list = field(default= ["electra", "base"], metadata={"help": "model name."})
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