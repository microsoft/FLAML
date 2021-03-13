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
    subdataset_name: str = field(default="glue", metadata={"sub dataset name"})
    dataset_config: dict = field(default=None, metadata={"dataset config"})
    task_name: str = field(default="text-classification", metadata={"task name"})

    #task_name: Optional[str] = field(default= "rte", metadata={"help": "The task name."})
    model_name: str = field(default= "google/electra-base-discriminator", metadata={"help": "model name."})
    model_type: str = field(default="electra", metadata={"help": "model type."})
    submit_mode: str = field(default="resplit", metadata={"help": "The submit mode."})

    scheduler_name: str = field(default="asha", metadata={"help": "The scheduler name."})
    hpo_method: str = field(default= "bs", metadata={"help": "The hpo method."})

    num_labels: Optional[str] = field(default= 2, metadata={"help": "number of labels"})

    train_dataset: Dataset = field(default= None, metadata={"help": "train dataset"})
    eval_dataset: Dataset = field(default=None, metadata={"help": "dev dataset"})
    test_dataset: Dataset = field(default=None, metadata={"help": "test dataset"})

    metric_name: str = field(default=None, metadata={"help": "metric name"})

    @property
    def folder_name(self):
        return self.hpo_method + "_" + self.scheduler_name + "_" + self.model_type + "_" + self.submit_mode

    @property
    def model_checkpoint(self):
        return os.path.join(self.output_path, "model", self.model_name)

    def init_and_make_dirs(self, search_space_dir):
        if not search_space_dir:
            self._search_space_dir = os.path.abspath(os.path.join(self.CODE_PATH_REL, "flaml/nlp/search_space/", self.task_name))
        else:
            self._search_space_dir = search_space_dir
        # self.search_space_grid = json.load(open(os.path.join(search_space_dir, self.task_name + "_grid.json", "r")))
        output_dir = os.path.join(self.output_path + "output/" + "_".join(self.dataset_name) + "/")
        self._ckpt_dir_abs = os.path.join(output_dir, "checkpoints/", self.folder_name + "/")
