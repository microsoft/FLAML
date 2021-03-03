import os,json, datasets

import ray
from datasets import (
    load_dataset,
)
from transformers import AutoTokenizer

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

class AutoHuggingFace:

    def __init__(self,
                 task_name,
                 origin_model_path):
        self.task_name = task_name
        self.tokenizer = AutoTokenizer.from_pretrained(origin_model_path, use_fast=True)
        self.init_paths()

        self.search_space_grid = json.load(open(os.path.join(self.search_space_dir, task_name + "_grid.json", "r")))
        self.search_space_hpo = json.load(open(os.path.join(self.search_space_dir, task_name + "_hpo.json", "r")))

        self.split_json = json.load(open(os.path.join(self.search_space_dir, task_name + "_split.json", "r")))

    def init_paths(self):
        self.search_space_dir = os.path.abspath(os.path.join("../../flaml/nlp/search_space/", self.task_name))

    def tokenize(self,
                 examples):
        sentence1_key, sentence2_key = task_to_keys[self.task_name]
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        return self.tokenizer(*args, padding="max length", max_length= self.search_space_grid["max_seq_length"][0], truncation=True)

    def prepare_glue_data(self,
                          mode,
                          split_portion=None):
        dev_name = "validation" if self.task_name != "mnli" else "validation_matched"
        test_name = "test" if self.task_name != "mnli" else "test_matched"

        data_raw = load_dataset("glue", self.task_name)
        data_encoded = data_raw.map(self.tokenize, batched=True)

        if mode == "resplit":
            train_dataset, val_dataset = data_encoded["train"], data_encoded[dev_name]
            data_train_dev = datasets.concatenate_datasets([train_dataset, val_dataset])
            data_train_dev = data_train_dev.shuffle(seed=42)

            train_start, train_end = int(split_portion[0] * len(data_train_dev)), int(split_portion[1] * len(data_train_dev))
            dev_start, dev_end = int(split_portion[0] * len(data_train_dev)), int(split_portion[1] * len(data_train_dev))
            test_start, test_end = int(split_portion[0] * len(data_train_dev)), int(split_portion[1] * len(data_train_dev))

            train_dataset = data_train_dev.select([x for x in range(train_start, train_end)]).flatten_indices()
            eval_dataset = data_train_dev.select([x for x in range(dev_start, dev_end)]).flatten_indices()
            test_dataset = data_train_dev.select([x for x in range(test_start, test_end)]).flatten_indices()
        elif mode == "origin_split":
            train_dataset, val_dataset, test_dataset = data_encoded["train"], data_encoded[dev_name], data_encoded[test_name]

        self._state.train_dataset = train_dataset
        self._state.eval_dataset = val_dataset
        self._state.test_dataset = test_dataset

    def tune(self,
            hpo_method,
            scheduler_name,
            device_nums):
        ray.init(num_cpus=device_nums["cpu"], num_gpus=device_nums["gpu"])
        if 'ASHA' == hpo_method:
            algo = None
        elif 'Optuna' == hpo_method:
            from ray.tune.suggest.optuna import OptunaSearch
            algo = OptunaSearch()
        elif 'CFO' == hpo_method:
            from flaml import CFO
            algo = CFO(points_to_evaluate=[{
                "num_train_epochs": 1,
            }])
        elif 'BlendSearch' == hpo_method:
            from flaml import BlendSearch
            algo = BlendSearch(points_to_evaluate=[{
                "num_train_epochs": 1,
                "per_device_train_batch_size": 128,
            }])
        elif 'Dragonfly' == hpo_method:
            from ray.tune.suggest.dragonfly import DragonflySearch
            algo = DragonflySearch()
        elif 'SkOpt' == hpo_method:
            from ray.tune.suggest.skopt import SkOptSearch
            algo = SkOptSearch()
        elif 'Nevergrad' == hpo_method:
            from ray.tune.suggest.nevergrad import NevergradSearch
            import nevergrad as ng
            algo = NevergradSearch(optimizer=ng.optimizers.OnePlusOne)
        elif 'ZOOpt' == hpo_method:
            from ray.tune.suggest.zoopt import ZOOptSearch
            algo = ZOOptSearch(budget=num_samples)
        elif 'Ax' == hpo_method:
            from ray.tune.suggest.ax import AxSearch
            algo = AxSearch(max_concurrent=3)
        elif 'HyperOpt' == hpo_method:
            from ray.tune.suggest.hyperopt import HyperOptSearch
            algo = HyperOptSearch()
            scheduler = None

