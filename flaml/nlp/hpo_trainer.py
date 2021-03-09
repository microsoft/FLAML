import os,json
import wandb

import ray

from transformers import AutoTokenizer
from transformers import Trainer

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

task_to_eval_name = {
    "cola": "eval_mcc",
    "mnli": "eval_mnli/acc",
    "mrpc": "eval_acc",
    "qnli": "eval_acc",
    "qqp":  "eval_acc",
    "rte":  "eval_acc",
    "sst2": "eval_acc",
    "stsb": "eval_pearson",
    "wnli": "eval_acc"
}

class HPOTrainer(Trainer):

    def __init__(self,
                 task_name,
                 abs_data_path,
                 model_name_short = None,
                 hpo_method = None,
                 scheduler_name = None,
                 submit_mode = "resplit",
                 split_portion = None,
                 wandb_key = None,
    ):
        super()

        self._task_name = task_name
        self._model_name_short = model_name_short
        self._abs_data_path = abs_data_path
        self._hpo_method = hpo_method or 'bs'
        self._scheduler_name = scheduler_name
        self._submit_mode = submit_mode
        self._split_portion = split_portion or {
                                  "train": (0.0, 0.8),
                                  "dev": (0.8, 0.9),
                                  "test": (0.9, 1.0)}

        self._set_folder_name()
        self._init_paths()
        if wandb_key:
            self._set_wandb(wandb_key)

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_dir_abs, use_fast=True)
        self._set_search_space()
        self._set_hp_metric()

    def _set_folder_name(self):
        self._folder_name = self._hpo_method + "_" + self._scheduler_name + self._model_name_short + "_" + self._submit_mode

    def _set_wandb(self,
                   wandb_key):
        os.environ["WANDB_API_KEY"] = wandb_key
        generated_id = wandb.util.generate_id()
        group_name = self._folder_name + "_" + generated_id
        os.environ["WANDB_RUN_GROUP"] = group_name

    def tokenize(self,
                 examples):
        sentence1_key, sentence2_key = task_to_keys[self._task_name]
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        return self.tokenizer(*args, padding="max length", max_length= self.search_space_grid["max_seq_length"][0], truncation=True)

    def _set_search_space(self):
        self.search_space_grid = json.load(open(os.path.join(self._search_space_dir, self._task_name + "_grid.json", "r")))
        self.search_space_hpo = json.load(open(os.path.join(self._search_space_dir, self._task_name + "_hpo.json", "r")))

    def _set_hp_metric(self):
        self._eval_acc_name = task_to_eval_name[self._task_name]

    def prepare_glue_data(self):

        dev_name = "validation" if self._task_name != "mnli" else "validation_matched"
        test_name = "test" if self._task_name != "mnli" else "test_matched"

        data_raw = load_dataset("glue", self._task_name)
        data_encoded = data_raw.map(self.tokenize, batched=True)

        if self._submit_mode == "resplit":
            train_dataset, val_dataset = data_encoded["train"], data_encoded[dev_name]
            data_train_dev = datasets.concatenate_datasets([train_dataset, val_dataset])
            data_train_dev = data_train_dev.shuffle(seed=42)

            train_start, train_end = int(self._split_portion[0] * len(data_train_dev)), int(self._split_portion[1] * len(data_train_dev))
            dev_start, dev_end = int(self._split_portion[0] * len(data_train_dev)), int(self._split_portion[1] * len(data_train_dev))
            test_start, test_end = int(self._split_portion[0] * len(data_train_dev)), int(self._split_portion[1] * len(data_train_dev))

            train_dataset = data_train_dev.select([x for x in range(train_start, train_end)]).flatten_indices()
            eval_dataset = data_train_dev.select([x for x in range(dev_start, dev_end)]).flatten_indices()
            test_dataset = data_train_dev.select([x for x in range(test_start, test_end)]).flatten_indices()
        elif self._submit_mode == "origin":
            train_dataset, val_dataset, test_dataset = data_encoded["train"], data_encoded[dev_name], data_encoded[test_name]

        self._state.train_dataset = train_dataset
        self._state.eval_dataset = val_dataset
        self._state.test_dataset = test_dataset

    def hyperparameter_search(self,
                              hpo_method="bs",
                              scheduler_name=None,
                              num_samples=10000,
                              time_budget=7200,
                              device_nums=None):

        if not device_nums:
            device_nums = {"cpu": 4, "gpu": 4}

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
        else:
            algo = None

        if scheduler_name == "ASHA":
            from ray.tune.schedulers import ASHAScheduler
            scheduler = ASHAScheduler(
                max_t= 100,
                grace_period=1)
        else:
            scheduler = None

        analysis = ray.tune.run(
            train_electra,
            metric= self._eval_acc_name,
            mode = "max",
            resources_per_trial={"gpu": 1, "cpu": 1},
            config= self.search_space_hpo,
            local_dir= self._ckpt_dir_abs,
            num_samples = num_samples,
            time_budget_s= time_budget,
            keep_checkpoints_num = 1,
            checkpoint_score_attr= self._eval_acc_name,
            scheduler= scheduler,
            search_alg= algo)