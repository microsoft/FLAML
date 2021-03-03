import copy
import os
import pickle
import re
import shutil

import torch
import itertools
torch.cuda.empty_cache()
import gc
gc.collect()
""" notice even if the directory is named glue, the RTE data is actually from superglue/RTE
    it was converted from the superglue/RTE with data_process/utils"""
"""The code under ray/tune/function_runner/ImplicitFunc is updated"""
import json
import wandb
import time

#wandb.init(resume=True)

import sys

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ray, argparse
from ray import tune
#from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandForBOHB, MedianStoppingRule, HyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
#from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.zoopt import ZOOptSearch
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
import datasets
from operator import itemgetter
from datasets import load_dataset, load_metric
from dataclasses import dataclass, field
from transformers.modeling_roberta import RobertaClassificationHead
from transformers.modeling_electra import ElectraClassificationHead
from transformers.modeling_deberta import DebertaClassificationHead
sys.path.insert(0, "../../")
from reproduce_exp.data_process.auto_script import auto_movefile, extract_best_ckpt

from ax.service.ax_client import AxClient
from ray.tune.suggest.ax import AxSearch
global this_train_dataset, this_eval_dataset, this_test_dataset
global this_train_feature, this_train_dtype, this_train_unique

LOW_STORE_THRESHOLD=0.0
NUM_EPOCHS=1

from ray.tune import CLIReporter
import tracemalloc
import math

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

task_to_count = {
    "cola": 8500,
    "mnli": 392000,
    "mrpc": 3700,
    "qnli": 105000,
    "qqp": 364000,
    "rte": 2500,
    "sst2": 67000,
    "stsb": 7000,
    "wnli": 634,
}

task2dirname = {
    "rte": "RTE",
    "mrpc": "MRPC",
    "cola": "CoLA",
    "sst2": "SST-2",
    "qnli": "QNLI",
    "qqp": "QQP",
    "mnli": "MNLI-m",
    "stsb": "STS-B",
    "wnli": 'WNLI'
}


all_search_algos = ["rs", "hyper", "optuna", "bohb", "bs"]

ray.shutdown()
ray.init(log_to_driver=True, ignore_reinit_error=True) #, local_mode=True)

search_space = set(["weight_decay", "learning_rate", "task_name", "data_dir", "model_name"])

import transformers
# from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import GlueDataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, PretrainedConfig

import torch
import numpy as np

def freeze_layer(model):
    roberta_params = [x for x in model.electra.children()]
    for param in roberta_params[0].parameters():
        param.requires_grad = False
    layers = [x for x in roberta_params[1].children()][0]
    for idx in range(0, 12):
        for param in layers[idx].parameters():
            param.requires_grad = False

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    # data_dir: str = field(
    #     default=None,
    #     metadata={"help": "The"},
    # )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl", "tsv"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl", "tsv"], "`validation_file` should be a csv or a json file."

def preprocess_function(examples):
    global this_train_dataset, this_eval_dataset, this_test_dataset
    global this_train_feature, this_train_dtype, this_train_unique
    global this_tokenizer
    global this_model, label_to_id
    # Tokenize the texts
    #tokenizer = AutoTokenizer.from_pretrained(tune_config["model_name"])
    sentence1_key, sentence2_key = task_to_keys[tune_config["task_name"]]
    #model = AutoModelForSequenceClassification.from_pretrained(tune_config["model_name"])
    this_data_args = DataTrainingArguments(task_name=tune_config["task_name"])
    #train_dataset = load_dataset("glue", tune_config["task_name"], split="train")

    if this_data_args.pad_to_max_length:
        padding = "max_length"
        max_length = this_data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None
    label_to_id = None
    if tune_config["task_name"] is not None:
        is_regression = tune_config["task_name"] == "stsb"
        if not is_regression:
            label_list = this_train_feature
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = this_train_dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = this_train_unique
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    if is_regression:
        # Some have all caps in their config, some don't.
        num_labels_old = AutoConfig.from_pretrained(tune_config["model_name"]).num_labels
        # model_config = AutoConfig.from_pretrained(
        #     tune_config["model_name"],
        #     num_labels=num_labels_old,
        #     finetuning_task=tune_config["task_name"],
        # )
        # label_name_to_id = {k.lower(): v for k, v in model_config.label2id.items()}
        # if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
        #     label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        # else:
        #     print(
        #         "Your model seems to have been trained with labels, but they don't match the dataset: ",
        #         f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
        #         "\nIgnoring the model labels as a result.",
        #     )
    elif tune_config["task_name"] is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = this_tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks
    if not is_regression and label_to_id and "label" in examples:
        result["label"] = [label_to_id[l] for l in examples["label"]]
    return result

def load_datasets(data_args):
    global this_train_dataset, this_eval_dataset, this_test_dataset
    global this_train_feature, this_train_dtype, this_train_unique
    global this_tokenizer

    this_mode = tune_config["mode"]

    if tune_config["task_name"] is not None:
        train_dataset = load_dataset("glue", data_args.task_name, split="train")
        if args.task_name != "mnli":
            dev_dataset = load_dataset("glue", data_args.task_name, split="validation")
        else:
            dev_dataset = load_dataset("glue", data_args.task_name, split="validation_matched")

        if this_mode == "compare":
            train_dev = datasets.concatenate_datasets([train_dataset, dev_dataset])
            train_dev = train_dev.shuffle(seed=42)

            if args.task_name in ("rte" "mrpc" "stsb" "cola"):
                split_portion = json.load(open(tune_config["search_space_dir"] + "split_small.json", "r"))
            else:
                split_portion = json.load(open(tune_config["search_space_dir"] + "split_large.json", "r"))

            train_start = int(split_portion["train"]["l"] * len(train_dev))  #0.0402 #0.1941
            train_end = int(split_portion["train"]["u"] * len(train_dev))

            dev_start = int(split_portion["dev"]["l"] * len(train_dev))
            dev_end = int(split_portion["dev"]["u"] * len(train_dev))

            test_start = int(split_portion["test"]["l"] * len(train_dev))
            test_end = int(split_portion["test"]["u"] * len(train_dev))

            train_dataset = train_dev.select([x for x in range(train_start, train_end)]).flatten_indices()
            eval_dataset = train_dev.select([x for x in range(dev_start, dev_end)]).flatten_indices()
            test_dataset = train_dev.select([x for x in range(test_start, test_end)]).flatten_indices()

            try:
                print("training data size" + str(len(train_dataset)))
            except:
                print("training data size error")
                pass
        else:
            if args.task_name != "mnli":
                eval_dataset = load_dataset("glue", data_args.task_name, split="validation")
                test_dataset = load_dataset("glue", data_args.task_name, split="test")
            else:
                eval_dataset = load_dataset("glue", data_args.task_name, split="validation_matched")
                test_dataset = load_dataset("glue", data_args.task_name, split="test_matched")

    if data_args.task_name != "stsb":
        this_train_feature = train_dataset.features["label"].names
    else:
        this_train_feature = None

    this_train_dtype = train_dataset.features["label"].dtype
    this_train_unique = train_dataset.unique("label")
    this_tokenizer = AutoTokenizer.from_pretrained(tune_config["model_name"])

    train_dataset = train_dataset.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    test_dataset = test_dataset.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    return train_dataset, eval_dataset, test_dataset

def compute_metrics_each(predictions, label_ids):
    preds = np.argmax(predictions, axis=1)
    return (preds == label_ids)

class TuneTransformerTrainer(transformers.Trainer):
    def get_optimizers(
            self, num_training_steps
    ):
        self.optimizer, self.current_scheduler = super(
        ).get_optimizers(num_training_steps)
        return (self.optimizer, self.lr_scheduler)

    def evaluate(self, eval_dataset=None, epoch = None, tr_loss = None, eval_type=None, model = None):
        #self.args.past_index = 2
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output, predicted_results, label_ids = self.prediction_loop(
            eval_dataloader, description="Evaluation")
        self.log(output.metrics)
        self.save_state()

        tune.report(**output.metrics)
        #wandb.log({"evaluation_accuracy": eval_acc})
        wandb.log({"learning_rate": self.args.learning_rate})
        wandb.log({"adam_epsilon": self.args.adam_epsilon})
        wandb.log({"adam_beta1": self.args.adam_beta1})
        wandb.log({"adam_beta2": self.args.adam_beta2})
        wandb.log({"batch_size": self.args.train_batch_size})
        wandb.log({"weight_decay": self.args.weight_decay})
        wandb.log({"num_train_epochs": self.args.num_train_epochs})
        wandb.log({"warmup_ratio": self.args.warmup_ratio})
        wandb.log({"warmup_steps": self.args.warmup_steps})
        wandb.log({"max_grad_norm": self.args.max_grad_norm})
        wandb.log({"seed": self.args.seed})
        wandb.log({"hidden_dropout_prob": self.args.hidden_dropout_prob})
        wandb.log({"attention_probs_dropout_prob":self.args.attention_probs_dropout_prob})
        wandb.log({"cls_dropout": self.args.cls_dropout})
        wandb.log({"val_loss": output.metrics["eval_loss"], "custom_step": output.metrics["epoch"]}) #, "test_acc": output_metric_test["test"]})
        wandb.log({"val_accuracy": output.metrics[self.args.eval_acc_name], "custom_step": output.metrics["epoch"]})

        grid_config = json.load(open(os.path.join(self.config["search_space_dir"], self.config["model_name_short"] + "_grid.json"), "r"))
        is_grid = self.config["learning_rate"] in grid_config["learning_rate"]
        val_acc = output.metrics[self.args.eval_acc_name]
        val_loss = output.metrics["eval_loss"]
        predicted_results = compute_metrics_each(predicted_results.predictions, label_ids)
        this_ckpt = get_checkpoint(self.status_reporter.get_checkpoint())
        save_pickle = os.path.join(self.args.result_dir, "save_configs", "pickle")
        save_json = os.path.join(self.args.result_dir, "save_configs", "json")

        json.dump({"val_acc": val_acc,
                   "val_loss": val_loss,
                   "start_time": self.args.start_time,
                   "trial_id": self.status_reporter.trial_id,
                   "ckpt": this_ckpt,
                   "pred_results": predicted_results.tolist(),
                   "is_grid": is_grid,
                   },
                  open(os.path.join(save_json, str(self.status_reporter.trial_id) + ".json"), "w"))
        pickle.dump(self.config,
                    open(os.path.join(save_pickle, str(self.status_reporter.trial_id) + ".config"),
                         "wb"))

        return output.metrics

    def save_state(self):
        with tune.checkpoint_dir(step=self.state.global_step) as checkpoint_dir:
            self.args.output_dir = checkpoint_dir
            # This is the directory name that Huggingface requires.
            output_dir = os.path.join(
                self.args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            self.save_model(output_dir)
            #if self.is_world_master():
            torch.save(self.optimizer.state_dict(),
                       os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(),
                       os.path.join(output_dir, "scheduler.pt"))


def recover_checkpoint(tune_checkpoint_dir, model_name=None):
    if tune_checkpoint_dir is None or len(tune_checkpoint_dir) == 0:
        return model_name
    # Get subdirectory used for Huggingface.
    subdirs = [
        os.path.join(tune_checkpoint_dir, name)
        for name in os.listdir(tune_checkpoint_dir)
        if os.path.isdir(os.path.join(tune_checkpoint_dir, name))
    ]
    # There should only be 1 subdir.
    assert len(subdirs) == 1, subdirs
    return subdirs[0]


from transformers import AutoConfig, TrainingArguments, glue_tasks_num_labels
from ray.tune.integration.wandb import wandb_mixin

def convert_methodname(old_methodname):
    new_methodname = re.sub("^(optuna|rs|asha|bs)_", "grid_search_", old_methodname)
    new_methodname = new_methodname.replace("_compare_", "_submit_")
    new_methodname = re.sub("_(\d+)_", "_", new_methodname)
    return new_methodname

def convert_config(config_origin):
    this_config = copy.deepcopy(config_origin)
    del this_config["grid_config"]
    this_config["method_name"] = convert_methodname(this_config["method_name"])
    this_config["wandb"]["group"] = this_config["method_name"]
    this_config["search_algo"] = "grid_search"
    this_config["SAMPLE_SIZE"] = 1
    this_config["mode"] = "submit"
    this_config["time_budget"] = 100000
    return this_config

@wandb_mixin
def train_transformer(config, reporter, checkpoint_dir=None):
    import random
    import torch

    if config["mode"] == "compare":
        pickle.dump(convert_config(config), open(os.path.join(config["result_dir"], "save_configs", reporter.trial_id + ".pickle"), "wb"))

    from transformers.trainer_utils import set_seed
    set_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, config["code_path"])
    from transformer_utils import build_compute_metrics_fn
    global this_train_dataset, this_eval_dataset, this_test_dataset

    rounded_batch_size = config["per_gpu_batch_size"] #int(math.pow(2, math.floor(config["per_gpu_batch_size"])))
    config["num_epochs"] = config["num_epochs"]

    if config["is_warmup"] == False:
        this_warmup_steps = 0
    elif config["warmup_steps"] != None:
        this_warmup_steps = config["warmup_steps"]
    elif config["warmup_ratio"] != None:
        this_warmup_steps = int(len(this_train_dataset) // rounded_batch_size * config["num_epochs"] * config["warmup_ratio"])

    if config["is_seed"] == False:
        this_seed = 42
    else:
        this_seed = config["seed"]

    training_args = TrainingArguments(
        eval_acc_name = config["eval_acc_name"],
        output_dir=tune.get_trial_dir(),
        result_dir = config["result_dir"],
        learning_rate=config["learning_rate"],
        do_train=True,
        do_eval=True,
        gradual_unfreeze=config["gradual_unfreeze"],
        evaluate_during_training=True,
        # Run eval after every epoch.
        eval_steps=(len(this_train_dataset) // (config["num_per_epoch"] * rounded_batch_size)) + 1,
        # We explicitly set save to 0, and do checkpointing in evaluate instead
        save_steps=0,
        fp16=True,
        num_train_epochs= config["num_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=rounded_batch_size,
        per_device_eval_batch_size=rounded_batch_size,
        adam_epsilon=config["adam_epsilon"],
        adam_beta1=config["adam_beta1"],
        adam_beta2=config["adam_beta2"],
        max_grad_norm=config["max_grad_norm"],
        warmup_steps = this_warmup_steps,
        warmup_ratio = config["warmup_ratio"],
        hidden_dropout_prob = config["hidden_dropout_prob"],
        attention_probs_dropout_prob = config["attention_probs_dropout_prob"],
        weight_decay= 0 if not config["weight_decay"] else config["weight_decay"],
        logging_dir= config["log_dir"],
        seed = this_seed,
        cls_dropout=config["cls_dropout"],
        trial_id = reporter.trial_id,
        submit_mode=config["mode"],
        start_time = time.time()
    )

    model_name_or_path = recover_checkpoint(checkpoint_dir, config["model_name"])
    glue_tasks_num_labels["stsb"] = glue_tasks_num_labels["sts-b"]
    del glue_tasks_num_labels["sts-b"]
    num_labels = glue_tasks_num_labels[config["task_name"]]

    num_labels_old = AutoConfig.from_pretrained(model_name_or_path).num_labels

    if config["model_name_short"] == "electra":
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels_old,
            finetuning_task=config["task_name"],
            hidden_dropout_prob = config["hidden_dropout_prob"],
            attention_probs_dropout_prob = config["attention_probs_dropout_prob"],
        )
    else:
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels_old,
            finetuning_task=config["task_name"],
            hidden_dropout_prob = config["hidden_dropout_prob"],
            attention_probs_dropout_prob = config["attention_probs_dropout_prob"],
            cls_dropout=config["cls_dropout"],
        )

    global this_model, this_tokenizer

    if num_labels != num_labels_old:
        model_config.num_labels = num_labels_old
        this_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=model_config,
        )
        model_config.num_labels = num_labels
        print('Reintializing model classifier layer...')
        this_model.num_labels = num_labels
        if args.model_name_short == "roberta":
            this_model.classifier = RobertaClassificationHead(model_config)
        elif args.model_name_short == "electra":
            this_model.classifier = ElectraClassificationHead(model_config)
        elif args.model_name_short == "deberta":
            this_model.classifier = DebertaClassificationHead(model_config)
    else:
        this_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=model_config,
        )
    this_model.resize_token_embeddings(len(this_tokenizer))
    if training_args.gradual_unfreeze == True:
        freeze_layer(this_model)

    # Use our modified TuneTransformerTrainer
    tune_trainer = TuneTransformerTrainer(
        model=this_model,
        args=training_args,
        train_dataset=this_train_dataset,
        eval_dataset=this_eval_dataset,
        test_dataset=this_test_dataset,
        compute_metrics= build_compute_metrics_fn(config["task_name"]),
    )
    tune_trainer.status_reporter = reporter
    tune_trainer.config = config

    tune_trainer.train(model_name_or_path)


def grid2list(grid_config):
    key_val_list = [[(key, each_val) for each_val in val_list] for (key, val_list) in grid_config.items()]
    config_list = [dict(x) for x in itertools.product(*key_val_list)]
    return config_list

def hyperparameter_search(tune_config):
    # Download and cache tokenizer, model, and features
    print("Downloading and caching Tokenizer")

    # Triggers tokenizer download to cache
    global this_tokenizer
    this_tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Downloading and caching pre-trained model")

    # Triggers model download to cache
    #AutoModelForSequenceClassification.from_pretrained(model_name)

    # search_config = {x: config[x] for x in tune_config.keys() if x in search_space}
    if tune_config["search_algo"] in ("grid", "grid_search"):
        algo = None
    elif tune_config["search_algo"] == "bo":
        algo = BayesOptSearch(
            metric= tune_config["eval_acc_name"],
            mode="max",
            utility_kwargs={
                "kind": "ucb",
                "kappa": 2.5,
                "xi": 0.0
            },
        )
    elif tune_config["search_algo"] == "rs":
        algo = None
    elif tune_config["search_algo"] == "bohb":
        algo = TuneBOHB(
            metric = tune_config["eval_acc_name"],
            mode = "max",
            max_concurrent=4,
            # points_to_evaluate=[
            #     {"learning_rate": 3e-5, "per_gpu_batch_size": 5, "weight_decay": 0, "warmup_ratio": 0},
            # ]
        )
    elif tune_config["search_algo"] == "zoopt":
        zoopt_search_config = {
            "parallel_num": 4 # how many workers to parallel
        }
        algo = ZOOptSearch(
            algo="Asracos",
            budget=tune_config["SAMPLE_SIZE"],
            metric=tune_config["eval_acc_name"],
            mode="max",
            random_state=tune_config["hpo_seed"],
            **zoopt_search_config
        )
    elif tune_config["search_algo"] == "optuna":
        algo = OptunaSearch(
            metric=tune_config["eval_acc_name"],
            mode="max",
            points_to_evaluate= grid2list(tune_config["grid_config"]))
    elif tune_config["search_algo"] == "hyper":
        algo = HyperOptSearch(
            metric=tune_config["eval_acc_name"],
            mode="max",
            )
    elif tune_config["search_algo"] == "bs":
        from flaml import BlendSearch

        if tune_config["init_config"] is True:
            algo = BlendSearch(
                metric=tune_config["eval_acc_name"],
                mode="max",
                points_to_evaluate= grid2list(tune_config["grid_config"]),
            )
        else:
            algo = BlendSearch(
                metric=tune_config["eval_acc_name"],
                mode="max"
            )

    if tune_config["search_algo"] in ("grid", "grid_search"):
        scheduler = None
    elif tune_config["search_algo"] == "bohb":
         scheduler = HyperBandForBOHB(time_attr="training_iteration", metric=tune_config["eval_acc_name"], mode="max")
    elif tune_config["tune_scheduler"] == "asha":
        scheduler = AsyncHyperBandScheduler(time_attr="training_iteration", metric=tune_config["eval_acc_name"], mode="max")
    elif tune_config["tune_scheduler"] == "hb":
        scheduler = HyperBandScheduler(time_attr="training_iteration", metric=tune_config["eval_acc_name"], mode="max")
    else:
        scheduler = None

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "weight_decay",
            "learning_rate": "learning_rate",
            "per_gpu_batch_size": "per_gpu_batch_size",
            "num_epochs": "num_epochs",
            "warmup_steps": "warmup_steps",
            "warmup_ratio": "warmup_ratio",
            "seed": "seed",
            "attention_probs_dropout_prob": "attention_probs_dropout_prob",
            "hidden_dropout_prob": "hidden_dropout_prob",
            "cls_dropout": "cls_dropout",
        },
        metric_columns=[
            tune_config["eval_acc_name"], "eval_loss", "epoch", "training_iteration"
        ])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    tracemalloc.start()

    tune_kwargs = {
        "num_samples": tune_config["SAMPLE_SIZE"],
        "config": tune_config
    }

    analysis = tune.run(
        train_transformer,
        #keep_checkpoints_num=1,
        #checkpoint_freq=0,
        resources_per_trial={
            "cpu": 1,
            "gpu": 1,
        },
        scheduler= scheduler,
        checkpoint_score_attr= "epoch",
        progress_reporter=reporter,
        local_dir= tune_config["output_dir"],
        name= tune_config["method_name"],
        stop={"epoch": 15},
        search_alg= algo,
        time_budget_s = tune_config["time_budget"],
        # callbacks=[WandbLoggerCallback(
        #     project="hpo",
        #     group=os.environ["WANDB_RUN_GROUP"],
        #     api_key=os.environ["WANDB_API_KEY"],
        #     reinit =True,
        #     allow_val_change=True,
        #     log_config=True)],
        **tune_kwargs
    )
    end.record()
    torch.cuda.synchronize()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return start.elapsed_time(end), current, peak

def get_label_list():
    global label_list, num_labels, label_to_id, label_name_to_id, is_regression
    label_to_id = None

    if not is_regression and tune_config["task_name"] is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

def reproduce_hpo(best_config, best_checkpoint, trial_id):
    sys.path.insert(0, best_config["code_path"])
    global this_train_dataset, this_eval_dataset, this_test_dataset
    from transformer_utils import build_compute_metrics_fn

    best_model = AutoModelForSequenceClassification.from_pretrained(
        best_checkpoint).to("cuda")

    test_args = TrainingArguments(output_dir=best_config["output_dir"]) #, seed = best_config["seed"])

    if best_config["task_name"] is not None:
        is_regression = best_config["task_name"] == "stsb"
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = this_train_dataset.features["label"].dtype in ["float32", "float64"]
    if not is_regression:
        label_list = this_train_dataset.features["label"].names

    test_trainer = transformers.Trainer(
        best_model,
        test_args,
        compute_metrics= build_compute_metrics_fn(best_config["task_name"]))

    eval_metrics = test_trainer.evaluate(this_eval_dataset)
    if best_config["mode"] == "compare":
        test_metrics = test_trainer.evaluate(this_test_dataset)
    else:
        test_metrics = {}

    tasks = [best_config["task_name"]]
    test_datasets = [this_test_dataset]

    if best_config["mode"] != "compare":
        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            try:
                test_dataset.remove_columns_("label")
            except ValueError:
                pass
            predictions = test_trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            this_method_name = best_config["method_name"]
            output_test_file = os.path.join(best_config["result_dir"], f"tested_results_{this_method_name}_{trial_id}.txt")
            if test_trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    writer.write("index\tprediction\n")
                    count = 0
                    for index, item in enumerate(predictions):
                        if is_regression:
                            if item > 5.0:
                                item = 5.0
                            writer.write(f"{index}\t{item:3.3f}\n")
                            count += 1
                        else:
                            if best_config["task_name"] in ("rte", "qnli", "mnli"):
                                item = label_list[item]
                                writer.write(f"{index}\t{item}\n")
                                count += 1
                            else:
                                if int(item) == item:
                                    item = int(item)
                                    writer.write(f"{index}\t{item}\n") #json.dumps({"idx": index, "label": item}) + "\n")
                                    count += 1
                                else:
                                    writer.write(f"{index}\t{item:3.3f}\n")
                                    count += 1
        result = {"eval_results": eval_metrics}
        open(os.path.join(best_config["result_dir"], f"submit_eval_{trial_id}.json"), "w").write(
            json.dumps(result, indent=4))

        return eval_metrics, test_metrics, None, f"tested_results_{this_method_name}_{trial_id}.txt"
    else:
        test_metrics = test_trainer.evaluate_metric(this_test_dataset)
        train_metrics = test_trainer.evaluate_metric(this_train_dataset)
        # trialid2avgscore = extract_best_trial_compare_mode(best_config["result_dir"])
        # sorted_trialid2avgscore = sorted(trialid2avgscore.items(), key=lambda x: x[1], reverse=True)
        # best_trialid = sorted_trialid2avgscore[0][0]
        return eval_metrics, test_metrics, train_metrics, ""


def get_checkpoint(checkpoint):
    checkpoint_dirname = checkpoint.split("/")[-1].replace("_", "-")
    return checkpoint + "/" + checkpoint_dirname + "/"

def get_free_size(output_path):
    result = os.statvfs(output_path)
    block_size = result.f_frsize
    total_blocks = result.f_blocks
    free_blocks = result.f_bfree
    # giga=1024*1024*1024
    giga = 1000 * 1000 * 1000
    total_size = total_blocks * block_size / giga
    free_size = free_blocks * block_size / giga
    return free_size / total_size

def garbage_collection(data_dir, ckpt_dir, low_space_threshold=0.5):
    free_ratio = get_free_size(data_dir)
    if free_ratio < low_space_threshold:
        """ try removing the checkpoint directory first"""
        ckpt_dirs = os.listdir(ckpt_dir)
        for each_dir in ckpt_dirs:
            print("removing " + os.path.join(ckpt_dir, each_dir))
            shutil.rmtree(os.path.join(ckpt_dir, each_dir))

    updated_free_ratio = get_free_size(data_dir)
    print("Old ratio: " + str(free_ratio) + "; new ratio: " + str(updated_free_ratio))

    if updated_free_ratio < low_space_threshold:
        raise ValueError("No enough space left on device, please remove checkpoints from " + ckpt_dir)

def zip_submission(this_codepath, save_zip, trial_id, result_dir, testfile_name, upper_task_name):
    print("preparing zip file for submission")
    src_path = os.path.join(this_codepath, "submission_template")
    dst_path = os.path.join(save_zip, str(trial_id))
    shutil.copytree(src_path, dst_path)

    src_path = os.path.join(result_dir, testfile_name)
    dst_path = os.path.join(save_zip, str(trial_id), upper_task_name + ".tsv")
    shutil.copyfile(src_path, dst_path)

    shutil.make_archive(os.path.join(save_zip, str(trial_id)), 'zip', os.path.join(save_zip, str(trial_id)))

def get_trials_larger_than_grid(save_zip):
    save_config_dir = os.path.join(save_zip, "save_configs")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--scheduler_name', type=str, help='scheduler name', required=False)
    arg_parser.add_argument('--data_path', type=str, help='data path', required=False)
    arg_parser.add_argument('--code_path', type=str, help='code path', required=False)
    arg_parser.add_argument('--search_algo', type=str, help='algo name', required=False)
    arg_parser.add_argument('--split', type=int, help='split', required=False)
    arg_parser.add_argument("--num_per_epoch", type=int, default = None, required=False)
    arg_parser.add_argument("--num_epochs", type=float, default=0, required=False)
    arg_parser.add_argument("--SAMPLE_SIZE", type=int, default=None, required=False)
    arg_parser.add_argument("--glue_name", type=str, default=None, required=False)
    arg_parser.add_argument("--task_name", type=str, default=None, required=False)
    arg_parser.add_argument("--model_name_short", type=str, default=None, required=False)
    arg_parser.add_argument("--mode", type=str, default=None, required=False)
    arg_parser.add_argument("--init_config", type=bool, default=False, required=False)
    arg_parser.add_argument("--gen_config", action='store_true', help='is generate config mode')
    arg_parser.add_argument("--is_predict", type=int, default=None, required=False)
    arg_parser.add_argument("--config_path", type=str, default=None, required=False)
    arg_parser.add_argument("--seed", type=int, default=None, required=False)
    arg_parser.add_argument("--hpo_seed", type=int, default=None, required=False)
    arg_parser.add_argument("--suffix", type=str, default=None, required=False)
    arg_parser.add_argument("--api_key", type=str, default=None, required=False)
    arg_parser.add_argument("--tune_mode", type=str, default="", required=False)
    arg_parser.add_argument("--gradual_unfreeze", type=bool, default=False, required=False)
    arg_parser.add_argument("--time_budget", type=int, default=36000, required=False)
    arg_parser.add_argument("--ckpt_path", type=str, default="", required=False)
    arg_parser.add_argument("--bestconfig_path", type=str, default="", required=False)
    arg_parser.add_argument("--dotconfig_path", type=str, default="", required=False)
    arg_parser.add_argument("--dir_name", type=str, default="", required=False)

    args = arg_parser.parse_args()

    if args.search_algo == "grid_search":
        args.SAMPLE_SIZE = 1
        args.time_budget = 1000000
    else:
        args.SAMPLE_SIZE = 1000000

    CODE_PATH_REL="../../"
    DATA_PATH_REL="../../../data/"

    if args.mode != "test_only":

        if args.task_name.lower() == args.task_name:
            upper_task_name = args.task_name.upper()
            if args.task_name == "cola":
                upper_task_name = "CoLA"
        else:
            upper_task_name = args.task_name
        output_dir = os.path.join(DATA_PATH_REL + "output/" + args.glue_name + "/", upper_task_name)
        output_dir_abs = os.path.join(args.data_path + "output/" + args.glue_name + "/", upper_task_name)

        input_dir = os.path.join(DATA_PATH_REL + "input/" + args.glue_name + "/", upper_task_name)
        input_dir_abs = os.path.join(args.data_path + "input/" + args.glue_name + "/", upper_task_name)

        config_dir = os.path.join(DATA_PATH_REL + "configs/")
        config_dir_abs = os.path.join(args.data_path + "configs/")

        search_space_dir = os.path.join(CODE_PATH_REL + "search_space/")
        search_space_abs_dir = os.path.join(args.code_path + "search_space/")

        eval_acc_name = json.load(open(search_space_dir + "metric.json", "r"))[args.task_name]
        model_name = os.path.join(args.data_path, "model", json.load(open(os.path.join(search_space_dir, "model_path.json"), "r"))[args.model_name_short])

        if args.search_algo not in ("grid", "grid_search"):
            method_name = args.search_algo + "_" + args.model_name_short + "_" + args.task_name + "_" + args.mode + "_" + str(args.hpo_seed) + "_" + args.suffix
        else:
            method_name = args.search_algo + "_" + args.model_name_short + "_" + args.task_name + "_" + args.mode  + "_" + args.suffix

        os.environ["WANDB_API_KEY"] = args.api_key
        generated_id = wandb.util.generate_id()
        group_name = method_name + "_" + generated_id
        os.environ["WANDB_RUN_GROUP"] = group_name

        outputing_dir = os.path.join(output_dir, "checkpoints/", method_name + "/")
        outputing_dir_abs = os.path.join(output_dir_abs, "checkpoints/", method_name + "/")

        log_dir = os.path.join(output_dir, "logs/", method_name + "/")
        log_dir_abs = os.path.join(output_dir_abs, "logs/", method_name + "/")

        ckpt_dir = os.path.join(output_dir, "checkpoints/", method_name + "/")
        ckpt_dir_abs = os.path.join(output_dir_abs, "checkpoints/", method_name + "/")

        result_dir = os.path.join(output_dir, "result/" + method_name + "/")
        result_dir_abs = os.path.join(output_dir_abs, "result/" + method_name + "/")

        save_config_dir = os.path.join(result_dir, "save_configs")

        import pathlib

        for each_dir in (outputing_dir, log_dir, ckpt_dir, result_dir):
            if not os.path.exists(each_dir):
                pathlib.Path(each_dir).mkdir(parents=True, exist_ok=True)

        if os.path.exists(result_dir + "val_test.txt"):
            os.remove(result_dir + "val_test.txt")

        if not os.path.exists(save_config_dir):
            os.mkdir(save_config_dir)
            os.mkdir(os.path.join(save_config_dir, "pickle"))
            os.mkdir(os.path.join(save_config_dir, "json"))
            os.mkdir(os.path.join(save_config_dir, "zip"))
        elif args.is_predict == 0:
            shutil.rmtree(save_config_dir)
            os.mkdir(save_config_dir)
            os.mkdir(os.path.join(save_config_dir, "pickle"))
            os.mkdir(os.path.join(save_config_dir, "json"))
            os.mkdir(os.path.join(save_config_dir, "zip"))

        if args.is_predict == 0:
            if os.path.exists(result_dir + "best_grid.json"):
                os.remove(result_dir + "best_grid.json")
            if os.path.exists(result_dir + "best_all.json"):
                os.remove(result_dir + "best_all.json")

        garbage_collection(DATA_PATH_REL, os.path.join(output_dir, "checkpoints/"), LOW_STORE_THRESHOLD)

        global tune_config

        if args.config_path != None:
            print("\n\n")
            print("loading config from " + args.config_path + "\n\n")
            tune_config = pickle.load(open(args.config_path, "rb"))
            tune_config["code_path"] = args.code_path
            tune_config["method_name"] = method_name
            tune_config["model_name"] = model_name
            tune_config["output_dir"] = outputing_dir_abs
            tune_config["log_dir"] = log_dir_abs
            tune_config["ckpt_dir"] = ckpt_dir_abs
            tune_config["result_dir"] = result_dir_abs
            if args.seed and args.hpo_seed:
                tune_config["seed"] = args.seed
                tune_config["hpo_seed"] = args.hpo_seed
            generated_id = args.config_path[:-7].split("_")[-1]
            group_name = method_name + "_" + generated_id
            os.environ["WANDB_RUN_GROUP"] = group_name
            print("-----------")
            print("num_epochs" + str(tune_config["num_epochs"]))
            print("time budget" + str(tune_config["time_budget"]))
            print("warmup ratio" + str(tune_config["warmup_ratio"]))
            print("weight decay" + str(tune_config["weight_decay"]))
            print("learning rate:" + str(tune_config["learning_rate"]))
            print("hidden dropout:" + str(tune_config["hidden_dropout_prob"]))
            print("attention dropout:" + str(tune_config["attention_probs_dropout_prob"]))
            print("batch size:" + str(tune_config["per_gpu_batch_size"]))
            print("-----------")

            if args.mode == "submit":
                print(method_name, args.dir_name)
                assert method_name == args.dir_name
                print(args.config_path, group_name + ".config")
                assert args.config_path.endswith(group_name + ".config")

            #import pdb; pdb.set_trace()
        else:
            grid_config = json.load(open(os.path.join(search_space_dir, args.model_name_short + "_grid.json"), "r"))
            hpo_config = json.load(open(os.path.join(search_space_dir, args.model_name_short + "_hpo.json"), "r"))

            hp_keys = ["per_gpu_batch_size", "warmup_ratio", "learning_rate", "weight_decay",
                       "hidden_dropout_prob", "attention_probs_dropout_prob", "seed", "num_epochs",
                       "max_grad_norm"]
            hpkey2var = {}

            if args.search_algo == "grid_search":
                for hp_id in range(len(hp_keys)):
                    each_hp_key = hp_keys[hp_id]
                    this_config = grid_config[each_hp_key]
                    assert isinstance(this_config, dict) or isinstance(this_config, list), "Config of " + each_hp_key + " must be dict or list"
                    hpkey2var[each_hp_key] = tune.grid_search(this_config)
                    if each_hp_key == "num_epochs" and args.model_name_short == "roberta" and args.task_name in ("rte", "stsb"):
                        hpkey2var[each_hp_key] = tune.grid_search([10.05])
            else:
                if args.tune_mode == "full":
                    for hp_id in range(len(hp_keys)):
                        each_hp_key = hp_keys[hp_id]
                        this_config = hpo_config[each_hp_key]
                        assert isinstance(this_config, dict) or isinstance(this_config, list), "Config of " + each_hp_key + " must be dict or list"
                        if isinstance(this_config, dict):
                            lower = this_config["l"]
                            upper = this_config["u"]
                            space = this_config["space"]
                            if space == "log":
                                hpkey2var[each_hp_key] = tune.loguniform(lower, upper)
                            elif space == "linear":
                                hpkey2var[each_hp_key] = tune.uniform(lower, upper)
                            elif space == "quniform":
                                hpkey2var[each_hp_key] = tune.quniform(lower, upper, this_config["interval"])
                        else:
                            hpkey2var[each_hp_key] = tune.choice(this_config)

            tune_config = {
                "init_config": args.init_config,
                "search_space_dir": search_space_abs_dir,
                # These 3 configs_large below were defined earlier
                "grid_config": grid_config,
                "model_name_short": args.model_name_short,
                "code_path": args.code_path,
                "code_path_rel": "../../",
                "time_budget": args.time_budget,
                "model_name": model_name,
                "task_name": args.task_name,
                "method_name": method_name,
                "gradual_unfreeze": args.gradual_unfreeze,
                "warmup_steps":
                    None,
                "is_warmup":
                    tune.choice([True]),
                "cls_dropout":
                        0.1,
                "is_seed":
                    tune.choice([True]),
                "hpo_seed":
                    args.hpo_seed,
                "adam_epsilon":
                    1e-6,
                "adam_beta1":
                    0.9,
                "adam_beta2":
                    0.999,
                "max_steps": -1,
                "wandb": {
                    "project": "hpo",
                    "group": group_name,
                    "reinit": True,
                    "allow_val_change": True
                },
                "tune_scheduler": args.scheduler_name,
                "split": args.split,
                "output_dir": outputing_dir_abs,
                "log_dir": log_dir_abs,
                "ckpt_dir": ckpt_dir_abs,
                "search_algo": args.search_algo,
                "num_per_epoch": args.num_per_epoch,
                "SAMPLE_SIZE": args.SAMPLE_SIZE,
                "eval_acc_name": eval_acc_name,
                "mode": args.mode,
                "result_dir": result_dir_abs,
            }

            for each_hp_key in hpkey2var.keys():
                tune_config[each_hp_key] = hpkey2var[each_hp_key]

            from shutil import copyfile

            pickle.dump(tune_config, open(os.path.join(config_dir, group_name + ".config"), "wb"))
            copyfile("auto_run.sh", os.path.join(config_dir, "auto_run_" + generated_id + ".sh"))

            if args.gen_config:
                tune_config["mode"] = args.mode
                if args.search_algo not in ("grid", "grid_search"):
                    method_name = args.search_algo + "_" + args.model_name_short + "_" + args.task_name + "_" + tune_config["mode"] + "_" + str(args.hpo_seed) + "_" + args.suffix
                else:
                    method_name = args.search_algo + "_" + args.model_name_short + "_" + args.task_name + "_" + tune_config["mode"] + "_" + args.suffix

                group_name = method_name + "_" + generated_id

                print('time budget' + str(tune_config['time_budget']))
                print('sample isze' + str(tune_config['SAMPLE_SIZE']))
                print('search algo' + tune_config['search_algo'])
                print('tune scheduler' + str(tune_config['tune_scheduler']))
                print('num epochs' + str(tune_config['num_epochs']))

                pickle.dump(tune_config, open(os.path.join(CODE_PATH_REL, "configs", group_name + ".config"), "wb"))
                sys.exit(1)

        global this_train_train_datasetdataset, this_eval_dataset, this_test_dataset, is_regression
        this_data_args = DataTrainingArguments(task_name=args.task_name)
        global origin_label2id, num_labels, label_list, label_to_id, label_name_to_id

        this_train_dataset, this_eval_dataset, this_test_dataset = load_datasets(this_data_args)

        if tune_config["task_name"] is not None:
            is_regression = tune_config["task_name"] == "stsb"
            if not is_regression:
                label_list = this_train_dataset.features["label"].names
                num_labels = len(label_list)
            else:
                num_labels = 1
        else:
            # Trying to have good defaults here, don't hesitate to tweak to your needs.
            is_regression = this_train_dataset.features["label"].dtype in ["float32", "float64"]
            if is_regression:
                num_labels = 1
            else:
                # A useful fast method:
                # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
                label_list = this_train_dataset.unique("label")
                label_list.sort()  # Let's sort it for determinism
                num_labels = len(label_list)

        num_labels_old = AutoConfig.from_pretrained(model_name).num_labels
        origin_model_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels_old,
            finetuning_task= tune_config["task_name"],
        )

        if num_labels != num_labels_old:
            origin_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config= origin_model_config,
            )
            origin_model_config.num_labels = num_labels
            origin_model.num_labels = num_labels
            if args.model_name_short == "roberta":
                origin_model.classifier = RobertaClassificationHead(origin_model_config)
            elif args.model_name_short == "electra":
                origin_model.classifier = ElectraClassificationHead(origin_model_config)
            elif args.model_name_short == "deberta":
                origin_model.classifier = DebertaClassificationHead(origin_model_config)
        else:
            origin_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=origin_model_config,
            )

        origin_label2id = origin_model.config.label2id
        get_label_list()

        save_pickle = os.path.join(result_dir, "save_configs", "pickle")
        save_json = os.path.join(result_dir, "save_configs", "json")
        save_zip = os.path.join(result_dir, "save_configs", "zip")

        if args.is_predict == 0:
            gpu_hours, gpu_current, gpu_peak = hyperparameter_search(tune_config)
        else:
            trials_larger = get_trials_larger_than_grid()
            fout = open(os.path.join(result_dir, "trial2score.tsv"), "w")
            fout.write("trial_id\tloss\tsignificance\tval_acc\ttest_acc\ttrain_acc\tstart_time\n")

            grid_json = json.load(open(os.path.join(result_dir, "best_grid.json"), "r"))
            grid_trialid = grid_json["trial_id"]

            if tune_config["mode"] != "compare":
                grid_config = pickle.load(open(os.path.join(result_dir, "best_grid.config"), "rb"))
                pred_results_grid, acc_grid, loss_grid = grid_json["pred_results"], grid_json["best_acc"], grid_json["best_loss"]
                grid_checkpoint = grid_json["best_ckpt"]
                _, _, _, testfile_grid = reproduce_hpo(grid_config, grid_checkpoint, grid_trialid)
                fout.write(str(grid_trialid) + "\t" + str(loss_grid) + "\t\t" + str(acc_grid) + "\n")
                fout.flush()
                zip_submission(args.code_path, save_zip, grid_trialid, result_dir, testfile_grid,
                               task2dirname[args.task_name])
            else:
                grid_config = pickle.load(open(os.path.join(result_dir, "best_grid.config"), "rb"))
                pred_results_grid, val_acc_grid, loss_grid, best_start = grid_json["pred_results"], grid_json["best_val_acc"], grid_json["best_val_loss"], grid_json["best_start"]
                grid_checkpoint = grid_json["best_ckpt"]
                _, test_metrics, train_metrics, _ = reproduce_hpo(grid_config, grid_checkpoint, grid_trialid)
                test_acc_grid = test_metrics[eval_acc_name]
                train_acc_grid = train_metrics[eval_acc_name]
                fout.write(str(grid_trialid) + "\t" + str(loss_grid) + "\t\t" + str(val_acc_grid) + "\t" + str(test_acc_grid) + "\t" + str(train_acc_grid) + "\t" + str(best_start) + "\n")
                fout.flush()

            from scipy import stats
            for each_file in os.listdir(save_json):
                trial_id = each_file[:-5]
                result_json = json.load(open(os.path.join(save_json, each_file), "r"))

                if tune_config["mode"] != "compare":
                    pred_results, eval_accs, eval_losses = result_json["pred_results"], result_json["best_acc"], result_json["best_loss"]
                    this_config = pickle.load(open(os.path.join(save_pickle, str(trial_id) + ".config"), "rb"))
                    this_checkpoint = result_json["best_ckpt"]
                    pval = stats.ttest_ind(pred_results_grid, pred_results)[1]
                    print("generating test result for " + str(trial_id) + ", eval acc " + str(eval_accs))
                    _, _, _, testfile_name = reproduce_hpo(this_config, this_checkpoint, trial_id)
                    zip_submission(args.code_path, save_zip, trial_id, result_dir, testfile_name,
                                   task2dirname[args.task_name])
                    fout.write(str(trial_id) + "\t" + str(eval_losses) + "\t" + str(pval) + "\t" + str(eval_accs) + "\n")
                    fout.flush()
                else:
                    pred_results, eval_accs, eval_losses, best_start = result_json["pred_results"], result_json["best_val_acc"], result_json["best_val_loss"], result_json["best_start"]
                    this_config = pickle.load(open(os.path.join(save_pickle, str(trial_id) + ".config"), "rb"))
                    this_checkpoint = result_json["best_ckpt"]
                    pval = stats.ttest_ind(pred_results_grid, pred_results)[1]
                    print("generating test result for " + str(trial_id) + ", eval acc " + str(eval_accs))
                    eval_metrics, test_metrics, train_metrics, _ = reproduce_hpo(this_config, this_checkpoint, trial_id)
                    test_acc = test_metrics[eval_acc_name]
                    train_acc = train_metrics[eval_acc_name]
                    fout.write(str(trial_id) + "\t" + str(eval_losses) + "\t" + str(pval) + "\t" + str(eval_accs) + "\t" + str(test_acc) + "\t" + str(train_acc) + "\t" + str(best_start) + "\n")
                    fout.flush()

            fout.close()
            """ saving checkpoint and test files"""
            # auto_movefile(args.data_path, args.code_path, args.model_name_short, upper_task_name, method_name, args.search_algo, method_name, generated_id)
            ckpt_root = os.path.join(output_dir, "checkpoints/")
            ckpt_dirs = os.listdir(ckpt_root)
            for each_dir in ckpt_dirs:
                print("removing " + os.path.join(ckpt_root, each_dir))
                shutil.rmtree(os.path.join(ckpt_root, each_dir))
            if os.path.exists("/home/xliu127/ray_results/"):
                shutil.rmtree("/home/xliu127/ray_results/")
    else:
        search_space_dir = os.path.join(args.code_path + "search_space/")
        eval_acc_name = json.load(open(search_space_dir + "metric.json", "r"))[args.task_name]
        model_name = os.path.join(args.data_path, "model", json.load(open(os.path.join(search_space_dir, "model_path.json"), "r"))[args.model_name_short])

        tune_config = pickle.load(open(args.dotconfig_path, "rb"))

        best_acc, best_ckpt_path, best_ckpt_config = extract_best_ckpt(args.ckpt_path, metric = eval_acc_name)

        print("best accuracy = " + str(best_acc))
        print("best ckpt = " + str(best_ckpt_path))

        best_ckpt_config = best_ckpt_config["config"]

        tune_config["code_path"] = args.code_path
        tune_config["model_name"] = model_name
        if args.task_name.lower() == args.task_name:
            upper_task_name = args.task_name.upper()
            if args.task_name == "cola":
                upper_task_name = "CoLA"
        else:
            upper_task_name = args.task_name
        output_dir = os.path.join(args.data_path + "output/" + args.glue_name + "/", upper_task_name)
        if args.search_algo not in ("grid", "grid_search"):
            method_name = args.search_algo + "_" + args.model_name_short + "_" + args.task_name + "_" + args.mode + "_" + str(args.hpo_seed) + "_" + args.suffix
        else:
            method_name = args.search_algo + "_" + args.model_name_short + "_" + args.task_name + "_" + args.mode  + "_" + args.suffix
        result_dir = os.path.join(output_dir, "result/" + method_name + "/")
        import pathlib
        if not os.path.exists(result_dir):
            pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
        tune_config["result_dir"] = result_dir
        best_ckpt_config["result_dir"] = result_dir
        this_data_args = DataTrainingArguments(task_name=tune_config["task_name"])
        this_train_dataset, this_eval_dataset, this_test_dataset = load_datasets(this_data_args)
        eval_metrics, test_metrics, best_trialid2avgscore = reproduce_hpo(best_ckpt_config, best_ckpt_path)

        result = {tune_config["eval_acc_name"]: eval_metrics,
                  "test_acc": test_metrics}
        print("result can be found in: " + tune_config["result_dir"])
        print("best validation and test: " + str(best_trialid2avgscore))
        open(os.path.join(tune_config["result_dir"], tune_config["method_name"] + ".log"), "w").write(json.dumps(result, indent=4))