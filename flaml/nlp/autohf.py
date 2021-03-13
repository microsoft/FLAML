import os,json
import wandb
import numpy as np

import time
import ray
import datasets
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, glue_tasks_num_labels, AutoConfig, \
    TrainingArguments
from transformers import Trainer
from functools import partial

import flaml
from flaml.nlp.autohf_args import AutoHFArguments
from flaml.nlp.modeling_auto import AutoClassificationHead

task_list = [
    "text-classification",
    "question-answering"
]

class AutoHuggingFace:

    def _set_wandb(self,
                   wandb_key):
        os.environ["WANDB_API_KEY"] = wandb_key
        generated_id = wandb.util.generate_id()
        group_name = self.args.folder_name + "_" + generated_id
        os.environ["WANDB_RUN_GROUP"] = group_name

    def _set_search_space(self):
        self.search_space_grid = json.load(open(os.path.join(self.args._search_space_dir, self.args.task_name + "_grid.json", "r")))
        self.search_space_hpo = json.load(open(os.path.join(self.args._search_space_dir, self.args.task_name + "_hpo.json", "r")))

    @property
    def _eval_acc_name(self):
        return self.dataset_module.eval_name_mapping[self.args.task_name][0]

    def _tokenize(self,
                  examples,
                  sentence_keys):
        if len(sentence_keys) > 1:
            sentence1_key, sentence2_key = sentence_keys[0], sentence_keys[1]
        else:
            sentence1_key = sentence_keys[0]
            sentence2_key = None

        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        # TODO: remove self.
        return self._tokenizer(*args, padding="max length", max_length=self.search_space_grid["max_seq_length"][0],
                              truncation=True)

    def _classify_datatype(self,
                           task_name,
                           dataset_name,
                           input_path):
        assert task_name or dataset_name or input_path

        if task_name:
            return "glue"
        elif dataset_name:
            return "hf"
        else:
            return "custom"

    def get_sentence_keys(self,
                          data_raw):
        return [x for x in data_raw["train"].features.keys() if x not in ("label", "idx")]

    def get_foldername(self,
                       data_raw):
        fold_keys = data_raw.keys()
        train_name = [x for x in fold_keys if x.startswith("train")][0]
        dev_name = [x for x in fold_keys if x.startswith("validation")][0]
        test_name = [x for x in fold_keys if x.startswith("test")][0]
        return train_name, dev_name, test_name

    def wrapper(self,
                 func,
                 *args):  # with star
        return func(*args)

    def prepare_data(self,
                     dataset_config,
                     model_name,
                     search_space_dir = None,
                     submit_mode="resplit",
                     output_path = "./",
                     split_portion=None):

        assert isinstance(dataset_config, dict) and "task" in dataset_config and "dataset_name" in dataset_config

        task_name = dataset_config["task"]
        dataset_name = dataset_config["dataset"]
        if "subdataset_name" in dataset_config:
            subdataset_name = dataset_config["subdataset_name"]
        else:
            subdataset_name = None
        if "input_path" in dataset_config:
            input_path = dataset_config["input_path"]
        else:
            input_path = None

        assert (input_path and search_space_dir) or (dataset_name)

        self.args = AutoHFArguments(
                    output_path=output_path,
                    dataset_config = dataset_config,
                    dataset_name = dataset_name,
                    subdataset_name = subdataset_name,
                    task_name = task_name,
                    model_name= model_name,
                    submit_mode = submit_mode,
        )
        self.args.init_and_make_dirs(search_space_dir)

        assert (submit_mode == "resplit" and split_portion) or (submit_mode == "origin" and not split_portion)

        self._tokenizer = AutoTokenizer.from_pretrained(self.args.model_checkpoint, use_fast=True)

        if not input_path:
            import importlib
            self.dataset_module = importlib.import_module("flaml.nlp.dataset." + dataset_name)
            if subdataset_name:
                data_raw = load_dataset(dataset_name[0], subdataset_name)
            else:
                data_raw = self.wrapper(load_dataset, *dataset_name)
        else:
            assert os.path.isdir(input_path), "input path format incorrect"
            data_raw = load_dataset(input_path)

        train_name, dev_name, test_name = self.get_foldername(data_raw)
        sentence_keys = self.get_sentence_keys(data_raw)

        data_encoded = data_raw.map(partial(self._tokenize, sentence_keys= sentence_keys, batched=True))

        if submit_mode == "resplit":
            assert split_portion, "in resplit mode but no split proportion given "
            train_dataset, val_dataset = data_encoded[train_name], data_encoded[dev_name]
            data_train_dev = datasets.concatenate_datasets([train_dataset, val_dataset])
            data_train_dev = data_train_dev.shuffle(seed=42)

            train_start, train_end = int(split_portion[0] * len(data_train_dev)), int(
                split_portion[1] * len(data_train_dev))
            dev_start, dev_end = int(split_portion[0] * len(data_train_dev)), int(
                split_portion[1] * len(data_train_dev))
            test_start, test_end = int(split_portion[0] * len(data_train_dev)), int(
                split_portion[1] * len(data_train_dev))

            train_dataset = data_train_dev.select([x for x in range(train_start, train_end)]).flatten_indices()
            eval_dataset = data_train_dev.select([x for x in range(dev_start, dev_end)]).flatten_indices()
            test_dataset = data_train_dev.select([x for x in range(test_start, test_end)]).flatten_indices()
        else:
            train_dataset, eval_dataset, test_dataset = data_encoded[train_name], data_encoded[dev_name], data_encoded[
                test_name]

        if self.args.task_name == "text-classification":
            self.args.num_labels = len(self.args.train_dataset.features["label"].names)
        elif self.args.task_name == "regression":
            self.args.num_labels = 1

        return train_dataset, eval_dataset, test_dataset

    def _load_model(self):
        if self.args.task_name == "text-classification":
            model_config = AutoConfig.from_pretrained(self.args.model_checkpoint)
            num_labels_old = model_config.num_labels
            model_type = model_config.get_config_dict(self.args.model_checkpoint)[0]["model_type"]
            self.args.model_type = model_type

            if self.args.num_labels != num_labels_old:
                this_model = AutoModelForSequenceClassification.from_pretrained(self.args.model_checkpoint, num_labels = num_labels_old)
                this_model.num_labels = self.args.num_labels
                this_model.classifier = AutoClassificationHead.from_config(model_config)
            else:
                this_model = AutoModelForSequenceClassification.from_pretrained(self.args.model_checkpoint, num_labels = num_labels_old)
            this_model.resize_token_embeddings(len(self._tokenizer))
            return this_model

    def compute_metrics(self,
                        eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        if self.args.dataset_name in ("glue", "super_glue"):
            metric = datasets.load.load_metric(self.args.dataset_name, self.args.subdataset_name)
        elif self.args.dataset_name in ("squad", "squad_v2"):
            metric = datasets.load.load_metric(self.args.dataset_name)
        else:
            assert self.args.metric_name
            metric = datasets.load.load_metric(self.args.metric_name)

        return metric.compute(predictions=predictions, references=labels)

    def _objective(self,
                  config,
                  reporter,
                  checkpoint_dir=None):

        this_model = self._load_model()

        training_args = TrainingArguments(
            output_dir=self.args._ckpt_dir_abs,
            do_eval=False,
            disable_tqdm=True,
            logging_steps=20000,
            save_total_limit=0,
            fp16=True,
            **config,
        )

        trainer = Trainer(
            this_model,
            training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self._tokenizer,
            compute_metrics= self.compute_metrics,
        )

        # train model
        trainer.train()

        # evaluate model
        eval_output = trainer.evaluate()

        flaml.tune.report(
            loss=eval_output["eval_loss"],
            accuracy=eval_output["eval_accuracy"],
        )

    def _get_hpo_algo(self,
                    hpo_method):
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
            algo = ZOOptSearch()
        elif 'Ax' == hpo_method:
            from ray.tune.suggest.ax import AxSearch
            algo = AxSearch(max_concurrent=3)
        elif 'HyperOpt' == hpo_method:
            from ray.tune.suggest.hyperopt import HyperOptSearch
            algo = HyperOptSearch()
        else:
            algo = None

        return algo

    def _get_scheduler(self,
                       scheduler_name):
        if scheduler_name == "ASHA":
            from ray.tune.schedulers import ASHAScheduler
            scheduler = ASHAScheduler(
                max_t= 100,
                grace_period=1)
        else:
            scheduler = None
        return scheduler

    def fit(self,
            train_dataset,
            eval_dataset,
            test_dataset,
            metric_name,
            mode_name,
            wandb_key = None,
            hpo_method="bs",
            scheduler_name=None,
            num_samples=10000,
            time_budget=7200,
            device_nums=None):

        #assert self.args.dataset_name in ("glue", "squad_v2", "squad", "super_glue") or metric_name

        self.args.hpo_method = hpo_method

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset

        if not device_nums:
            device_nums = {"cpu": 4, "gpu": 4}

        ray.init(num_cpus=device_nums["cpu"], num_gpus=device_nums["gpu"])

        hpo_algo = self._get_hpo_algo(hpo_method)
        scheduler = self._get_scheduler(scheduler_name)

        if wandb_key:
            self._set_wandb(wandb_key)

        self._set_search_space()

        start_time = time.time()

        analysis = ray.tune.run(
            self._objective,
            metric= metric_name,
            mode = mode_name,
            resources_per_trial={"gpu": 1, "cpu": 1},
            config= self.search_space_hpo,
            local_dir= self.args._ckpt_dir_abs,
            num_samples = num_samples,
            time_budget_s= time_budget,
            keep_checkpoints_num = 1,
            checkpoint_score_attr= metric_name,
            scheduler= scheduler,
            search_alg= hpo_algo)

        ray.shutdown()

        best_trial = analysis.get_best_trial(metric_name, mode_name, "all")
        metric = best_trial.metric_analysis[metric_name][mode_name]

        # import logging
        # logger = logging.getLogger(__name__)
        #
        # logger.info(f"method={hpo_method}")
        # logger.info(f"n_trials={len(analysis.trials)}")
        # logger.info(f"time={time.time() - start_time}")
        # logger.info(f"Best model eval {metric_name}: {metric:.4f}")
        # logger.info(f"Best model parameters: {best_trial.config}")