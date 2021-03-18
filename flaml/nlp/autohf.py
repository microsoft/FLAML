import os,json

try:
    import ray
    import transformers
except:
    print("pip install torch transformers datasets flaml[blendsearch,ray]")

import wandb
import numpy as np

from ray.tune import CLIReporter
from datasets import Dataset

import time
import ray
import datasets
from datasets import load_dataset
from transformers.trainer_utils import IntervalStrategy

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from functools import partial

from flaml.nlp.path_utils import PathUtils
from flaml.nlp.searchalgo_auto import AutoSearchAlgorithm
from flaml.nlp.modeling_auto import AutoSeqClassificationHead
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from flaml.nlp.trainer_for_autohf import TrainerForAutoHF
from flaml.nlp.training_args import TuneTrainingArguments

task_list = [
    "text-classification",
    "regression",
    "question-answering"
]

class AutoHuggingFace:

    '''The AutoHuggingFace class

    Example:

        .. code-block:: python

            autohf = AutoHuggingFace()
            autohf_settings = {"metric_name": "accuracy",
                   "mode_name": "max",
                   "resources_per_trial": {"gpu": 4, "cpu": 4},
                   "wandb_key": wandb_key,
                   "search_algo": method,
                   "num_samples": 4,
                   "time_budget": 7200,
                   "points_to_evaluate": [{
                       "num_train_epochs": 1,
                       "per_device_train_batch_size": 128, }]
                   }

            autohf.fit(train_dataset,
                       eval_dataset,
                       **autohf_settings,)

    '''

    task_name: str = field(default="text-classification", metadata={"help": "task name"})
    model_type: str = field(default="electra", metadata={"help": "model type."})
    submit_mode: str = field(default="resplit", metadata={"help": "The submit mode."})

    scheduler_name: str = field(default="asha", metadata={"help": "The scheduler name."})
    search_algo: str = field(default="bs", metadata={"help": "The hpo method."})

    num_labels: Optional[int] = field(default=2, metadata={"help": "number of labels"})

    metric_name: str = field(default=None, metadata={"help": "metric name"})

    max_seq_length: Optional[int] = field(default=128, metadata={"help": "max seq length"})

    def _set_wandb(self,
                   wandb_key):
        os.environ["WANDB_API_KEY"] = wandb_key
        self.path_utils.group_hash_id = wandb.util.generate_id()
        group_name = self.path_utils.folder_name + "_" + self.path_utils.group_hash_id
        os.environ["WANDB_RUN_GROUP"] = group_name

    def _convert_json_to_search_space(self,
                                      config_json,
                                      mode = "grid_search"):
        search_space = {}

        if mode == "grid_search":
            for each_hp in config_json.keys():
                this_config = config_json[each_hp]
                assert isinstance(this_config, dict) or isinstance(this_config, list), "config of " + each_hp + " must be dict or list"
                search_space[each_hp] = ray.tune.grid_search(this_config)
        else:
            for each_hp in config_json.keys():
                this_config = config_json[each_hp]
                assert isinstance(this_config, dict) or isinstance(this_config, list), "config of " + each_hp + " must be dict or list"
                if isinstance(this_config, dict):
                    lower = this_config["l"]
                    upper = this_config["u"]
                    space = this_config["space"]
                    if space == "log":
                        search_space[each_hp] = ray.tune.loguniform(lower, upper)
                    elif space == "linear":
                        search_space[each_hp] = ray.tune.uniform(lower, upper)
                    elif space == "quniform":
                        search_space[each_hp] = ray.tune.quniform(lower, upper, this_config["interval"])
                else:
                    search_space[each_hp] = ray.tune.choice(this_config)

        return search_space

    def _set_search_space(self,
                          search_space_dir=None):
        assert self.model_type
        if not search_space_dir:
            search_space_dir = self.path_utils.get_search_space_dir(self.model_type)

        search_space_grid_json = json.load(open(os.path.join(search_space_dir, self.model_type + "_grid.json"), "r"))
        self.search_space_grid = self._convert_json_to_search_space(search_space_grid_json, mode="grid_search")

        if self.search_algo != "grid_search":
            search_space_hpo_json = json.load(open(os.path.join(search_space_dir, self.model_type + "_hpo.json"), "r"))
            self.search_space_hpo = self._convert_json_to_search_space(search_space_hpo_json, mode="hpo")

    #TODO: try remove the requirement on mapping
    @property
    def _eval_acc_name(self):
        return self.dataset_module.eval_name_mapping[self.path_utils.task_name][0]

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
        return self._tokenizer(*args, padding="max_length", max_length=self.max_seq_length, truncation=True)

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

    def _get_sentence_keys(self,
                           data_raw):
        return [x for x in data_raw["train"].features.keys() if x not in ("label", "idx")]

    def _get_foldername(self,
                        data_raw):
        fold_keys = data_raw.keys()
        train_name = [x for x in fold_keys if x.startswith("train")][0]
        dev_name = [x for x in fold_keys if x.startswith("validation")][0]
        test_name = [x for x in fold_keys if x.startswith("test")][0]
        return train_name, dev_name, test_name

    def _wrapper(self,
                 func,
                 *args):  # with star
        return func(*args)

    def prepare_data(self,
                     dataset_config,
                     model_name,
                     split_mode,
                     output_path,
                     max_seq_length = 128,
                     resplit_portion=None):
        '''Prepare data

            Args:
                dataset_config:
                    a dict for data specification, it must contain two keys:
                        -- "task": the task name, e.g., "text-classification" "question-answering"
                        -- "dataset_name": the dataset name, must be one of the dataset name in huggingface, or "custom"
                        -- "input_path": the custom path for specifying the custom dataset, must be specified if dataset_name = "custom"
                        -- "subdataset_name": the sub dataset name, e.g., "glue", "qnli". Not required.
                    e.g., {"task": "text-classification",
                            "dataset_name": ["glue"],
                            "subdataset_name": "rte"}
                model_name:
                    the model name path under huggingface.co/models
                    e.g., "google/electra-base-discriminator"
                split_mode:
                    the mode for splitting the dataset, must be one of two:
                    -- "resplit": mixing train and dev, then resplit them into a proportion defined by the resplit_portion parameter, this
                     mode is mostly for resplitting glue after considering the overfitting problem in the few-sample subdatasets, e.g., RTE, MRPC, SST, QNLI
                    -- "origin": keep the original train/dev/test split.
                output_path:
                    the root path for outputting the checkpoints and evaluation results
                max_seq_length:
                    max_seq_length for the model, this hyperparameter must be specified at the data processing step
                resplit_portion:
                    the proportion for resplitting the train and dev data when split_mode="resplit". Not required.
            '''

        assert isinstance(dataset_config, dict) and ("task" in dataset_config) and ("dataset_name" in dataset_config)
        assert dataset_config["task"] in task_list

        task_name = dataset_config["task"]
        dataset_name = dataset_config["dataset_name"]
        if "subdataset_name" in dataset_config:
            subdataset_name = dataset_config["subdataset_name"]
        else:
            subdataset_name = None
        if "input_path" in dataset_config:
            input_path = dataset_config["input_path"]
        else:
            input_path = None

        assert (input_path) or (subdataset_name)

        self.task_name = task_name
        self.submit_mode = split_mode
        self.max_seq_length = max_seq_length

        self.path_utils = PathUtils(
                    hpo_output_dir = output_path,
                    dataset_name = dataset_name,
                    subdataset_name = subdataset_name,
                    model_name= model_name,
        )
        self.path_utils.init_and_make_dirs()

        assert (split_mode == "resplit" and resplit_portion) or (split_mode == "origin" and not resplit_portion)

        self._tokenizer = AutoTokenizer.from_pretrained(self.path_utils.model_checkpoint, use_fast=True)

        if not input_path:
            import importlib
            self.dataset_module = importlib.import_module("flaml.nlp.dataset." + dataset_name[0])
            if subdataset_name:
                data_raw = load_dataset(dataset_name[0], subdataset_name)
            else:
                data_raw = self._wrapper(load_dataset, *dataset_name)
        else:
            assert os.path.isdir(input_path), "input path format incorrect"
            data_raw = load_dataset(input_path)

        train_name, dev_name, test_name = self._get_foldername(data_raw)
        sentence_keys = self._get_sentence_keys(data_raw)

        data_encoded = data_raw.map(partial(self._tokenize, sentence_keys= sentence_keys), batched=True)

        if split_mode == "resplit":
            assert resplit_portion, "in resplit mode but no split proportion given "
            train_dataset, val_dataset = data_encoded[train_name], data_encoded[dev_name]
            data_train_dev = datasets.concatenate_datasets([train_dataset, val_dataset])
            data_train_dev = data_train_dev.shuffle(seed=42)

            train_start, train_end = int(resplit_portion["train"][0] * len(data_train_dev)), int(resplit_portion["train"][1] * len(data_train_dev))
            dev_start, dev_end = int(resplit_portion["dev"][0] * len(data_train_dev)), int(resplit_portion["dev"][1] * len(data_train_dev))
            test_start, test_end = int(resplit_portion["test"][0] * len(data_train_dev)), int(resplit_portion["test"][1] * len(data_train_dev))

            train_dataset = data_train_dev.select([x for x in range(train_start, train_end)]).flatten_indices()
            eval_dataset = data_train_dev.select([x for x in range(dev_start, dev_end)]).flatten_indices()
            test_dataset = data_train_dev.select([x for x in range(test_start, test_end)]).flatten_indices()
        else:
            train_dataset, eval_dataset, test_dataset = data_encoded[train_name], data_encoded[dev_name], data_encoded[
                test_name]

        if self.task_name == "text-classification":
            self.num_labels = len(train_dataset.features["label"].names)
        elif self.task_name == "regression":
            self.num_labels = 1

        return train_dataset, eval_dataset, test_dataset

    def _set_model_type(self):
        model_config = AutoConfig.from_pretrained(self.path_utils.model_checkpoint)
        model_type = model_config.get_config_dict(self.path_utils.model_checkpoint)[0]["model_type"]
        self.model_type = model_type

    def _load_model(self):
        if self.task_name == "text-classification":
            model_config = AutoConfig.from_pretrained(self.path_utils.model_checkpoint)
            num_labels_old = model_config.num_labels

            if self.num_labels != num_labels_old:
                this_model = AutoModelForSequenceClassification.from_pretrained(self.path_utils.model_checkpoint, num_labels = num_labels_old)
                this_model.num_labels = self.num_labels
                this_model.classifier = AutoSeqClassificationHead.from_config(model_config)
            else:
                this_model = AutoModelForSequenceClassification.from_pretrained(self.path_utils.model_checkpoint, num_labels = num_labels_old)
            this_model.resize_token_embeddings(len(self._tokenizer))
            return this_model

    def _compute_metrics_by_dataset_name(self,
                                         eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        if self.path_utils.dataset_name[0] in ("glue", "super_glue"):
            metric = datasets.load.load_metric(self.path_utils.dataset_name[0], self.path_utils.subdataset_name)
        elif self.path_utils.dataset_name[0] in ("squad", "squad_v2"):
            metric = datasets.load.load_metric(self.path_utils.dataset_name[0])
        else:
            assert self.metric_name
            metric = datasets.load.load_metric(self.metric_name)

        return metric.compute(predictions=predictions, references=labels)

    def _compute_checkpoint_freq(self,
                                 num_train_epochs,
                                 batch_size,
                                 mode="last"):
        assert mode in ("last")
        if mode == "last":
            ckpt_step_freq = int(num_train_epochs * len(self.train_dataset) / batch_size / self.resources_per_trial["gpu"]) + 1

        return ckpt_step_freq

    def _objective(self,
                  config,
                  reporter,
                  checkpoint_dir=None):

        this_model = self._load_model()

        trial_id = reporter.trial_id
        self.path_utils.make_dir_per_trial(trial_id)

        ckpt_freq = self._compute_checkpoint_freq(
            num_train_epochs = config["num_train_epochs"],
            batch_size = config["per_device_train_batch_size"],
            mode="last")

        assert self.path_utils._ckpt_dir_per_trial
        training_args = TuneTrainingArguments(
            output_dir=self.path_utils._ckpt_dir_per_trial,
            do_eval=False,
            per_device_eval_batch_size=32,
            eval_steps= ckpt_freq,
            evaluation_strategy = IntervalStrategy.STEPS,
            save_steps= ckpt_freq,
            save_total_limit=0,
            fp16=True,
            **config,
        )

        trainer = TrainerForAutoHF(
            this_model,
            training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self._tokenizer,
            compute_metrics= self._compute_metrics_by_dataset_name,
        )

        trainer.train()

    def _verify_init_config(self,
                            **search_algo_args):
        for key in search_algo_args.keys():
            if key == "points_to_evaluate":
                for each_init_config in search_algo_args[key]:
                   for each_hp in each_init_config.keys():
                       assert each_hp in self.search_space_hpo.keys(), \
                           "points_to_evaluate hp must be within the search space"

                       assert isinstance(each_init_config[each_hp], int) or \
                              isinstance(each_init_config[each_hp], float) or \
                              isinstance(each_init_config[each_hp], str) or \
                              isinstance(each_init_config[each_hp], bool), " points_to_evaluate must be a scalar"

                       assert isinstance(self.search_space_hpo[each_hp], ray.tune.sample.Categorical) or \
                              isinstance(self.search_space_hpo[each_hp], ray.tune.sample.Float)

                       if isinstance(self.search_space_hpo[each_hp], ray.tune.sample.Categorical):
                           assert each_init_config[each_hp] in self.search_space_hpo[each_hp].categories, \
                               f"points_to_evaluate {each_hp} value must be within the search space"
                       else:
                           assert each_init_config[each_hp] >= self.search_space_hpo[each_hp].lower \
                                  and each_init_config[each_hp] <= self.search_space_hpo[each_hp].upper, \
                               f"points_to_evaluate {each_hp} value must be within the search space"

    def _get_hpo_algo(self,
                      search_algo,
                      **search_algo_args):

        self._verify_init_config(**search_algo_args)
        if search_algo_args:
            search_algo = AutoSearchAlgorithm.from_config_and_method_name(search_algo, **search_algo_args)
        else:
            search_algo = AutoSearchAlgorithm.from_config_and_method_name(search_algo)
        return search_algo

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

    def _recover_checkpoint(self,
                            tune_checkpoint_dir):
        assert tune_checkpoint_dir
        # Get subdirectory used for Huggingface.
        subdirs = [
            os.path.join(tune_checkpoint_dir, name)
            for name in os.listdir(tune_checkpoint_dir)
            if os.path.isdir(os.path.join(tune_checkpoint_dir, name))
        ]
        # There should only be 1 subdir.
        assert len(subdirs) == 1, subdirs
        return subdirs[0]

    def save_ckpt_json(self,
                       best_ckpt):
        json.dump({"best_ckpt": best_ckpt}, open(os.path.join(self.path_utils._result_dir_per_run, "result_" + self.path_utils.folder_name + ".json"), "w"))

    def load_ckpt_json(self,
                       ckpt_dir = None):
        if not ckpt_dir:
            assert self.path_utils.folder_name
            ckpt_dir = os.path.join(self.path_utils._result_dir_per_run, "result_" + self.path_utils.folder_name + ".json")
        ckpt_json = json.load(open(ckpt_dir))
        return ckpt_json["best_ckpt"]

    def fit(self,
            train_dataset,
            eval_dataset,
            metric_name,
            mode_name,
            resources_per_trial,
            wandb_key = "f38cc048c956367de27eeb2749c23e6a94519ab8",
            search_algo="bs",
            search_space_path = None,
            scheduler_name=None,
            num_samples=10000,
            time_budget=7200,
            **search_algo_kwargs):
        '''Fine tuning the model using the hpo setting

        Args:
            train_dataset:
                the training data of type datasets.Dataset, loaded from datasets.load_dataset
            eval_dataset:
                the validation data of type datasets.Dataset, loaded from datasets.load_dataset
            metric_name:
                A string of the metric name or a function,
                e.g., 'accuracy', 'f1', 'loss',
                if passing a customized metric function, the function needs to
                have the follwing signature:

                .. code-block:: python

                    def custom_metric(X_test, y_test, estimator, labels,
                     X_train, y_train, weight_test=None, weight_train=None):
                        return metric_to_minimize, metrics_to_log

                which returns a float number as the minimization objective,
                and a tuple of floats as the metrics to log
            mode_name:
                A string of the mode name,
                e.g., "max", "min", "last", "all"
            resources_per_trial:
                A dict showing the resources used by each trial,
                e.g., {"gpu": 4, "cpu": 4}
            wandb_key:
                The hash code for wandb
            search_algo:
                The search algoritihm for AutoHF()
                e.g., "blendsearch" "cfo" "bo"
            search_space_path:
                a path for the json file for search space,
                e.g., search_space_path = "./search_space/electra/"
            scheduler_name:
                A string of the scheduler name,
                e.g., "ASHA", "HyperBand"
            num_samples:
                An int variable of the maximum number of trials
            time_budget:
                An int variable of the maximum time budget
            search_algo_kwargs:
                The keyword arguments to be fed into the search algorith, e.g.,
                search_algo_kwargs = {"points_to_evaluate": [{
                           "num_train_epochs": 1,
                           "per_device_train_batch_size": 128, }]}
        '''

        self.path_utils.search_algo = search_algo
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.resources_per_trial = resources_per_trial
        self.search_algo = search_algo
        self.scheduler_name = scheduler_name
        self.metric_name = metric_name

        ray.init(local_mode=True)

        self._set_model_type()
        self._set_search_space(search_space_path)

        self.path_utils.set_folder_name(search_algo, scheduler_name, self.model_type, self.submit_mode)

        search_algo = self._get_hpo_algo(search_algo, **search_algo_kwargs)
        scheduler = self._get_scheduler(scheduler_name)

        self._set_wandb(wandb_key)
        self.path_utils.make_dir_per_run()

        start_time = time.time()

        assert self.path_utils._ckpt_dir_per_run

        analysis = ray.tune.run(
            self._objective,
            metric= metric_name,
            mode = mode_name,
            name = "ray_result",
            resources_per_trial = resources_per_trial,
            config= self.search_space_hpo,
            local_dir= self.path_utils._ckpt_dir_per_run,
            num_samples = num_samples,
            time_budget_s= time_budget,
            keep_checkpoints_num = 1,
            scheduler= scheduler,
            search_alg= search_algo)

        ray.shutdown()

        best_trial = analysis.get_best_trial(scope="all", metric= metric_name, mode= mode_name)
        metric = best_trial.metric_analysis[metric_name][mode_name]

        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"method={search_algo}")
        logger.info(f"n_trials={len(analysis.trials)}")
        logger.info(f"time={time.time() - start_time}")
        logger.info(f"Best model eval {metric_name}: {metric:.4f}")
        logger.info(f"Best model parameters: {best_trial.config}")

        get_best_ckpt = analysis.get_best_checkpoint(best_trial, metric=metric_name, mode=mode_name)
        best_ckpt = self._recover_checkpoint(get_best_ckpt)

        self.save_ckpt_json(best_ckpt)

    def predict(self,
                test_dataset,
                ckpt_json_dir = None):
        '''Predict label for test data.

        Args:
            test_dataset:
                the test dataset
            ckpt_json_dir:
                the checkpoint for the fine-tuned model if you wish to override the saved checkpoint in the training stage under self.path_utils._result_dir_per_run

        Returns:
            A numpy array of shape n * 1 - - each element is a predicted class
            label for an instance.
        '''

        best_checkpoint = self.load_ckpt_json(ckpt_json_dir)

        best_model = AutoModelForSequenceClassification.from_pretrained(
            best_checkpoint).to("cuda")

        test_trainer = transformers.Trainer(best_model)

        predictions = test_trainer.predict(test_dataset=test_dataset).predictions
        predictions = np.squeeze(predictions) if self.task_name == "regression" else np.argmax(predictions, axis=1)

        return predictions