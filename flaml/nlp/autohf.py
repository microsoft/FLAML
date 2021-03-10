import os,json
import wandb

import ray
import datasets
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, glue_tasks_num_labels, AutoConfig
from transformers import Trainer
from functools import partial

from flaml.nlp.hpo_training_args import AutoHFArguments

task_list = [
    "text-classification",
    "question-answering"
]

class AutoHuggingFace:

    # def __init__(self,
    #              task_name,
    #              abs_data_path,
    #              model_name_short = None,
    #              hpo_method = None,
    #              scheduler_name = None,
    #              submit_mode = "resplit",
    #              split_portion = None,
    #              wandb_key = None,
    # ):
    #     super()
    #
    #     self._task_name = task_name
    #     self._model_name_short = model_name_short
    #     self._abs_data_path = abs_data_path
    #     self._hpo_method = hpo_method or 'bs'
    #     self._scheduler_name = scheduler_name
    #     self._submit_mode = submit_mode
    #     self._split_portion = split_portion or {
    #                               "train": (0.0, 0.8),
    #                               "dev": (0.8, 0.9),
    #                               "test": (0.9, 1.0)}
    #
    #     self._set_folder_name()
    #     self._init_paths()
    #     if wandb_key:
    #         self._set_wandb(wandb_key)
    #
    #     self.tokenizer = AutoTokenizer.from_pretrained(self._model_dir_abs, use_fast=True)
    #     self._set_search_space()
    #     self._set_hp_metric()

    def _set_folder_name(self):
        self._folder_name = self._hpo_method + "_" + self._scheduler_name + self._model_name_short + "_" + self._submit_mode

    def _set_wandb(self,
                   wandb_key):
        os.environ["WANDB_API_KEY"] = wandb_key
        generated_id = wandb.util.generate_id()
        group_name = self._folder_name + "_" + generated_id
        os.environ["WANDB_RUN_GROUP"] = group_name

    def _set_search_space(self):
        self.search_space_grid = json.load(open(os.path.join(self._search_space_dir, self._task_name + "_grid.json", "r")))
        self.search_space_hpo = json.load(open(os.path.join(self._search_space_dir, self._task_name + "_hpo.json", "r")))

    @property
    def eval_acc_name(self):
        return task_to_eval_name[self._task_name]

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

    def prepare_data(self,
                     submit_mode="resplit",
                     output_path = "./",
                     model_name = "electra",
                     dataset_config = ("text-classification", "glue", "qnli"),
                     sentence_keys=None,
                     split_portion=None):

        assert isinstance(dataset_config, tuple) and (len(dataset_config) in (2, 3)) and dataset_config[0] in task_list

        self.args = AutoHFArguments(
                    output_path=output_path,
                    dataset_config = dataset_config,
                    dataset_name = dataset_config[1],
                    model_name=model_name,
                    submit_mode = submit_mode,
        )
        self.args.init_path()

        assert (submit_mode == "resplit" and split_portion) or (submit_mode == "origin" and not split_portion)

        self._tokenizer = AutoTokenizer.from_pretrained(self.args.model_checkpoint, use_fast=True)

        if dataset_config[1] != "custom":
            new_module = __import__(modulename)

            glue_task_name = dataset_config[1]
            dev_name = "validation" if glue_task_name != "mnli" else "validation_matched"
            test_name = "test" if glue_task_name != "mnli" else "test_matched"
            data_raw = load_dataset("glue", dataset_config[1])
            data_encoded = data_raw.map(partial(self._tokenize, sentence_keys=glue_task_to_key[glue_task_name]), batched=True)
        #TODO: squad may have a different mapping function, may need modification below
        elif dataset_config[0] == "hf":
            dataset_name = dataset_config[1]
            data_raw = load_dataset(dataset_name)
            dev_name = "validation"
            test_name = "test"
            data_encoded = data_raw.map(partial(self._tokenize, sentence_keys=hf_data_to_key[dataset_name]), batched=True)
        else:
            input_path = dataset_config[1]
            assert os.path.isdir(input_path)
            data_raw = load_dataset(input_path)
            dev_name = "validation"
            test_name = "test"
            data_encoded = data_raw.map(partial(self._tokenize, sentence_keys=sentence_keys), batched=True)

        if submit_mode == "resplit":
            assert split_portion, "in resplit mode but no split proportion given "
            train_dataset, val_dataset = data_encoded["train"], data_encoded[dev_name]
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
            train_dataset, eval_dataset, test_dataset = data_encoded["train"], data_encoded[dev_name], data_encoded[
                test_name]

        self.args.train_dataset = train_dataset
        self.args.eval_dataset = eval_dataset
        self.args.test_dataset = test_dataset

        if dataset_config[0] == "glue":
            self.args.num_labels = len(self.args.train_dataset.features["label"].names)

    def _load_model(self):

        if self.args.dataset_config[0] == "glue":
            glue_task_name = self.args.dataset_config[1]

            glue_tasks_num_labels["stsb"] = glue_tasks_num_labels["sts-b"]
            del glue_tasks_num_labels["sts-b"]

            #num_labels = glue_tasks_num_labels[glue_task_name]
            num_labels_old = AutoConfig.from_pretrained(self.args.model_checkpoint).num_labels

            if self.args.num_labels != num_labels_old:
                model_config.num_labels = num_labels_old
                this_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path,
                    config=model_config,
                )
                model_config.num_labels = self.args.num_labels
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

    def fit(self,
            hpo_method="bs",
            scheduler_name=None,
            num_samples=10000,
            time_budget=7200,
            device_nums=None):

        this_model = AutoModelForSequenceClassification.from_pretrained(self.args.model_checkpoint,
                                                                        num_labels=self.args.num_labels)

        this_model = self.

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