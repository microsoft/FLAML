import json
import os
import numpy as np
import time

try:
    """Notice ray is required by flaml/nlp. This try except is for telling user to
    install flaml[nlp]. In future, if flaml/nlp contains a module that does not require
    ray, this line needs to be deleted. In addition, need to remove `ray[tune]` from
    setup.py: nlp. Further more, for all the test code for those modules (that does not
     require ray), need to remove the try...except before the test functions and address
     import errors in the library code accordingly."""
    import ray
    import transformers
    from transformers import TrainingArguments
    import datasets
    from .result_analysis.azure_utils import JobID
    from .huggingface.trainer import TrainerForAutoTransformers
    from typing import Optional, Tuple, List, Union, Dict
    from ray.tune.trial import Trial
    import argparse
except ImportError:
    raise ImportError("To use the nlp component in flaml, run pip install flaml[nlp]")

task_list = ["seq-classification", "regression", "question-answering"]


class AutoTransformers:
    """The AutoTransformers class

    Example:

        .. code-block:: python

            autohf = AutoTransformers()
            autohf_settings = {
                "dataset_config": ["glue", "mrpc"],
                "pretrained_model": "google/electra-small-discriminator",
                "output_dir": "data/",
                "resources_per_trial": {"cpu": 1, "gpu": 1},
                "num_samples": -1,
                "time_budget": 300,
            }
            validation_metric, analysis = autohf.fit(**autohf_settings)

    """

    @staticmethod
    def _convert_dict_to_ray_tune_space(config_json, mode="grid"):
        search_space = {}

        if mode == "grid":
            for each_hp, val in config_json.items():
                assert isinstance(val, dict) or isinstance(val, list), (
                    "config of " + each_hp + " must be dict or list for grid search"
                )
                search_space[each_hp] = ray.tune.grid_search(val)
        else:
            for each_hp, val in config_json.items():
                assert isinstance(val, dict) or isinstance(val, list), (
                    "config of " + each_hp + " must be dict or list"
                )
                if isinstance(val, dict):
                    lower = val["l"]
                    upper = val["u"]
                    space = val["space"]
                    if space == "log":
                        search_space[each_hp] = ray.tune.loguniform(lower, upper)
                    elif space == "linear":
                        search_space[each_hp] = ray.tune.uniform(lower, upper)
                    elif space == "quniform":
                        search_space[each_hp] = ray.tune.quniform(
                            lower, upper, val["interval"]
                        )
                else:
                    search_space[each_hp] = ray.tune.choice(val)

        return search_space

    def _set_search_space(self):
        from .hpo.hpo_searchspace import AutoHPOSearchSpace
        from .hpo.grid_searchspace_auto import GRID_SEARCH_SPACE_MAPPING

        if self.jobid_config.mod == "grid":
            if self.custom_hpo_args.grid_space_model_type is None:
                if self.jobid_config.pre in GRID_SEARCH_SPACE_MAPPING:
                    setattr(
                        self.custom_hpo_args,
                        "grid_space_model_type",
                        self.jobid_config.pre,
                    )
                else:
                    # grid mode is not designed for user to use. Therefore we only allow
                    # grid search in the following two cases: (1) if
                    # user has specified a model, and the model contains a grid space in FLAML
                    # or (2) the user has specified which model's grid space to use
                    raise NotImplementedError(
                        f"The grid space for {self.jobid_config.pre} is not implemented in FLAML."
                    )
        else:
            setattr(
                self.custom_hpo_args, "grid_space_model_type", self.jobid_config.pre
            )

        if self.custom_hpo_args.custom_search_space is not None:
            self.jobid_config.spa = "cus"

        search_space_hpo_json = AutoHPOSearchSpace.from_model_and_dataset_name(
            hpo_searchspace_mode=self.jobid_config.spa,
            model_type=self.custom_hpo_args.grid_space_model_type,
            model_size_type=self.jobid_config.presz,
            dataset_name_list=self.jobid_config.dat,
            algo_mode=self.jobid_config.mod,
            custom_search_space=self.custom_hpo_args.custom_search_space,
        )
        self._search_space_hpo = AutoTransformers._convert_dict_to_ray_tune_space(
            config_json=search_space_hpo_json, mode=self.jobid_config.mod
        )

        if self.jobid_config.spa == "grid":
            electra_space = AutoTransformers._convert_dict_to_ray_tune_space(
                AutoHPOSearchSpace.from_model_and_dataset_name(
                    hpo_searchspace_mode="grid",
                    model_type="electra",
                    model_size_type="base",
                    dataset_name_list=self.jobid_config.dat,
                    algo_mode="grid",
                )
            )

            for key, value in electra_space.items():
                if key not in self._search_space_hpo:
                    self._search_space_hpo[key] = value

    def _get_start_end_bound(
        self,
        concatenated_data=None,
        resplit_portion_key=None,
        lower_bound_portion=0.0,
        upper_bound_portion=1.0,
    ):
        data_len = len(concatenated_data)
        target_fold_start, target_fold_end = int(
            resplit_portion_key[0] * data_len
        ), int(resplit_portion_key[1] * data_len)
        lower_bound, upper_bound = int(lower_bound_portion * data_len), int(
            upper_bound_portion * data_len
        )
        return target_fold_start, target_fold_end, lower_bound, upper_bound

    def _get_targetfold_start_end(
        self,
        concatenated_data=None,
        resplit_portion_key=None,
        lower_bound_portion=0.0,
        upper_bound_portion=1.0,
    ):
        def crange(start_pos, end_pos, lower_bound, upper_bound):
            from itertools import chain

            assert (
                start_pos >= lower_bound and end_pos <= upper_bound
            ), "start and end portion must be within [lower_bound, upper_bound]"
            if start_pos <= end_pos:
                return range(start_pos, end_pos)
            else:
                return chain(range(start_pos, upper_bound), range(lower_bound, end_pos))

        (
            target_fold_start,
            target_fold_end,
            lower_bound,
            upper_bound,
        ) = self._get_start_end_bound(
            concatenated_data,
            resplit_portion_key,
            lower_bound_portion,
            upper_bound_portion,
        )
        subfold_dataset = concatenated_data.select(
            crange(target_fold_start, target_fold_end, lower_bound, upper_bound)
        ).flatten_indices()
        return subfold_dataset

    def _is_custom_dataset(self, dataset_config):
        return (
            (dataset_config is not None)
            and ("path" in dataset_config)
            and (dataset_config["path"] in ("csv", "json"))
            and ("data_files" in dataset_config)
            and (len(dataset_config["data_files"]) > 0)
        )

    def _is_huggingface_dataset(self, dataset_config=None, data_mapping=None):
        return (
            (dataset_config is not None)
            and (data_mapping is not None)
            and ("path" in dataset_config)
        )

    @staticmethod
    def get_resplit_portion_key(key, split_portion):
        key2lb_id = {"train": 0, "validation": 2, "test": 4}
        key2ub_id = {"train": 1, "validation": 3, "test": 5}
        return [
            split_portion[key2lb_id[key]],
            split_portion[key2ub_id[key]],
        ]

    def autoencodetext_from_model_and_dataset_name(self, subfold_dataset):
        from .dataset.dataprocess_auto import AutoEncodeText

        tokenized_dat = AutoEncodeText.from_model_and_dataset_name(
            subfold_dataset=subfold_dataset,
            model_checkpoint_path=self.jobid_config.pre_full,
            dataset_name_list=self.jobid_config.dat,
            custom_sentence_keys=self.custom_hpo_args.custom_sentence_keys,
            **{"max_seq_length": self.custom_hpo_args.max_seq_length},
        )
        return tokenized_dat

    @staticmethod
    def _get_cv_split_points(lower_bound, upper_bound, idx, k):
        return (upper_bound - lower_bound) / k * idx + lower_bound, (
            upper_bound - lower_bound
        ) / k * (idx + 1) + lower_bound

    def _split_for_cross_validation(self, train_fold):
        train_val_lower = min(
            self.get_resplit_portion_key("train")[0],
            self.get_resplit_portion_key("validation")[0],
        )
        train_val_upper = max(
            self.get_resplit_portion_key("train")[1],
            self.get_resplit_portion_key("validation")[1],
        )
        cv_k = self.custom_hpo_args.cv_k
        eval_datasets = []
        train_datasets = []
        for idx in range(cv_k):
            this_fold_lower, this_fold_upper = AutoTransformers._get_cv_split_points(
                train_val_lower, train_val_upper, idx, cv_k
            )
            subfold_dataset = self._get_targetfold_start_end(
                concatenated_data=train_fold,
                resplit_portion_key=[this_fold_lower, this_fold_upper],
                lower_bound_portion=train_val_lower,
                upper_bound_portion=train_val_upper,
            )
            eval_datasets.append(
                self.autoencodetext_from_model_and_dataset_name(subfold_dataset)
            )
            subfold_dataset = self._get_targetfold_start_end(
                concatenated_data=train_fold,
                resplit_portion_key=[this_fold_upper, this_fold_lower],
                lower_bound_portion=train_val_lower,
                upper_bound_portion=train_val_upper,
            )
            train_datasets.append(
                self.autoencodetext_from_model_and_dataset_name(subfold_dataset)
            )
        return train_datasets, eval_datasets

    def _resplit_data(
        self,
        data_fold_for_split,
    ):
        data_fold_for_split = data_fold_for_split.shuffle(seed=self.jobid_config.sddt)

        for key in ["train", "validation", "test"]:
            subfold_dataset = self._get_targetfold_start_end(
                concatenated_data=data_fold_for_split,
                resplit_portion_key=AutoTransformers.get_resplit_portion_key(key),
            )
            this_encoded_data = self.autoencodetext_from_model_and_dataset_name(
                subfold_dataset
            )
            if key == "train":
                train_dataset = this_encoded_data
            elif key == "validation":
                eval_dataset = this_encoded_data
            else:
                test_dataset = this_encoded_data
        return train_dataset, eval_dataset, test_dataset

    def _reset_max_seq_length(self, train_dataset, eval_dataset, test_dataset):
        all_folds = [train_dataset, eval_dataset, test_dataset]
        _max_seq_length = 0

        _max_seq_length = max(
            _max_seq_length,
            max([sum(x["attention_mask"]) for x in all_folds]),
        )
        self.custom_hpo_args._max_seq_length = int((_max_seq_length + 15) / 16) * 16

    def unify_data_raw_and_data_mapping(
        self,
        data_raw,
        dataset_config,
        data_mapping,
    ):
        self.foldname_mapping = {}

        data_mapping_unified = {}
        if data_mapping is not None:
            assert isinstance(data_mapping, list) or isinstance(data_mapping, dict)
            if isinstance(data_mapping, list):
                assert set(data_mapping).issubset({"train", "validation", "test"}), (
                    "To select folds from a HuggingFace dataset, the folds must either be {'train', 'validation'}"
                    " or {'train'} "
                )
                for fold_key in data_mapping:
                    self.foldname_mapping[fold_key] = fold_key
                    data_mapping_unified[fold_key] = data_raw[fold_key]
            else:
                assert set(data_mapping.keys()).issubset(
                    {"train", "validation", "test"}
                ), (
                    "To select folds from a HuggingFace dataset, the folds must either be {'train', 'validation'}"
                    " or {'train'} "
                )
                for fold_key in data_mapping.keys():
                    if isinstance(data_mapping[fold_key], str):
                        self.foldname_mapping[fold_key] = data_mapping[fold_key]
                        try:
                            data_mapping_unified[fold_key] = data_raw[
                                data_mapping[fold_key]
                            ]
                        except KeyError:
                            raise KeyError(
                                "The key {} is not found in the dataset".format(
                                    data_mapping[fold_key]
                                )
                            )
                    else:
                        self.foldname_mapping[fold_key] = fold_key
                        data_mapping_unified[fold_key] = data_mapping[fold_key]
        else:
            assert set(dataset_config["data_files"].keys()).issubset(
                {"train", "validation", "test"}
            ), "If the dataset is custom, the data fold must be a subset of {'train', 'validation', 'test'} "
            data_mapping_unified = data_raw
        return data_mapping_unified

    def tokenize_text(self, data_dict):

        tokenized_data_dict = {}
        for each_key in data_dict:
            tokenized_data_dict[
                each_key
            ] = self.autoencodetext_from_model_and_dataset_name(
                data_dict[each_key],
            )
        return tokenized_data_dict

    def _prepare_data(
        self,
        dataset_config,
        data_mapping=None,
    ):
        from datasets import load_dataset
        from .utils import PathUtils

        if self.custom_hpo_args.is_wandb_on:
            from .result_analysis.wandb_utils import WandbUtils

            self.wandb_utils = WandbUtils(
                is_wandb_on=self.custom_hpo_args.is_wandb_on,
                wandb_key_path=self.custom_hpo_args.key_path,
                jobid_config=self.jobid_config,
            )
            self.wandb_utils.set_wandb_per_run()
        else:
            self.wandb_utils = None

        self.path_utils = PathUtils(
            self.jobid_config, hpo_output_dir=self.custom_hpo_args.output_dir
        )

        if isinstance(dataset_config, dict):
            data_raw = load_dataset(**dataset_config)
        else:
            data_raw = load_dataset(*dataset_config)

        data_mapping_unified = self.unify_data_raw_and_data_mapping(
            data_raw, dataset_config, data_mapping
        )

        tokenized_data_dict = self.tokenize_text(data_dict=data_mapping_unified)

        # decide whether to use holdout or cross validation based on the fold names. If the fold name is {"train"}
        # must use cross validation; if the fold name is {"train", "validation"}, must use holdout

        assert (
            set(tokenized_data_dict.keys()) == {"train"}
            or set(tokenized_data_dict.keys()) == {"train", "validation"}
            or set(tokenized_data_dict.keys()) == {"test"}
        ), (
            "The folds you specify can only be one of the following: "
            "(1) {'train'} (2) {'train', 'validation'} (3) {'test'}"
        )

        if set(tokenized_data_dict) == {"train"}:
            assert (
                self.custom_hpo_args.eval_method == "cv"
            ), "Because the validation fold is not specified, eval_method must be set to cv"
            print(
                "Because the validation fold is not specified, eval_method is set to cv, if you need to use the "
                "hold out mode, set eval_method to 'holdout'"
            )
            train_datasets, eval_datasets = self._split_for_cross_validation(
                tokenized_data_dict["train"]
            )
            self.train_datasets = train_datasets
            self.eval_datasets = eval_datasets
        elif set(tokenized_data_dict) == {"train", "validation"}:
            assert (
                self.custom_hpo_args.eval_method == "holdout"
            ), "Because the validation fold is specified, eval_method must be set to holdout"
            print(
                "Because the validation fold is specified, eval_method is set to holdout, if you need to use the "
                "cv mode, set eval_method to 'cv'"
            )
            self.train_dataset = tokenized_data_dict["train"]
            self.eval_dataset = tokenized_data_dict["validation"]
        else:
            self.test_dataset = tokenized_data_dict["test"]

    def _load_model(self, checkpoint_path=None, per_model_config=None):
        from .dataset.task_auto import get_default_task
        from transformers import AutoConfig
        from .huggingface.switch_head_auto import (
            AutoSeqClassificationHead,
            MODEL_CLASSIFICATION_HEAD_MAPPING,
        )

        this_task = get_default_task(self.jobid_config.dat, self.custom_hpo_args.task)
        if not checkpoint_path:
            checkpoint_path = self.jobid_config.pre_full

        def get_this_model():
            from transformers import AutoModelForSequenceClassification

            return AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path, config=model_config
            )

        def is_pretrained_model_in_classification_head_list():
            return self.jobid_config.pre in MODEL_CLASSIFICATION_HEAD_MAPPING

        def _set_model_config():
            if per_model_config and len(per_model_config) > 0:
                model_config = AutoConfig.from_pretrained(
                    checkpoint_path,
                    num_labels=model_config_num_labels,
                    **per_model_config,
                )
            else:
                model_config = AutoConfig.from_pretrained(
                    checkpoint_path, num_labels=model_config_num_labels
                )
            return model_config

        if this_task == "seq-classification":
            num_labels_old = AutoConfig.from_pretrained(checkpoint_path).num_labels
            if is_pretrained_model_in_classification_head_list():
                model_config_num_labels = num_labels_old
            else:
                self._set_num_labels()
                model_config_num_labels = self._num_labels
            model_config = _set_model_config()

            if is_pretrained_model_in_classification_head_list():
                if self._num_labels != num_labels_old:
                    this_model = get_this_model()
                    model_config.num_labels = self._num_labels
                    this_model.num_labels = self._num_labels
                    this_model.classifier = (
                        AutoSeqClassificationHead.from_model_type_and_config(
                            self.jobid_config.pre, model_config
                        )
                    )
                else:
                    this_model = get_this_model()
            else:
                this_model = get_this_model()
            this_model.resize_token_embeddings(self.vocab_size)
            return this_model
        elif this_task == "regression":
            model_config_num_labels = 1
            model_config = _set_model_config()
            this_model = get_this_model()
            return this_model

    def _get_metric_func(self):
        if self.jobid_config.dat[0] in ("glue", "super_glue"):
            metric = datasets.load.load_metric(self.jobid_config.dat)
        else:
            metric = datasets.load.load_metric(self.custom_hpo_args.metric_name)
        return metric

    def _compute_metrics_by_dataset_name(self, eval_pred):
        predictions, labels = eval_pred
        predictions = (
            np.squeeze(predictions)
            if self.custom_hpo_args.task == "regression"
            else np.argmax(predictions, axis=1)
        )
        metric_func = self._get_metric_func()
        return metric_func.compute(predictions=predictions, references=labels)

    def _compute_checkpoint_freq(self, num_train_epochs, batch_size):
        if "gpu" in self.custom_hpo_args.resources_per_trial:
            ckpt_step_freq = (
                int(
                    min(num_train_epochs, 1)
                    * len(self._get_first_train_fold())
                    / batch_size
                    / self.custom_hpo_args.resources_per_trial["gpu"]
                    / self.custom_hpo_args.ckpt_per_epoch
                )
                + 1
            )
        else:
            ckpt_step_freq = (
                int(
                    min(num_train_epochs, 1)
                    * len(self._get_first_train_fold())
                    / batch_size
                    / self.custom_hpo_args.resources_per_trial["cpu"]
                    / self.custom_hpo_args.ckpt_per_epoch
                )
                + 1
            )

        return ckpt_step_freq

    @staticmethod
    def _separate_config(config):

        training_args_config = {}
        per_model_config = {}

        for key, val in config.items():
            if key in TrainingArguments.__dict__:
                training_args_config[key] = val
            else:
                per_model_config[key] = val

        return training_args_config, per_model_config

    def _train_one_fold_routime(
        self, this_model, training_args, model_init, trial_id, tokenizer
    ):
        trainer = TrainerForAutoTransformers(
            model=this_model,
            args=training_args,
            model_init=model_init,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self._compute_metrics_by_dataset_name,
        )
        trainer.trial_id = trial_id

        """
            create a wandb run. If os.environ["WANDB_MODE"] == "offline", run = None
        """
        trainer.hp_search_backend = None
        trainer.train()
        output_metrics = trainer.evaluate(self.eval_dataset)
        return trainer, output_metrics

    def _objective(self, config, reporter=None, checkpoint_dir=None):
        from transformers.trainer_utils import set_seed
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.jobid_config.pre_full, use_fast=True
        )
        self.vocab_size = len(tokenizer)
        self._set_transformers_verbosity(self.custom_hpo_args.transformers_verbose)

        def model_init():
            return self._load_model()

        set_seed(config["seed"])

        training_args_config, per_model_config = AutoTransformers._separate_config(
            config
        )
        this_model = self._load_model(per_model_config=per_model_config)

        # If reporter != None, set trial_id to reporter.trial_id, i.e., when the trainable function is used for hpo
        trial_id = reporter.trial_id if reporter else "0000"

        self.path_utils.make_dir_per_trial(trial_id)

        ckpt_freq = self._compute_checkpoint_freq(
            num_train_epochs=config["num_train_epochs"],
            batch_size=config["per_device_train_batch_size"],
        )

        assert self.path_utils.ckpt_dir_per_trial is not None

        if transformers.__version__.startswith("3"):
            training_args = TrainingArguments(
                output_dir=self.path_utils.ckpt_dir_per_trial,
                do_train=True,
                do_eval=True,
                per_device_eval_batch_size=1,
                eval_steps=ckpt_freq,
                evaluate_during_training=True,
                save_steps=ckpt_freq,
                save_total_limit=0,
                fp16=self.custom_hpo_args.fp16,
                **training_args_config,
            )
        else:
            from transformers import IntervalStrategy

            training_args = TrainingArguments(
                output_dir=self.path_utils.ckpt_dir_per_trial,
                do_train=True,
                do_eval=True,
                per_device_eval_batch_size=1,
                eval_steps=ckpt_freq,
                evaluation_strategy=IntervalStrategy.STEPS,
                save_steps=ckpt_freq,
                save_total_limit=0,
                fp16=self.custom_hpo_args.fp16,
                **training_args_config,
            )

        if self.wandb_utils:
            run = self.wandb_utils.set_wandb_per_trial()
            import wandb

            for each_hp in config:
                wandb.log({each_hp: config[each_hp]})
        else:
            run = None

        if self.custom_hpo_args.eval_method == "cv":
            import mlflow

            all_output_metrics = []
            for cv_idx in range(self.custom_hpo_args.cv_k):
                self.train_dataset = self.train_datasets[cv_idx]
                self.eval_dataset = self.eval_datasets[cv_idx]
                trainer, output_metrics = self._train_one_fold_routime(
                    this_model, training_args, model_init, trial_id, tokenizer
                )
                mlflow.end_run()
                all_output_metrics.append(output_metrics)
            TrainerForAutoTransformers.tune_report(
                mode="cv", output_metrics=all_output_metrics
            )
        else:
            trainer, output_metrics = self._train_one_fold_routime(
                this_model, training_args, model_init, trial_id, tokenizer
            )
            TrainerForAutoTransformers.tune_report(
                mode="holdout", output_metrics=output_metrics
            )
        """
            If a wandb run was created, close the run after train and evaluate finish
        """
        if run:
            run.finish()

    def _verify_init_config(self, points_to_evaluate=None):
        if points_to_evaluate is not None:
            for each_init_config in points_to_evaluate:
                for each_hp in list(each_init_config.keys()):
                    if each_hp not in self._search_space_hpo:
                        del each_init_config[each_hp]
                        print(
                            each_hp,
                            "is not in the search space, deleting from init config",
                        )
                        continue
                    hp_value = each_init_config[each_hp]
                    domain = self._search_space_hpo[each_hp]

                    if isinstance(domain, ray.tune.sample.Categorical):
                        assert (
                            hp_value in domain.categories
                        ), f"points_to_evaluate {each_hp} value must be within the search space"
                    elif isinstance(domain, ray.tune.sample.Float) or isinstance(
                        domain, ray.tune.sample.Integer
                    ):
                        assert (
                            domain.lower <= hp_value <= domain.upper
                        ), f"points_to_evaluate {each_hp} value must be within the search space"

    def _get_search_algo(
        self,
        search_algo_name,
        search_algo_args_mode,
        time_budget,
        metric_name,
        metric_mode_name,
        points_to_evaluate=None,
    ):
        from .hpo.searchalgo_auto import AutoSearchAlgorithm

        if search_algo_name in ("bs", "cfo"):
            self._verify_init_config(points_to_evaluate)
        search_algo = AutoSearchAlgorithm.from_method_name(
            search_algo_name=search_algo_name,
            search_algo_args_mode=search_algo_args_mode,
            hpo_search_space=self._search_space_hpo,
            time_budget=time_budget,
            metric_name=metric_name,
            metric_mode_name=metric_mode_name,
            seed_bs=self.jobid_config.sdbs,
            custom_hpo_args=self.custom_hpo_args,
        )
        return search_algo

    @staticmethod
    def _recover_checkpoint(tune_checkpoint_dir):
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

    def _save_ckpt_json(self, best_ckpt):
        with open(
            os.path.join(
                self.path_utils.result_dir_per_run,
                "save_ckpt_" + self.jobid_config.to_jobid_string() + ".json",
            ),
            "w",
        ) as fout:
            json.dump(
                {"best_ckpt": best_ckpt},
                fout,
            )

    def _load_ckpt_json(self, ckpt_dir=None):
        if not ckpt_dir:
            ckpt_dir = os.path.join(
                self.path_utils.result_dir_per_run,
                "save_ckpt_" + self.jobid_config.to_jobid_string() + ".json",
            )
        try:
            with open(ckpt_dir) as fin:
                ckpt_json = json.load(fin)
                return ckpt_json["best_ckpt"]
        except FileNotFoundError as err:
            print(
                "Saved checkpoint not found. Please make sure checkpoint is stored under",
                ckpt_dir,
            )
            raise err

    def _set_metric(self, is_custom_data, custom_hpo_args):
        from .dataset.metric_auto import get_default_and_alternative_metric
        from .utils import _variable_override_default_alternative

        if is_custom_data:
            assert "metric_name" in custom_hpo_args, (
                "If the dataset is customized, "
                "you must provide task in flaml.nlp.utils:HPOArgs"
            )
            assert "metric_mode_name" in custom_hpo_args, (
                "If the dataset is customized, "
                "you must provide task in flaml.nlp.utils:HPOArgs"
            )

        custom_metric_name = (
            custom_hpo_args["metric_name"] if "metric_name" in custom_hpo_args else None
        )
        custom_metric_mode_name = (
            custom_hpo_args["metric_mode_name"]
            if "metric_mode_name" in custom_hpo_args
            else None
        )

        # Step 1: get the default metric and all the metrics
        (
            default_metric,
            default_mode,
            all_metrics,
            all_modes,
        ) = get_default_and_alternative_metric(
            dataset_name_list=self.jobid_config.dat,
            custom_metric_name=custom_metric_name,
            custom_metric_mode_name=custom_metric_mode_name,
        )

        # Step 2: override the default metric and mode with the custom metric and mode, if
        # there exists one
        _variable_override_default_alternative(
            obj_ref=self,
            var_name="metric_name",
            default_value=default_metric,
            all_values=all_metrics,
            overriding_value=custom_metric_name,
        )
        _variable_override_default_alternative(
            obj_ref=self,
            var_name="metric_mode_name",
            default_value=default_mode,
            all_values=all_modes,
            overriding_value=custom_metric_mode_name,
        )
        self._all_metrics = all_metrics
        self._all_modes = all_modes

    def _set_task(self, is_custom_data, custom_hpo_args):
        from .dataset.task_auto import get_default_task

        if is_custom_data:
            assert "task" in custom_hpo_args, (
                "If the dataset is customized, "
                "you must provide task in flaml.nlp.utils:HPOArgs"
            )

        setattr(
            self.custom_hpo_args,
            "task",
            get_default_task(
                self.jobid_config.dat,
                custom_hpo_args["task"] if "task" in custom_hpo_args else None,
            ),
        )

    def _get_first_train_fold(self):
        if self.custom_hpo_args.eval_method == "cv":
            return self.train_datasets[0]
        else:
            return self.train_dataset

    def _set_num_labels(self):
        if self.custom_hpo_args.task == "regression":
            self._num_labels = 1
            return
        elif self.custom_hpo_args.task == "seq-classification":
            first_train_fold = self._get_first_train_fold()
            assert self.custom_hpo_args.label_name in first_train_fold.features, (
                "The label column {} is not specified in the dataset, you need to specify the label column with "
                "flaml.nlp.utils:HPOArgs::label_name"
            )
            try:
                self._num_labels = len(
                    first_train_fold.features[self.custom_hpo_args.label_name].names
                )
            except AttributeError:
                self._num_labels = len(
                    set([x[self.custom_hpo_args.label_name] for x in first_train_fold])
                )

    def _set_transformers_verbosity(self, transformers_verbose):
        import transformers

        if transformers_verbose == transformers.logging.ERROR:
            transformers.logging.set_verbosity_error()
        elif transformers_verbose == transformers.logging.WARNING:
            transformers.logging.set_verbosity_warning()
        elif transformers_verbose == transformers.logging.INFO:
            transformers.logging.set_verbosity_info()
        elif transformers_verbose == transformers.logging.DEBUG:
            transformers.logging.set_verbosity_debug()
        else:
            raise ValueError(
                "transformers_verbose must be set to ERROR, WARNING, INFO or DEBUG"
            )

    @staticmethod
    def get_best_trial_with_checkpoint(
        analysis, metric=None, mode=None, scope="last", filter_nan_and_inf=True
    ):
        """Retrieve the best trial object.

        Compares all trials' scores on ``metric``.
        If ``metric`` is not specified, ``self.default_metric`` will be used.
        If `mode` is not specified, ``self.default_mode`` will be used.
        These values are usually initialized by passing the ``metric`` and
        ``mode`` parameters to ``tune.run()``.

        Args:
            metric (str): Key for trial info to order on. Defaults to
                ``self.default_metric``.
            mode (str): One of [min, max]. Defaults to ``self.default_mode``.
            scope (str): One of [all, last, avg, last-5-avg, last-10-avg].
                If `scope=last`, only look at each trial's final step for
                `metric`, and compare across trials based on `mode=[min,max]`.
                If `scope=avg`, consider the simple average over all steps
                for `metric` and compare across trials based on
                `mode=[min,max]`. If `scope=last-5-avg` or `scope=last-10-avg`,
                consider the simple average over the last 5 or 10 steps for
                `metric` and compare across trials based on `mode=[min,max]`.
                If `scope=all`, find each trial's min/max score for `metric`
                based on `mode`, and compare trials based on `mode=[min,max]`.
            filter_nan_and_inf (bool): If True (default), NaN or infinite
                values are disregarded and these trials are never selected as
                the best trial.
        """
        from ray.tune.utils.util import is_nan_or_inf

        if scope not in ["all", "last", "avg", "last-5-avg", "last-10-avg"]:
            raise ValueError(
                "ExperimentAnalysis: attempting to get best trial for "
                'metric {} for scope {} not in ["all", "last", "avg", '
                '"last-5-avg", "last-10-avg"]. '
                '"'.format(metric, scope)
            )
        best_trial = None
        best_metric_score = None
        for trial in analysis.trials:
            best_ckpt = analysis.get_best_checkpoint(trial, metric=metric, mode=mode)
            if best_ckpt is None:
                print(
                    "No checkpoints have been found for trial {}, continuing".format(
                        trial
                    )
                )
                continue

            if scope in ["last", "avg", "last-5-avg", "last-10-avg"]:
                metric_score = trial.metric_analysis[metric][scope]
            else:
                metric_score = trial.metric_analysis[metric][mode]

            if filter_nan_and_inf and is_nan_or_inf(metric_score):
                continue

            if best_metric_score is None:
                best_metric_score = metric_score
                best_trial = trial
                continue

            if (mode == "max") and (best_metric_score < metric_score):
                best_metric_score = metric_score
                best_trial = trial
            elif (mode == "min") and (best_metric_score > metric_score):
                best_metric_score = metric_score
                best_trial = trial

        return best_trial

    def _check_input_args(self, dataset_config):
        from .utils import check_custom_data_format

        self.custom_hpo_args.check_grid_config()
        check_custom_data_format(
            dataset_config, self.custom_hpo_args.custom_sentence_keys
        )

    def fit(
        self,
        dataset_config: Dict,
        data_mapping: Union[Dict, List] = None,
        load_config_mode="args",
        **custom_hpo_args,
    ):
        """Fine tuning the huggingface using the hpo setting

        Example:

            .. code-block:: python

                autohf = AutoTransformers()

                autohf_settings = {
                    "dataset_config": ["glue", "mrpc"],
                    "pretrained_model": "google/electra-small-discriminator",
                    "output_dir": "data/",
                    "resources_per_trial": {"cpu": 1, "gpu": 1},
                    "num_samples": -1,
                    "time_budget": 300,
                }

                validation_metric, analysis = autohf.fit(**autohf_settings)

        Args:
            dataset_config:
                dataset config, which is either a dict or a list, the dict or list must be consistent with the
                argument in datasets.load_dataset. See the documentation for HPOArgs::dataset_config
            custom_sentence_keys:
                the custom sentence key, must be set to a tuple of str if dataset_config["path"] is "csv" or "json"
                (i.e., custom data files)
            load_config_mode:
                A string, the mode for loading args. "args" if setting the args from argument, "console"
                if setting the args from console
            **custom_hpo_args:
                The custom arguments, please find all candidate arguments from ``flaml.nlp.utils::HPOArgs``

        Returns:

            validation_metric: A dict storing the validation score

            analysis: A ray.tune.analysis.Analysis object storing the analysis results from tune.run
        """
        from .utils import HPOArgs

        hpo_args = HPOArgs()

        assert self._is_huggingface_dataset(
            dataset_config, data_mapping
        ) ^ self._is_custom_dataset(dataset_config), (
            "The specified data must be in one of the two modes: (1) custom data, by specifying data_files "
            "in dataset_config; (2) HuggingFace data, by specifying the path argument in dataset_config and "
            "which fold(s) from the dataset you will be loading with the data_mapping argument"
        )

        self.custom_hpo_args = hpo_args.load_args(load_config_mode, **custom_hpo_args)
        self._check_input_args(dataset_config)

        self.jobid_config = JobID()
        self.jobid_config.set_jobid_from_console_args(
            dataset_config=dataset_config, console_args=self.custom_hpo_args
        )

        self._prepare_data(dataset_config=dataset_config, data_mapping=data_mapping)

        from .hpo.scheduler_auto import AutoScheduler

        assert self.jobid_config is not None

        is_custom_data = self._is_custom_dataset(dataset_config)

        self._set_metric(is_custom_data=is_custom_data, custom_hpo_args=custom_hpo_args)
        self._set_task(is_custom_data=is_custom_data, custom_hpo_args=custom_hpo_args)
        self._set_num_labels()
        self._set_search_space()

        search_algo = self._get_search_algo(
            search_algo_name=self.jobid_config.alg,
            search_algo_args_mode=self.jobid_config.arg,
            time_budget=self.custom_hpo_args.time_budget,
            metric_name=self.custom_hpo_args.metric_name,
            metric_mode_name=self.custom_hpo_args.metric_mode_name,
            points_to_evaluate=self.custom_hpo_args.points_to_evaluate,
        )
        if self.jobid_config.alg == "bs":
            search_algo.set_search_properties(
                config=self._search_space_hpo,
                metric=self.custom_hpo_args.metric_name,
                mode=self.custom_hpo_args.metric_mode_name,
            )
            search_algo.set_search_properties(
                config={"time_budget_s": self.custom_hpo_args.time_budget}
            )

        scheduler = AutoScheduler.from_scheduler_name(self.jobid_config.pru)
        self.path_utils.make_dir_per_run()

        assert self.path_utils.ckpt_dir_per_run
        start_time = time.time()

        tune_config = self._search_space_hpo
        # if the search space does not contain the seed for huggingface,
        # set the seed to the default value in self.jobid_config
        if "seed" not in tune_config:
            tune_config["seed"] = self.jobid_config.sdhf

        if self.jobid_config.alg in ("bs", "rs", "cfo"):
            import numpy as np

            np.random.seed(self.jobid_config.sdbs + 7654321)

        self.jobid_config.check_model_type_consistency()
        analysis = ray.tune.run(
            self._objective,
            metric=self.custom_hpo_args.metric_name,
            mode=self.custom_hpo_args.metric_mode_name,
            name="ray_result",
            resources_per_trial=self.custom_hpo_args.resources_per_trial,
            config=tune_config,
            verbose=self.custom_hpo_args.ray_verbose,
            local_dir=self.path_utils.ckpt_dir_per_run,
            num_samples=self.custom_hpo_args.sample_num,
            time_budget_s=self.custom_hpo_args.time_budget,
            keep_checkpoints_num=self.custom_hpo_args.keep_checkpoints_num,
            checkpoint_score_attr=self.custom_hpo_args.metric_name,
            scheduler=scheduler,
            search_alg=search_algo,
            fail_fast=True,
        )
        duration = time.time() - start_time
        self.last_run_duration = duration
        print("Total running time: {} seconds".format(duration))

        ray.shutdown()

        best_trial = AutoTransformers.get_best_trial_with_checkpoint(
            analysis,
            scope="all",
            metric=self.custom_hpo_args.metric_name,
            mode=self.custom_hpo_args.metric_mode_name,
        )
        if best_trial is not None:
            validation_metric = {
                "eval_"
                + self.custom_hpo_args.metric_name: best_trial.metric_analysis[
                    self.custom_hpo_args.metric_name
                ][self.custom_hpo_args.metric_mode_name]
            }
            for i, metric in enumerate(self._all_metrics):
                validation_metric["eval_" + metric] = best_trial.metric_analysis[
                    metric
                ][self._all_modes[i]]

            get_best_ckpt = analysis.get_best_checkpoint(
                best_trial,
                metric=self.custom_hpo_args.metric_name,
                mode=self.custom_hpo_args.metric_mode_name,
            )
            best_ckpt = AutoTransformers._recover_checkpoint(get_best_ckpt)

            self._save_ckpt_json(best_ckpt)

            return analysis
        else:
            return analysis

    def predict(
        self,
        dataset_config=None,
        data_mapping: Union[Dict, List] = None,
        input_text: Union[Dict, List, List[Dict], List[List]] = None,
        ckpt_json_dir=None,
    ):
        """Predict label for test data.

        An example:
            predictions, test_metric = autohf.predict()

        Args:
            ckpt_json_dir:
                the checkpoint for the fine-tuned huggingface if you wish to override
                the saved checkpoint in the training stage under self.path_utils._result_dir_per_run

        Returns:
            A numpy array of shape n * 1 - - each element is a predicted class
            label for an instance.
        """
        from .huggingface.trainer import TrainerForAutoTransformers
        from .dataset.dataprocess_auto import AutoEncodeText

        assert (input_text is not None) ^ (
            self._is_huggingface_dataset(dataset_config, data_mapping)
            or self._is_custom_dataset(dataset_config)
        ), (
            "The specified data must be in one of the three modes: (1) custom data, by specifying data_files "
            "in dataset_config; (2) HuggingFace data, by specifying the path argument in dataset_config and "
            "which fold(s) from the dataset you will be loading with the data_mapping argument, and (3) input "
            "text mode, which is either a sentence or a list of sentences"
        )

        assert self.jobid_config is not None, (
            "jobid_config is not set in the AutoTransformers object, "
            "did you forget to run AutoTransformers.fit before AutoTransformers.predict?"
        )

        if input_text is not None:
            if (isinstance(input_text, List) and isinstance(input_text[0], List)) or (
                isinstance(input_text, List) and isinstance(input_text[0], Dict)
            ):
                self.test_dataset = AutoEncodeText.from_model_and_example_list(
                    example_list=input_text,
                    dataset_name_list=self.jobid_config.dat,
                    custom_sentence_keys=self.custom_hpo_args.custom_sentence_keys,
                )
            else:
                self.test_dataset = AutoEncodeText.from_model_and_example(
                    this_example=input_text,
                    dataset_name_list=self.jobid_config.dat,
                    custom_sentence_keys=self.custom_hpo_args.custom_sentence_keys,
                )
        else:
            self._prepare_data(dataset_config=dataset_config, data_mapping=data_mapping)

        best_checkpoint = self._load_ckpt_json(ckpt_json_dir)
        best_model = self._load_model(checkpoint_path=best_checkpoint)
        training_args = TrainingArguments(
            per_device_eval_batch_size=1, output_dir=self.path_utils.result_dir_per_run
        )
        test_trainer = TrainerForAutoTransformers(best_model, training_args)

        if self.jobid_config.spt == "ori":
            if "label" in self.test_dataset.features:
                self.test_dataset.remove_columns_(["label"])
                print("Cleaning the existing label column from test data")

        test_dataloader = test_trainer.get_test_dataloader(self.test_dataset)
        predictions, labels, _ = test_trainer.prediction_loop(
            test_dataloader, description="Prediction"
        )
        from .dataset.task_auto import get_default_task

        predictions = (
            np.squeeze(predictions)
            if self.custom_hpo_args.task == "regression"
            else np.argmax(predictions, axis=1)
        )
        # TODO: add another function for post processing
        return predictions

    def output_prediction(
        self, predictions=None, output_prediction_path=None, output_zip_file_name=None
    ):
        """
        When using the original GLUE split, output the prediction on test data,
        and prepare the .zip file for submission

        Example:
            local_archive_path = self.autohf.output_prediction(predictions,
                                  output_prediction_path= self.console_args.output_dir + "result/",
                                  output_zip_file_name=azure_save_file_name)

        Args:
            predictions:
                A list of predictions, which is the output of AutoTransformers.predict()
            output_prediction_path:
                Output path for the prediction
            output_zip_file_name:
                An string, which is the name of the output zip file

        Returns:
            The path of the output .zip file
        """
        from .dataset.submission_auto import auto_output_prediction

        return auto_output_prediction(
            self.jobid_config.dat,
            output_prediction_path,
            output_zip_file_name,
            predictions,
            self.train_dataset,
            self.foldname_mapping["validation"],
        )
