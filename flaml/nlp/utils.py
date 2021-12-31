import argparse
from dataclasses import dataclass, field
from typing import Dict, Any
from ..data import SUMMARIZATION, SEQREGRESSION, SEQCLASSIFICATION, MAKE_TOKENS, NLG_TASKS


def load_default_huggingface_metric_for_task(task):
    if task == SEQCLASSIFICATION:
        return "accuracy", "max"
    elif task == SEQREGRESSION:
        return "rmse", "max"
    elif task == SUMMARIZATION:
        return "rouge", "max"
    elif task == MAKE_TOKENS:
        return "seqeval", "max"
    # TODO: elif task == your task, return the default metric name for your task,
    #  e.g., if task == MULTIPLECHOICE, return "accuracy"
    #  notice this metric name has to be in ['accuracy', 'bertscore', 'bleu', 'bleurt',
    #  'cer', 'chrf', 'code_eval', 'comet', 'competition_math', 'coval', 'cuad',
    #  'f1', 'gleu', 'glue', 'google_bleu', 'indic_glue', 'matthews_correlation',
    #  'meteor', 'pearsonr', 'precision', 'recall', 'rouge', 'sacrebleu', 'sari',
    #  'seqeval', 'spearmanr', 'squad', 'squad_v2', 'super_glue', 'ter', 'wer',
    #  'wiki_split', 'xnli']


global tokenized_column_names


def tokenize_text(X, Y=None, task=None, custom_hpo_args=None):
    if task in (SEQCLASSIFICATION, SEQREGRESSION):
        X_tokenized, _ = tokenize_onedataframe(
            X, this_tokenizer=None, task=task, custom_hpo_args=custom_hpo_args
        )
        return X_tokenized, None
    elif task == MAKE_TOKENS:
        return tokenize_text_tokclassification(X, Y, custom_hpo_args)
    elif task in NLG_TASKS:
        return tokenize_seq2seq(X, Y, task=task, custom_hpo_args=custom_hpo_args)

def tokenize_and_align_labels(examples, tokenizer, custom_hpo_args, X_sent_key, Y_sent_key=None):
    global tokenized_column_names

    padding = "max_length" if custom_hpo_args.pad_to_max_length else False
    tokenized_inputs = tokenizer(
        [list(examples[X_sent_key])],
        padding="max_length",
        truncation=True,
        max_length=custom_hpo_args.max_seq_length,
        is_split_into_words=True,
    )
    if Y_sent_key is not None:
        previous_word_idx = None
        label_ids = []
        import numbers
        for word_idx in tokenized_inputs.word_ids(batch_index=0):
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                if isinstance(examples[Y_sent_key][word_idx], numbers.Number):
                    label_ids.append(examples[Y_sent_key][word_idx])
                # else:
                #     label_ids.append(label_to_id[label[word_idx]])
            else:
                if isinstance(examples[Y_sent_key][word_idx], numbers.Number):
                    label_ids.append(examples[Y_sent_key][word_idx])
                # else:
                #     label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
            previous_word_idx = word_idx
        tokenized_inputs["label"] = label_ids
    tokenized_column_names = sorted(tokenized_inputs.keys())
    tokenized_input_and_labels = [tokenized_inputs[x] for x in tokenized_column_names]
    for key_idx, each_key in enumerate(tokenized_column_names):
        if each_key != "label":
            tokenized_input_and_labels[key_idx] = tokenized_input_and_labels[key_idx][0]
    return tokenized_input_and_labels


def tokenize_text_tokclassification(X, Y, custom_hpo_args):
    from transformers import AutoTokenizer
    import pandas as pd

    global tokenized_column_names
    this_tokenizer = AutoTokenizer.from_pretrained(
        custom_hpo_args.model_path, use_fast=True
    )
    if Y is not None:
        X_and_Y = pd.concat([X, Y.to_frame()], axis=1)
        X_key = list(X.keys())[0]
        Y_key = list(Y.to_frame().keys())[0]
        X_and_Y_tokenized = X_and_Y.apply(
            lambda x: tokenize_and_align_labels(x, tokenizer=this_tokenizer, custom_hpo_args=custom_hpo_args, X_sent_key=X_key, Y_sent_key=Y_key),
            axis=1,
            result_type="expand",
        )
        label_idx = tokenized_column_names.index("label")
        other_indices = sorted(set(range(len(tokenized_column_names))).difference({label_idx}))
        other_column_names = [tokenized_column_names[x] for x in other_indices]
        d = X_and_Y_tokenized.iloc[:, other_indices]
        y_tokenized = X_and_Y_tokenized.iloc[:, label_idx]
    else:
        X_key = list(X.keys())[0]
        d = X.apply(
            lambda x: tokenize_and_align_labels(x, tokenizer=this_tokenizer, custom_hpo_args=custom_hpo_args, X_sent_key=X_key, Y_sent_key=None),
            axis=1,
            result_type="expand",)
        other_column_names = tokenized_column_names
        y_tokenized = None
    X_tokenized = pd.DataFrame(columns=other_column_names)
    X_tokenized[other_column_names] = d
    return X_tokenized, y_tokenized

def tokenize_seq2seq(X, Y, task=None, custom_hpo_args=None):
    model_inputs, tokenizer = tokenize_onedataframe(
        X,
        this_tokenizer=None,
        task=task,
        custom_hpo_args=custom_hpo_args,
    )
    labels = None
    if Y is not None:
        labels, _ = tokenize_onedataframe(
            Y.to_frame(),
            this_tokenizer=tokenizer,
            task=task,
            custom_hpo_args=custom_hpo_args,
        )
        labels["label"] = [
            [(each_l if each_l != tokenizer.pad_token_id else -100) for each_l in label]
            for label in labels["input_ids"]
        ]
        labels = labels.drop(
            columns=["attention_mask", "input_ids", "decoder_input_ids"]
        )
    return model_inputs, labels


def tokenize_onedataframe(
    X,
    this_tokenizer=None,
    task=None,
    custom_hpo_args=None,
):
    from transformers import AutoTokenizer
    import pandas

    global tokenized_column_names

    if this_tokenizer:
        with this_tokenizer.as_target_tokenizer():
            d = X.apply(
                lambda x: tokenize_row(
                    x,
                    this_tokenizer,
                    prefix=("",) if task is SUMMARIZATION else None,
                    task=task,
                    custom_hpo_args=custom_hpo_args,
                ),
                axis=1,
                result_type="expand",
            )
    else:
        this_tokenizer = AutoTokenizer.from_pretrained(
            custom_hpo_args.model_path, use_fast=True
        )
        d = X.apply(
            lambda x: tokenize_row(
                x,
                this_tokenizer,
                prefix=("summarize: ",) if task is SUMMARIZATION else None,
                task=task,
                custom_hpo_args=custom_hpo_args,
            ),
            axis=1,
            result_type="expand",
        )
    X_tokenized = pandas.DataFrame(columns=tokenized_column_names)
    X_tokenized[tokenized_column_names] = d
    return X_tokenized, this_tokenizer


def postprocess_text(preds, labels):
    import nltk

    nltk.download("punkt")
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def tokenize_row(
    this_row, this_tokenizer, prefix=None, task=None, custom_hpo_args=None
):
    global tokenized_column_names
    assert (
        "max_seq_length" in custom_hpo_args.__dict__
    ), "max_seq_length must be provided for glue"

    if prefix:
        this_row = tuple(["".join(x) for x in zip(prefix, this_row)])

    tokenized_example = this_tokenizer(
        *tuple(this_row),
        padding="max_length",
        max_length=custom_hpo_args.max_seq_length,
        truncation=True,
    )
    if task in NLG_TASKS:
        tokenized_example["decoder_input_ids"] = tokenized_example["input_ids"]
    tokenized_column_names = sorted(tokenized_example.keys())
    return [tokenized_example[x] for x in tokenized_column_names]


def separate_config(config, task):
    if task in NLG_TASKS:
        from transformers import Seq2SeqTrainingArguments, TrainingArguments

        trainargs_class_list = [Seq2SeqTrainingArguments, TrainingArguments]
    else:
        from transformers import TrainingArguments

        trainargs_class_list = [TrainingArguments]

    training_args_config = {}
    per_model_config = {}

    for key, val in config.items():
        is_in_training_args = any(key in x.__dict__ for x in trainargs_class_list)
        if is_in_training_args:
            training_args_config[key] = val
        else:
            per_model_config[key] = val

    return training_args_config, per_model_config


def get_num_labels(task, y_train):
    from ..data import SEQCLASSIFICATION, SEQREGRESSION, MAKE_TOKENS

    if task == SEQREGRESSION:
        return 1
    elif task == SEQCLASSIFICATION:
        return len(set(y_train))
    elif task == MAKE_TOKENS:
        return len(set([a for b in y_train.tolist() for a in b]))
    else:
        return None


def _clean_value(value: Any) -> str:
    if isinstance(value, float):
        return "{:.5}".format(value)
    else:
        return str(value).replace("/", "_")


def format_vars(resolved_vars: Dict) -> str:
    """Formats the resolved variable dict into a single string."""
    out = []
    for path, value in sorted(resolved_vars.items()):
        if path[0] in ["run", "env", "resources_per_trial"]:
            continue  # TrialRunner already has these in the experiment_tag
        pieces = []
        last_string = True
        for k in path[::-1]:
            if isinstance(k, int):
                pieces.append(str(k))
            elif last_string:
                last_string = False
                pieces.append(k)
        pieces.reverse()
        out.append(_clean_value("_".join(pieces)) + "=" + _clean_value(value))
    return ",".join(out)


counter = 0


def date_str():
    from datetime import datetime

    return datetime.today().strftime("%Y-%m-%d_%H-%M-%S")


def _generate_dirname(experiment_tag, trial_id):
    generated_dirname = f"train_{str(trial_id)}_{experiment_tag}"
    generated_dirname = generated_dirname[:130]
    generated_dirname += f"_{date_str()}"
    return generated_dirname.replace("/", "_")


def get_logdir_name(dirname, local_dir):
    import os

    local_dir = os.path.expanduser(local_dir)
    logdir = os.path.join(local_dir, dirname)
    return logdir


def get_trial_fold_name(local_dir, trial_config, trial_id):
    global counter
    counter = counter + 1
    experiment_tag = "{0}_{1}".format(str(counter), format_vars(trial_config))
    logdir = get_logdir_name(
        _generate_dirname(experiment_tag, trial_id=trial_id), local_dir
    )
    return logdir


def load_model(checkpoint_path, task, num_labels, per_model_config=None):
    from transformers import AutoConfig
    from .huggingface.switch_head_auto import (
        AutoSeqClassificationHead,
        MODEL_CLASSIFICATION_HEAD_MAPPING,
    )
    from ..data import SEQCLASSIFICATION, SEQREGRESSION, MAKE_TOKENS

    this_model_type = AutoConfig.from_pretrained(checkpoint_path).model_type
    this_vocab_size = AutoConfig.from_pretrained(checkpoint_path).vocab_size

    def get_this_model(task):
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoModelForSeq2SeqLM , AutoModelForTokenClassification

        if task in (SEQCLASSIFICATION, SEQREGRESSION):
            return AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path, config=model_config
            )
        elif task in NLG_TASKS:
            return AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_path, config=model_config
            )
        elif task == MAKE_TOKENS:
            return AutoModelForTokenClassification.from_pretrained(
                checkpoint_path, config=model_config
            )

    def is_pretrained_model_in_classification_head_list(model_type):
        return model_type in MODEL_CLASSIFICATION_HEAD_MAPPING

    def _set_model_config(checkpoint_path):
        if task in (SEQCLASSIFICATION, SEQREGRESSION, MAKE_TOKENS):
            if per_model_config:
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
        else:
            if per_model_config and len(per_model_config) > 0:
                model_config = AutoConfig.from_pretrained(
                    checkpoint_path,
                    **per_model_config,
                )
            else:
                model_config = AutoConfig.from_pretrained(checkpoint_path)
            return model_config

    if task == SEQCLASSIFICATION:
        num_labels_old = AutoConfig.from_pretrained(checkpoint_path).num_labels
        if is_pretrained_model_in_classification_head_list(this_model_type):
            model_config_num_labels = num_labels_old
        else:
            model_config_num_labels = num_labels
        model_config = _set_model_config(checkpoint_path)

        if is_pretrained_model_in_classification_head_list(this_model_type):
            if num_labels != num_labels_old:
                this_model = get_this_model(task)
                model_config.num_labels = num_labels
                this_model.num_labels = num_labels
                this_model.classifier = (
                    AutoSeqClassificationHead.from_model_type_and_config(
                        this_model_type, model_config
                    )
                )
            else:
                this_model = get_this_model(task)
        else:
            this_model = get_this_model(task)
        this_model.resize_token_embeddings(this_vocab_size)
        return this_model
    else:
        if task == SEQREGRESSION:
            model_config_num_labels = 1
        elif task == MAKE_TOKENS:
            model_config_num_labels = num_labels
        model_config = _set_model_config(checkpoint_path)
        this_model = get_this_model(task)
        return this_model


def compute_checkpoint_freq(
    train_data_size,
    custom_hpo_args,
    num_train_epochs,
    batch_size,
):
    ckpt_step_freq = (
        int(
            min(num_train_epochs, 1)
            * train_data_size
            / batch_size
            / custom_hpo_args.ckpt_per_epoch
        )
        + 1
    )
    return ckpt_step_freq


@dataclass
class HPOArgs:
    """The HPO setting.

    Args:
        output_dir (str): data root directory for outputing the log, etc.
        model_path (str, optional, defaults to "facebook/muppet-roberta-base"): A string,
            the path of the language model file, either a path from huggingface
            model card huggingface.co/models, or a local path for the model.
        fp16 (bool, optional, defaults to "False"): A bool, whether to use FP16.
        max_seq_length (int, optional, defaults to 128): An integer, the max length of the sequence.
        ckpt_per_epoch (int, optional, defaults to 1): An integer, the number of checkpoints per epoch.

    """

    output_dir: str = field(
        default="data/output/", metadata={"help": "data dir", "required": True}
    )

    model_path: str = field(
        default="facebook/muppet-roberta-base",
        metadata={"help": "model path model for HPO"},
    )

    fp16: bool = field(default=True, metadata={"help": "whether to use the FP16 mode"})

    max_seq_length: int = field(default=128, metadata={"help": "max seq length"})

    ckpt_per_epoch: int = field(default=1, metadata={"help": "checkpoint per epoch"})

    @staticmethod
    def load_args():
        from dataclasses import fields

        arg_parser = argparse.ArgumentParser()
        for each_field in fields(HPOArgs):
            print(each_field)
            arg_parser.add_argument(
                "--" + each_field.name,
                type=each_field.type,
                help=each_field.metadata["help"],
                required=each_field.metadata["required"]
                if "required" in each_field.metadata
                else False,
                choices=each_field.metadata["choices"]
                if "choices" in each_field.metadata
                else None,
                default=each_field.default,
            )
        console_args, unknown = arg_parser.parse_known_args()
        return console_args
