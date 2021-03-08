import datasets
from datasets import (
    load_dataset,
)
from typing import Callable, Dict
from transformers import EvalPrediction
from transformers import glue_compute_metrics, glue_output_modes
import numpy as np

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

def tokenize(self,
             examples):
    sentence1_key, sentence2_key = task_to_keys[self._task_name]
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key])
    )
    return self.tokenizer(*args, padding="max length", max_length=self.search_space_grid["max_seq_length"][0],
                          truncation=True)

def prepare_data(submit_mode,
                 task_name,
                 split_portion = None):
    dev_name = "validation" if task_name != "mnli" else "validation_matched"
    test_name = "test" if task_name != "mnli" else "test_matched"

    data_raw = load_dataset("glue", task_name)
    data_encoded = data_raw.map(tokenize, batched=True)

    assert submit_mode in ("resplit", "origin"), "submit_mode must be resplit or origin"

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

    return train_dataset, eval_dataset, test_dataset

def build_compute_metrics_fn(
        task_name: str) -> Callable[[EvalPrediction], Dict]:
    """Function from transformers/examples/text-classification/run_glue.py"""
    try:
        glue_output_modes["stsb"] = glue_output_modes["sts-b"]
        del glue_output_modes["sts-b"]
    except KeyError:
        pass
    output_mode = glue_output_modes[task_name]

    def compute_metrics_fn(p: EvalPrediction):
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        if task_name != "stsb":
            metrics = glue_compute_metrics(task_name, preds, p.label_ids)
        else:
            metrics = glue_compute_metrics("sts-b", preds, p.label_ids)
        return metrics

    return compute_metrics_fn

