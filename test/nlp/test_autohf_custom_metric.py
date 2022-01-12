import sys
import pytest


def custom_metric(
    X_test,
    y_test,
    estimator,
    labels,
    X_train,
    y_train,
    weight_test=None,
    weight_train=None,
    config=None,
    groups_test=None,
    groups_train=None,
):
    from datasets import Dataset
    from flaml.model import TransformersEstimator
    from flaml.nlp.utils import load_default_huggingface_metric_for_task

    if estimator._trainer is None:
        estimator._init_model_for_predict(X_test)
        trainer = estimator._trainer
        estimator._trainer = None
    else:
        trainer = estimator._trainer
    if y_test is not None:
        X_test, _ = estimator._preprocess(X_test)
        eval_dataset = Dataset.from_pandas(TransformersEstimator._join(X_test, y_test))
    else:
        X_test, _ = estimator._preprocess(X_test)
        eval_dataset = Dataset.from_pandas(X_test)

    trainer_compute_metrics_cache = trainer.compute_metrics
    trainer.compute_metrics = None

    metrics = trainer.evaluate(eval_dataset)
    trainer.compute_metrics = trainer_compute_metrics_cache

    auto_metric = load_default_huggingface_metric_for_task(estimator._task)

    estimator_metric_cache = estimator._metric
    estimator._metric = auto_metric

    eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
    from transformers.trainer_pt_utils import (
        nested_concat,
        nested_numpify,
        nested_truncate,
    )

    preds_host = None
    labels_host = None
    all_preds = None
    all_labels = None

    for step, inputs in enumerate(eval_dataloader):
        loss, logits, labels = trainer.prediction_step(
            trainer.model, inputs, prediction_loss_only=None, ignore_keys=None
        )

        if logits is not None:
            logits = trainer._pad_across_processes(logits)
            logits = trainer._nested_gather(logits)
            preds_host = (
                logits
                if preds_host is None
                else nested_concat(preds_host, logits, padding_index=-100)
            )
        if labels is not None:
            labels = trainer._pad_across_processes(labels)
            labels = trainer._nested_gather(labels)
            labels_host = (
                labels
                if labels_host is None
                else nested_concat(labels_host, labels, padding_index=-100)
            )

        # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
        num_samples = len(eval_dataset)

        if (
            trainer.args.eval_accumulation_steps is not None
            and (step + 1) % trainer.args.eval_accumulation_steps == 0
        ):
            if preds_host is not None:
                logits = nested_numpify(preds_host)
                all_preds = (
                    logits
                    if all_preds is None
                    else nested_concat(all_preds, logits, padding_index=-100)
                )
            if labels_host is not None:
                labels = nested_numpify(labels_host)
                all_labels = (
                    labels
                    if all_labels is None
                    else nested_concat(all_labels, labels, padding_index=-100)
                )
            preds_host, labels_host = None, None

        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = (
                logits
                if all_preds is None
                else nested_concat(all_preds, logits, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels
                if all_labels is None
                else nested_concat(all_labels, labels, padding_index=-100)
            )

        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

    eval_pred = all_preds, all_labels
    val_loss_dict = estimator._compute_metrics_by_dataset_name(eval_pred)
    metrics[auto_metric] = val_loss_dict["val_loss"]

    estimator._metric = estimator_metric_cache
    return metrics["eval_loss"], metrics


@pytest.mark.skipif(sys.platform == "darwin", reason="do not run on mac os")
def test_custom_metric():
    from flaml import AutoML
    import requests
    from datasets import load_dataset

    try:
        train_dataset = (
            load_dataset("glue", "mrpc", split="train").to_pandas().iloc[0:4]
        )
        dev_dataset = load_dataset("glue", "mrpc", split="train").to_pandas().iloc[0:4]
    except requests.exceptions.ConnectionError:
        return

    custom_sent_keys = ["sentence1", "sentence2"]
    label_key = "label"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    automl = AutoML()

    # testing when max_iter=1 and do retrain only without hpo

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 1,
        "time_budget": 5,
        "task": "seq-classification",
        "metric": custom_metric,
        "log_file_name": "seqclass.log",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "google/electra-small-discriminator",
        "output_dir": "data/output/",
        "ckpt_per_epoch": 5,
        "fp16": False,
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )

    # testing calling custom metric in TransformersEstimator._compute_metrics_by_dataset_name

    automl_settings["max_iter"] = 3
    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings
    )

    del automl


if __name__ == "__main__":
    test_custom_metric()
