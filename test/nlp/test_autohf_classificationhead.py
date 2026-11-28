"""Offline classification-head compatibility using tiny locally created checkpoints."""

import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")


def _create_local_checkpoint(directory, num_labels):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "good", "bad", "example"]
    vocabulary.extend(f"unused{index}" for index in range(32 - len(vocabulary)))
    (directory / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(directory / "vocab.txt")).save_pretrained(directory)
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=num_labels,
    )
    BertForSequenceClassification(config).save_pretrained(directory)


@pytest.mark.parametrize("source_labels", [1, 2, 3])
@pytest.mark.parametrize("target_labels", [1, 2, 3])
def test_switch_classification_head_with_local_checkpoint(tmp_path, source_labels, target_labels):
    from flaml.automl.nlp.huggingface.utils import load_model
    from flaml.automl.task.task import SEQCLASSIFICATION, SEQREGRESSION

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            _create_local_checkpoint(tmp_path, source_labels)
            task = SEQREGRESSION if target_labels == 1 else SEQCLASSIFICATION
            model = load_model(str(tmp_path), task=task, num_labels=target_labels)
            labels = torch.tensor([0.1, 0.2]) if target_labels == 1 else torch.tensor([0, 1])
            output = model(
                input_ids=torch.tensor([[2, 5, 6, 3], [2, 7, 8, 3]]),
                attention_mask=torch.ones((2, 4), dtype=torch.long),
                labels=labels,
            )
            assert model.config.num_labels == target_labels
            assert output.logits.shape == (2, target_labels)
            assert torch.isfinite(output.logits).all()
            assert torch.isfinite(output.loss)
            output.loss.backward()
            assert model.classifier.weight.grad is not None
            assert torch.isfinite(model.classifier.weight.grad).all()
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("task,num_labels", [("seq-classification", 2), ("seq-regression", 1)])
def test_offline_automl_fit_and_predict_with_local_checkpoint(tmp_path, task, num_labels):
    import numpy as np
    import pandas as pd

    from flaml import AutoML

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        _create_local_checkpoint(checkpoint, 3)
        X = pd.DataFrame({"text": ["hello world", "good example", "bad example", "hello example"] * 3})
        y = np.array([0, 1, 0, 1] * 3) if num_labels == 2 else np.linspace(0.0, 1.0, len(X))
        automl = AutoML(
            estimator_list=["transformer"],
            max_iter=2,
            time_budget=30,
            n_jobs=1,
            verbose=0,
            mlflow_logging=False,
            retrain_full=False,
            featurization="off",
            custom_hp={
                "transformer": {
                    "global_max_steps": {"domain": 2, "init_value": 2},
                    "num_train_epochs": {"domain": 1, "init_value": 1},
                    "per_device_train_batch_size": {"domain": 4, "init_value": 4},
                }
            },
        )
        automl.fit(
            X.iloc[:8],
            y[:8],
            X_val=X.iloc[8:],
            y_val=y[8:],
            task=task,
            metric="accuracy" if num_labels == 2 else "r2",
            fit_kwargs_by_estimator={
                "transformer": {
                    "model_path": str(checkpoint),
                    "output_dir": str(tmp_path / "training"),
                    "fp16": False,
                    "max_seq_length": 16,
                    "report_to": [],
                    "disable_tqdm": True,
                    "logging_steps": 1,
                    "eval_steps": 1,
                    "save_steps": 1,
                }
            },
            gpu_per_trial=0,
        )
        assert automl.model is not None
        predictions = np.asarray(automl.predict(X.iloc[8:]))
        assert predictions.shape == (4,)
        assert np.isfinite(predictions).all()
        assert automl.model.estimator["model"].config.num_labels == num_labels
        if num_labels == 2:
            assert set(predictions) <= {0, 1}
    finally:
        torch.set_num_threads(previous_threads)
