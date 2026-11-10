import json
import pickle
import threading

import numpy as np
import pytest

from flaml import AutoML
from flaml.automl.training_log import training_log_reader, training_log_writer


def append_trial(writer, n_estimators):
    writer.append(1, None, 0.1, 0.1, 1.0 / n_estimators, {"n_estimators": n_estimators}, "rf", 100)


def test_appended_trial_can_be_loaded_from_checkpoint(tmp_path):
    filename = tmp_path / "training.log"
    for count, append in ((4, False), (8, True), (16, True)):
        with training_log_writer(filename, append=append) as writer:
            append_trial(writer, count)
            writer.checkpoint()

    records = [json.loads(line) for line in filename.read_text().splitlines()]
    checkpoint_id = records[-1]["curr_best_record_id"]
    estimator = AutoML().get_estimator_from_log(str(filename), checkpoint_id, "classification")
    assert estimator.n_estimators == 16
    assert [record["record_id"] for record in records if "record_id" in record] == [0, 1, 2]
    assert [record["curr_best_record_id"] for record in records if "curr_best_record_id" in record] == [0, 1, 2]


@pytest.mark.parametrize("existing_ids", [[], [0], [2, 5], [5, 2], [0, 1, 0]])
def test_append_uses_next_unused_record_id(tmp_path, existing_ids):
    filename = tmp_path / "training.log"
    with training_log_writer(filename) as writer:
        for record_id in existing_ids:
            writer.current_record_id = record_id
            append_trial(writer, 4)
        writer.checkpoint()
    original = filename.read_bytes()

    with training_log_writer(filename, append=True) as writer:
        append_trial(writer, 8)
        append_trial(writer, 16)
        writer.checkpoint()

    assert filename.read_bytes().startswith(original)
    expected = max(existing_ids, default=-1) + 1
    with training_log_reader(filename) as reader:
        records = list(reader.records())
    assert [record.record_id for record in records] == existing_ids + [expected, expected + 1]
    with training_log_reader(filename) as reader:
        assert reader.get_record(expected).config == {"n_estimators": 8}


def test_append_creates_missing_log(tmp_path):
    filename = tmp_path / "new.log"
    with training_log_writer(filename, append=True) as writer:
        append_trial(writer, 4)

    with training_log_reader(filename) as reader:
        assert reader.get_record(0).config == {"n_estimators": 4}


def test_appended_checkpoint_keeps_current_run_best(tmp_path):
    filename = tmp_path / "training.log"
    with training_log_writer(filename) as writer:
        append_trial(writer, 16)
        writer.checkpoint()
    with training_log_writer(filename, append=True) as writer:
        append_trial(writer, 4)
        writer.checkpoint()

    records = [json.loads(line) for line in filename.read_text().splitlines()]
    assert records[-1] == {"curr_best_record_id": 1}
    with training_log_reader(filename) as reader:
        assert reader.get_record(1).config == {"n_estimators": 4}


def test_append_skips_checkpoint_only_log(tmp_path):
    filename = tmp_path / "training.log"
    filename.write_text('{"curr_best_record_id": 10}\n')
    with training_log_writer(filename, append=True) as writer:
        append_trial(writer, 4)
    with training_log_reader(filename) as reader:
        assert reader.get_record(0).config == {"n_estimators": 4}


@pytest.mark.parametrize("checkpoint_tail", [False, True])
@pytest.mark.parametrize("line_ending", [b"", b"\n", b"\r\n", b"\r"])
def test_append_preserves_valid_record_and_checkpoint_tails(tmp_path, checkpoint_tail, line_ending):
    filename = tmp_path / "training.log"
    with training_log_writer(filename) as writer:
        append_trial(writer, 4)
        if checkpoint_tail:
            writer.checkpoint()
    original = filename.read_bytes().rstrip(b"\r\n") + line_ending
    filename.write_bytes(original)

    with training_log_writer(filename, append=True) as writer:
        append_trial(writer, 8)
        writer.checkpoint()

    assert filename.read_bytes().startswith(original)
    with training_log_reader(filename) as reader:
        assert [record.record_id for record in reader.records()] == [0, 1]
    checkpoint_id = json.loads(filename.read_text().splitlines()[-1])["curr_best_record_id"]
    estimator = AutoML().get_estimator_from_log(str(filename), checkpoint_id, "classification")
    assert estimator.n_estimators == 8


@pytest.mark.parametrize("append", [False, True])
def test_writer_does_not_wait_for_another_fit(tmp_path, append):
    filename = tmp_path / "training.log"
    opened = threading.Event()
    errors = []

    def open_other_writer():
        try:
            with training_log_writer(filename, append=append):
                opened.set()
        except BaseException as error:
            errors.append(error)

    with training_log_writer(filename):
        other = threading.Thread(target=open_other_writer)
        other.start()
        opened_while_first_active = opened.wait(2)
    other.join(timeout=10)
    assert not other.is_alive()
    assert not errors
    assert opened_while_first_active, "Logging must not wait for another fit to finish"
    assert not filename.with_suffix(".log.lock").exists()


@pytest.mark.parametrize("append", [False, True])
def test_writer_closes_after_context_error(tmp_path, append):
    filename = tmp_path / "training.log"
    with pytest.raises(RuntimeError, match="training failed"):
        with training_log_writer(filename, append=append) as writer:
            append_trial(writer, 4)
            raise RuntimeError("training failed")

    writer.close()
    assert pickle.loads(pickle.dumps(writer)).file is None
    with training_log_reader(filename) as reader:
        assert reader.get_record(0).config == {"n_estimators": 4}


def test_append_preserves_invalid_log(tmp_path):
    filename = tmp_path / "training.log"
    original = '{"record_id":'
    filename.write_text(original)
    with pytest.raises(json.JSONDecodeError):
        with training_log_writer(filename, append=True):
            pytest.fail("An invalid log should not be opened for append")

    assert filename.read_text() == original
    assert not filename.with_suffix(".log.lock").exists()


def test_writer_handles_open_error(tmp_path):
    filename = tmp_path / "training.log"
    filename.mkdir()
    with pytest.raises(OSError):
        with training_log_writer(filename):
            pytest.fail("A directory should not be opened as a log file")

    assert not filename.with_suffix(".log.lock").exists()


def test_default_writer_overwrites_existing_records(tmp_path):
    filename = tmp_path / "training.log"
    for count in (4, 8):
        with training_log_writer(filename) as writer:
            append_trial(writer, count)
            writer.checkpoint()

    with training_log_reader(filename) as reader:
        records = list(reader.records())
    assert [record.record_id for record in records] == [0]
    assert records[0].config == {"n_estimators": 8}


def test_autofe_fit_append_preserves_records_without_resuming_previous_search(tmp_path):
    filename = tmp_path / "autofe.log"
    X = np.arange(120.0).reshape(40, 3)
    y = np.tile([0, 1], 20)
    automl = AutoML(
        estimator_list=["rf"],
        max_iter=2,
        n_jobs=1,
        verbose=0,
        mlflow_logging=False,
        featurization="auto",
        eval_method="holdout",
        log_type="all",
        log_file_name=str(filename),
    )
    automl.fit(X, y, metric=lambda *args: (0.1, {}))
    original_bytes = filename.read_bytes()
    with training_log_reader(filename) as reader:
        first = list(reader.records())
    assert first
    assert all(any(name.startswith("fe.") for name in record.config) for record in first)

    automl.fit(X, y, metric=lambda *args: (0.9, {}), append_log=True)
    assert filename.read_bytes().startswith(original_bytes)
    with training_log_reader(filename) as reader:
        combined = list(reader.records())
    assert len(combined) > len(first)
    assert [record.record_id for record in combined] == list(range(len(combined)))
    assert automl.best_loss == pytest.approx(0.9)
    assert all(record.validation_loss == pytest.approx(0.9) for record in combined[len(first) :])
    checkpoint = json.loads(filename.read_text().splitlines()[-1])["curr_best_record_id"]
    assert checkpoint >= len(first)
