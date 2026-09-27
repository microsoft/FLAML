import json

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
