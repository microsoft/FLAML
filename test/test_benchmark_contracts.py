import gzip
import json
import random
import runpy
import sys
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("runner", ["runner_flaml", "runner_h2o"])
def test_download_failure_can_be_written_to_benchmark_report(monkeypatch, runner):
    pmlb = ModuleType("pmlb")
    pmlb.fetch_data = Mock(side_effect=OSError("download unavailable"))
    h2o = ModuleType("h2o")
    h2o.init = Mock()
    monkeypatch.setitem(sys.modules, "pmlb", pmlb)
    monkeypatch.setitem(sys.modules, "h2o", h2o)
    script = Path(__file__).resolve().parents[1] / "benchmark" / "pmlb" / "runbenchmark.py"
    random_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        module = runpy.run_path(str(script))
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
    result = module[runner]({"dataset": "missing", "metric": "r2", "task": "regression"})
    assert result["score"] is None
    assert result["metric"] == "r2"
    assert result["status"] == "download_failed"
    assert "download unavailable" in result["error"]
    flattened = module["flatten_results"]({"missing": {runner: result}})
    assert flattened["missing"][f"{runner}_score"] is None
    assert flattened["missing"][f"{runner}_status"] == "download_failed"
    h2o.init.assert_not_called()


def test_dataset_extraction_preserves_existing_csv_on_repeat(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "benchmark" / "pmlb" / "extract_dataset.py"
    source = tmp_path / "benchmark" / "pmlb" / "datasets" / "pmlb" / "datasets" / "tsv_values"
    source.mkdir(parents=True)
    with gzip.open(source / "tsv_values.tsv.gz", "wt", encoding="utf-8") as file:
        file.write("feature\ttarget\n1\t2\n")
    destination = tmp_path / "benchmark" / "pmlb" / "csv_datasets"
    destination.mkdir(parents=True)
    untouched = destination / "untouched.csv"
    untouched.write_text("value\n99\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for _ in range(2):
        runpy.run_path(str(script), run_name="__main__")
        assert untouched.read_text(encoding="utf-8") == "value\n99\n"
        assert pd.read_csv(destination / "tsv_values.csv").to_dict("list") == {"feature": [1], "target": [2]}


def test_notebook_download_failure_does_not_break_task_selection(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "benchmark" / "pmlb" / "build_benchmark.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    runner_source = next(source for source in sources if "def runner_flaml" in source)
    selection_source = next(source for source in sources if "selected_tasks = []" in source)
    pmlb = ModuleType("pmlb")
    pmlb.fetch_data = Mock(side_effect=OSError("download unavailable"))
    monkeypatch.setitem(sys.modules, "pmlb", pmlb)
    namespace = {"pmlb": pmlb, "BENCHMARK_TASKS": {"missing": {"dataset": "missing", "metric": "r2"}}}
    exec(compile(runner_source, "benchmark-runner-cell", "exec"), namespace)
    failed = namespace["runner_flaml"]("missing")
    assert failed["status"] == "download_failed"
    assert failed["score"] is None
    namespace["result"] = {
        "missing": failed,
        "legacy_failure": {"score": 0, "duration": 0},
        "successful": {"score": 0.7, "duration": 1, "metric": "r2"},
    }
    exec(compile(selection_source, "benchmark-selection-cell", "exec"), namespace)
    assert namespace["selected_tasks"] == [{"dataset": "successful", "metric": "r2", "result": 0.7}]


def test_notebook_successful_fetch_reaches_real_model_fitting(monkeypatch):
    from types import SimpleNamespace

    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import train_test_split

    from flaml import AutoML

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "benchmark" / "pmlb" / "build_benchmark.ipynb").read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def runner_flaml" in "".join(cell.get("source", []))
    )
    X = np.random.RandomState(7).normal(size=(40, 3))
    pmlb = ModuleType("pmlb")
    pmlb.fetch_data = lambda *args, **kwargs: (X, X[:, 0])
    monkeypatch.setitem(sys.modules, "pmlb", pmlb)
    namespace = {
        "pmlb": pmlb,
        "flaml": SimpleNamespace(AutoML=lambda: AutoML(n_jobs=1, verbose=0, mlflow_logging=False)),
        "BENCHMARK_TASKS": {"local": {"dataset": "local", "metric": "r2", "task": "regression", "size": "small"}},
        "ITERS_SIZE": {"small": 1},
        "train_test_split": train_test_split,
        "accuracy_score": accuracy_score,
        "r2_score": r2_score,
        "time": time,
    }
    exec(compile(source, "benchmark-success-cell", "exec"), namespace)
    result = namespace["runner_flaml"]("local")
    assert result["status"] == "ok"
    assert result["error"] is None
    assert np.isfinite(result["score"])


@pytest.mark.parametrize(
    "task,labels", [("classification", [0, 1]), ("classification", [0, 1, 2]), ("regression", [0.1, 0.2])]
)
def test_h2o_response_typing_and_untouched_test_partition(monkeypatch, task, labels):
    frames = []

    class Frame:
        def __init__(self, values, factor=False):
            self.values = values.copy()
            self.factor = factor

        def __getitem__(self, name):
            return Frame(self.values[[name]], self.factor)

        def __setitem__(self, name, column):
            self.values[name] = column.values[name]
            self.factor = column.factor

        def asfactor(self):
            return Frame(self.values, factor=True)

        def as_data_frame(self):
            return self.values.copy()

        def split_frame(self, **kwargs):
            assert self.factor is (task == "classification")
            middle = len(self.values) // 2
            train = Frame(self.values.iloc[:middle], self.factor)
            test = Frame(self.values.iloc[middle:], self.factor)
            frames.extend([train, test])
            return train, test

    pmlb = ModuleType("pmlb")
    pmlb.fetch_data = Mock(return_value=pd.DataFrame({"x": range(len(labels) * 2), "target": labels * 2}))
    performance = Mock()
    performance.r2.return_value = 0.25
    performance.accuracy.side_effect = AssertionError("Do not select a threshold using test labels")
    learner = Mock()
    learner.leader.model_performance.return_value = performance
    predictions = [0, 0] if len(labels) == 2 else [0, 0, 2]
    learner.leader.predict.return_value = Frame(pd.DataFrame({"predict": predictions}))
    h2o = ModuleType("h2o")
    h2o.init = Mock()
    h2o.H2OFrame = Frame
    h2o.automl = Mock()
    h2o.automl.H2OAutoML.return_value = learner
    monkeypatch.setitem(sys.modules, "pmlb", pmlb)
    monkeypatch.setitem(sys.modules, "h2o", h2o)
    script = Path(__file__).resolve().parents[1] / "benchmark" / "pmlb" / "runbenchmark.py"
    module = runpy.run_path(str(script))
    metric = "accuracy" if task == "classification" else "r2"
    result = module["runner_h2o"]({"dataset": "local", "metric": metric, "task": task})
    arguments = learner.train.call_args.kwargs
    assert arguments["training_frame"] is frames[0]
    assert "leaderboard_frame" not in arguments
    assert "validation_frame" not in arguments
    expected = (0.5 if len(labels) == 2 else 2 / 3) if task == "classification" else 0.25
    assert result["score"] == pytest.approx(expected)
    assert isinstance(result["score"], float)


def test_flaml_loss_curve_cell_uses_flaml_values_and_elapsed_seconds():
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "benchmark" / "pmlb" / "analyze.ipynb").read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "def plot_flaml_runs_loss_curve" in "".join(cell.get("source", []))
    )
    plot = Mock()
    title = Mock()
    frame = pd.DataFrame({"Start Time": ["2026-01-01 01:00:00", "2026-01-01 00:00:00"], "val_log_loss": [0.2, 0.4]})
    namespace = {
        "pd": pd,
        "flaml_runs_example": frame,
        "plt": SimpleNamespace(plot=plot, title=title, xlabel=Mock(), ylabel=Mock(), figure=Mock(), show=Mock()),
    }
    exec(compile(source, "flaml-loss-curve-cell", "exec"), namespace)
    np.testing.assert_allclose(plot.call_args.args[0], [0.0, 3600.0])
    np.testing.assert_allclose(plot.call_args.args[1], [0.4, 0.2])
    title.assert_called_once_with("Loss Curve for FLAML AutoML Experiment")
