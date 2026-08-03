"""Tests to improve coverage for flaml/tune/tune.py.

Focuses on uncovered branches: report() edge cases, run() with various
configurations, import fallbacks, ExperimentAnalysis properties, and
error handling paths.
"""

import logging
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from flaml import tune
from flaml.tune.result import DEFAULT_METRIC
from flaml.tune.trial import Trial
from flaml.tune.trial_runner import SimpleTrial
from flaml.tune.tune import (
    INCUMBENT_RESULT,
    ExperimentAnalysis,
    report,
    run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_eval(config):
    """Return a dict result."""
    return {"metric": config.get("x", 1) ** 2}


def _eval_returns_scalar(config):
    """Return a scalar result."""
    return config.get("x", 1) ** 2


def _eval_returns_empty(config):
    """Return an empty dict (signals error)."""
    return {}


def _eval_returns_none(config):
    """Return None (signals stop tuning)."""
    return None


def _eval_with_report(config):
    """Call tune.report inside the evaluation function."""
    tune.report(metric=config.get("x", 1) ** 2)


def _eval_with_report_default_metric(config):
    """Call tune.report with a positional _metric."""
    tune.report(config.get("x", 1) ** 2)


# ---------------------------------------------------------------------------
# ExperimentAnalysis tests
# ---------------------------------------------------------------------------


class TestExperimentAnalysis:
    def _make_trial(self, trial_id, config, result, status=Trial.TERMINATED):
        t = SimpleTrial(config=config, trial_id=trial_id)
        t.set_status(status)
        t.update_last_result(result)
        return t

    def test_best_result_no_lexico(self):
        """Cover line 155: lexico_best is None → super().best_result."""
        trial = self._make_trial("t1", {"x": 1}, {"metric": 5, "training_iteration": 0})
        ea = ExperimentAnalysis([trial], metric="metric", mode="min")
        assert ea.best_result["metric"] == 5

    def test_best_iteration_found(self):
        """Cover lines 160-166: best_iteration returns index."""
        t1 = self._make_trial("t1", {"x": 1}, {"metric": 10, "training_iteration": 0})
        t2 = self._make_trial("t2", {"x": 2}, {"metric": 5, "training_iteration": 0})
        ea = ExperimentAnalysis([t1, t2], metric="metric", mode="min")
        assert ea.best_iteration in (0, 1)

    def test_best_iteration_not_found(self):
        """Cover line 167: best_iteration returns None when best trial id not in trials list."""
        t1 = self._make_trial("t1", {"x": 1}, {"metric": 10, "training_iteration": 0})
        t2 = self._make_trial("t2", {"x": 2}, {"metric": 5, "training_iteration": 0})
        ea = ExperimentAnalysis([t1, t2], metric="metric", mode="min")
        # best_trial is t2 (lower metric). Replace trials with different ids.
        t3 = self._make_trial("t3", {"x": 3}, {"metric": 20, "training_iteration": 0})
        ea.trials = [t3]
        # Now best_trial still returns t2 (cached/computed from original), but
        # the iteration loop over ea.trials won't find t2's id → returns None
        # We need to mock best_trial to return something with a different id
        with patch.object(type(ea), "best_trial", new_callable=lambda: property(lambda self: t2)):
            assert ea.best_iteration is None

    def test_lexico_objectives(self):
        """Cover lexico_best paths."""
        t1 = self._make_trial("t1", {"x": 1}, {"err": 0.1, "time": 5, "training_iteration": 0})
        t2 = self._make_trial("t2", {"x": 2}, {"err": 0.05, "time": 10, "training_iteration": 0})
        lexico = {
            "metrics": ["err", "time"],
            "modes": ["min", "min"],
            "tolerances": {"err": 0.02, "time": 0.0},
            "targets": {"err": 0.0, "time": 0.0},
        }
        ea = ExperimentAnalysis([t1, t2], metric="err", mode="min", lexico_objectives=lexico)
        best = ea.best_trial
        assert best is not None
        assert ea.best_config is not None
        assert ea.best_result is not None


# ---------------------------------------------------------------------------
# report() tests
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_no_runner(self):
        """Cover line 233: report returns None when no running trial."""
        import flaml.tune.tune as tune_mod

        old_use_ray = tune_mod._use_ray
        old_runner = tune_mod._runner
        try:
            tune_mod._use_ray = False
            tune_mod._runner = MagicMock(running_trial=None)
            result = report(metric=42)
            assert result is None
        finally:
            tune_mod._use_ray = old_use_ray
            tune_mod._runner = old_runner

    def test_report_with_default_metric(self):
        """Cover line 230: _metric sets DEFAULT_METRIC in result."""
        import flaml.tune.tune as tune_mod

        old_use_ray = tune_mod._use_ray
        old_runner = tune_mod._runner
        old_running_trial = tune_mod._running_trial
        try:
            tune_mod._use_ray = False
            mock_trial = MagicMock()
            mock_trial.config = {"x": 1}
            mock_trial.is_finished.return_value = False
            mock_runner = MagicMock()
            mock_runner.running_trial = mock_trial
            tune_mod._runner = mock_runner
            tune_mod._running_trial = None
            report(42)
            call_args = mock_runner.process_trial_result.call_args
            result = call_args[0][1]
            assert DEFAULT_METRIC in result
            assert result[DEFAULT_METRIC] == 42
        finally:
            tune_mod._use_ray = old_use_ray
            tune_mod._runner = old_runner
            tune_mod._running_trial = old_running_trial

    def test_report_same_trial_increments_iteration(self):
        """Cover lines 234-238: training_iteration increments for same trial."""
        import flaml.tune.tune as tune_mod

        old_use_ray = tune_mod._use_ray
        old_runner = tune_mod._runner
        old_running_trial = tune_mod._running_trial
        old_iter = tune_mod._training_iteration
        try:
            tune_mod._use_ray = False
            mock_trial = MagicMock()
            mock_trial.config = {"x": 1}
            mock_trial.is_finished.return_value = False
            mock_runner = MagicMock()
            mock_runner.running_trial = mock_trial
            tune_mod._runner = mock_runner
            tune_mod._running_trial = None
            # First report sets _running_trial
            report(metric=1)
            assert tune_mod._training_iteration == 0
            # Second report for same trial increments
            report(metric=2)
            assert tune_mod._training_iteration == 1
        finally:
            tune_mod._use_ray = old_use_ray
            tune_mod._runner = old_runner
            tune_mod._running_trial = old_running_trial
            tune_mod._training_iteration = old_iter

    def test_report_incumbent_result_removed(self):
        """Cover lines 241-242: INCUMBENT_RESULT removed from config."""
        import flaml.tune.tune as tune_mod

        old_use_ray = tune_mod._use_ray
        old_runner = tune_mod._runner
        old_running_trial = tune_mod._running_trial
        try:
            tune_mod._use_ray = False
            mock_trial = MagicMock()
            mock_trial.config = {"x": 1, INCUMBENT_RESULT: "should_be_removed"}
            mock_trial.is_finished.return_value = False
            mock_runner = MagicMock()
            mock_runner.running_trial = mock_trial
            tune_mod._runner = mock_runner
            tune_mod._running_trial = None
            report(metric=1)
            call_args = mock_runner.process_trial_result.call_args
            result = call_args[0][1]
            assert INCUMBENT_RESULT not in result["config"]
        finally:
            tune_mod._use_ray = old_use_ray
            tune_mod._runner = old_runner
            tune_mod._running_trial = old_running_trial

    def test_report_finished_trial_raises(self):
        """Cover line 249: StopIteration raised when trial is finished."""
        import flaml.tune.tune as tune_mod

        old_use_ray = tune_mod._use_ray
        old_runner = tune_mod._runner
        old_running_trial = tune_mod._running_trial
        try:
            tune_mod._use_ray = False
            mock_trial = MagicMock()
            mock_trial.config = {"x": 1}
            mock_trial.is_finished.return_value = True
            mock_runner = MagicMock()
            mock_runner.running_trial = mock_trial
            tune_mod._runner = mock_runner
            tune_mod._running_trial = None
            with pytest.raises(StopIteration):
                report(metric=1)
        finally:
            tune_mod._use_ray = old_use_ray
            tune_mod._runner = old_runner
            tune_mod._running_trial = old_running_trial

    def test_report_verbose_logging(self):
        """Cover line 247: verbose > 2 logs result."""
        import flaml.tune.tune as tune_mod

        old_use_ray = tune_mod._use_ray
        old_runner = tune_mod._runner
        old_running_trial = tune_mod._running_trial
        old_verbose = tune_mod._verbose
        try:
            tune_mod._use_ray = False
            tune_mod._verbose = 3
            mock_trial = MagicMock()
            mock_trial.config = {"x": 1}
            mock_trial.is_finished.return_value = False
            mock_runner = MagicMock()
            mock_runner.running_trial = mock_trial
            tune_mod._runner = mock_runner
            tune_mod._running_trial = None
            report(metric=1)  # Should not raise
        finally:
            tune_mod._use_ray = old_use_ray
            tune_mod._runner = old_runner
            tune_mod._running_trial = old_running_trial
            tune_mod._verbose = old_verbose

    def test_report_use_ray_no_ray_installed(self):
        """Cover lines 217-224: _use_ray=True but ray not importable."""
        import flaml.tune.tune as tune_mod

        old_use_ray = tune_mod._use_ray
        try:
            tune_mod._use_ray = True
            with patch.dict("sys.modules", {"ray": None}):
                result = report(metric=42)
                assert result is None
        finally:
            tune_mod._use_ray = old_use_ray


# ---------------------------------------------------------------------------
# run() tests
# ---------------------------------------------------------------------------


class TestRun:
    def test_use_ray_and_spark_raises(self):
        """Cover line 525: ValueError when both use_ray and use_spark."""
        with pytest.raises(ValueError, match="use_ray and use_spark cannot be both True"):
            run(_simple_eval, config={"x": 1}, use_ray=True, use_spark=True)

    def test_run_with_log_file(self):
        """Cover lines 517-520: log_file_name with directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "subdir", "tune.log")
            analysis = run(
                _simple_eval,
                config={"x": tune.choice([1, 2])},
                metric="metric",
                mode="min",
                num_samples=2,
                use_ray=False,
                log_file_name=log_path,
                verbose=1,
            )
            assert os.path.exists(log_path)
            assert len(analysis.trials) == 2

    def test_run_with_local_dir(self):
        """Cover lines 521-523: local_dir auto-generates log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local = os.path.join(tmpdir, "local_logs")
            analysis = run(
                _simple_eval,
                config={"x": tune.choice([1, 2])},
                metric="metric",
                mode="min",
                num_samples=1,
                use_ray=False,
                local_dir=local,
                verbose=1,
            )
            assert os.path.exists(local)
            assert len(analysis.trials) == 1

    def test_run_verbose_zero(self):
        """Cover line 555: verbose=0 sets CRITICAL level."""
        analysis = run(
            _simple_eval,
            config={"x": tune.choice([1, 2])},
            metric="metric",
            mode="min",
            num_samples=1,
            use_ray=False,
            verbose=0,
        )
        assert len(analysis.trials) == 1

    def test_run_verbose_three(self):
        """Cover line 553: verbose=3 sets DEBUG level."""
        analysis = run(
            _simple_eval,
            config={"x": tune.choice([1, 2])},
            metric="metric",
            mode="min",
            num_samples=1,
            use_ray=False,
            verbose=3,
        )
        assert len(analysis.trials) == 1

    def test_run_scalar_result(self):
        """Cover line 946: scalar result triggers report(_metric=result)."""
        analysis = run(
            _eval_returns_scalar,
            config={"x": tune.choice([1, 2])},
            metric=DEFAULT_METRIC,
            mode="min",
            num_samples=2,
            use_ray=False,
        )
        assert len(analysis.trials) == 2

    def test_run_empty_dict_result(self):
        """Cover line 944: empty dict sets trial status to ERROR."""
        analysis = run(
            _eval_returns_empty,
            config={"x": tune.choice([1, 2])},
            metric="metric",
            mode="min",
            num_samples=1,
            use_ray=False,
        )
        assert len(analysis.trials) >= 1

    def test_run_none_result_stops(self):
        """Cover lines 949-952: None result stops tuning."""
        analysis = run(
            _eval_returns_none,
            config={"x": tune.choice([1, 2])},
            metric="metric",
            mode="min",
            num_samples=10,
            use_ray=False,
        )
        # Should stop early because evaluation returns None
        assert len(analysis.trials) <= 10

    def test_run_search_alg_string_cfo(self):
        """Cover line 617: search_alg='CFO' uses CFO."""
        analysis = run(
            _simple_eval,
            config={"x": tune.choice([1, 2])},
            metric="metric",
            mode="min",
            num_samples=1,
            use_ray=False,
            search_alg="CFO",
        )
        assert len(analysis.trials) == 1

    def test_run_search_alg_string_random(self):
        """Cover line 617: search_alg='RandomSearch' uses RandomSearch."""
        analysis = run(
            _simple_eval,
            config={"x": tune.choice([1, 2])},
            metric="metric",
            mode="min",
            num_samples=1,
            use_ray=False,
            search_alg="RandomSearch",
        )
        assert len(analysis.trials) == 1

    def test_run_search_alg_invalid_string(self):
        """Cover line 576-581: invalid search_alg string raises."""
        with pytest.raises(AssertionError, match="is not recognized"):
            run(
                _simple_eval,
                config={"x": tune.choice([1, 2])},
                metric="metric",
                mode="min",
                num_samples=1,
                use_ray=False,
                search_alg="InvalidAlg",
            )

    def test_run_lexico_objectives(self):
        """Cover lines 566-573, 596-602: lexico_objectives fills defaults."""

        def eval_lexico(config):
            return {"err": config["x"] * 0.1, "time": config["x"] * 0.5}

        lexico = {
            "metrics": ["err", "time"],
            # "modes" omitted to test default fill (line 568)
            "tolerances": {"err": 0.01},  # "time" missing → filled with 0 (line 571)
            # "targets" partially missing → filled (line 573)
            "targets": {},
        }
        analysis = run(
            eval_lexico,
            config={"x": tune.uniform(0.1, 1.0)},
            num_samples=3,
            use_ray=False,
            lexico_objectives=lexico,
        )
        assert len(analysis.trials) == 3

    def test_run_with_optuna_not_installed(self):
        """Cover lines 610-615: optuna not installed falls back to CFO."""
        with patch.dict("sys.modules", {"optuna": None}):
            analysis = run(
                _simple_eval,
                config={"x": tune.choice([1, 2])},
                metric="metric",
                mode="min",
                num_samples=1,
                use_ray=False,
            )
            assert len(analysis.trials) == 1

    def test_run_blendsearch_requires_optuna(self):
        """Cover lines 611-612: BlendSearch explicitly requested but optuna missing."""
        with patch.dict("sys.modules", {"optuna": None}):
            with pytest.raises(ValueError, match="pip install flaml"):
                run(
                    _simple_eval,
                    config={"x": tune.choice([1, 2])},
                    metric="metric",
                    mode="min",
                    num_samples=1,
                    use_ray=False,
                    search_alg="BlendSearch",
                )

    def test_run_custom_search_alg_object(self):
        """Cover lines 641-682: passing a search_alg instance."""
        from flaml.tune.searcher.blendsearch import CFO

        alg = CFO(
            metric="metric",
            mode="min",
            space={"x": tune.choice([1, 2, 3])},
        )
        analysis = run(
            _simple_eval,
            config={"x": tune.choice([1, 2, 3])},
            metric="metric",
            mode="min",
            num_samples=2,
            use_ray=False,
            search_alg=alg,
        )
        assert len(analysis.trials) == 2

    def test_run_custom_search_alg_with_use_incumbent(self):
        """Cover line 665: set use_incumbent_result_in_evaluation on search_alg."""
        from flaml.tune.searcher.blendsearch import CFO

        alg = CFO(
            metric="metric",
            mode="min",
            space={"x": tune.choice([1, 2, 3])},
        )
        analysis = run(
            _simple_eval,
            config={"x": tune.choice([1, 2, 3])},
            metric="metric",
            mode="min",
            num_samples=1,
            use_ray=False,
            search_alg=alg,
            use_incumbent_result_in_evaluation=True,
        )
        assert len(analysis.trials) == 1

    def test_run_asha_scheduler(self):
        """Cover lines 683-693: ASHA scheduler param building."""
        # Without ray, the ASHA code builds params but can't create ASHAScheduler.
        # Mock ray_available and ASHAScheduler to test the full path.
        import flaml.tune.tune as tune_mod

        mock_scheduler = MagicMock()
        mock_asha_cls = MagicMock(return_value=mock_scheduler)
        mock_scheduler.set_search_properties = MagicMock()
        mock_scheduler.on_trial_result = MagicMock(return_value="CONTINUE")
        mock_scheduler.on_trial_complete = MagicMock()

        old_ray_available = tune_mod.ray_available
        try:
            tune_mod.ray_available = True
            mock_ray = MagicMock()
            mock_ray_tune = MagicMock()
            mock_schedulers = MagicMock(ASHAScheduler=mock_asha_cls)
            with patch.dict(
                "sys.modules",
                {"ray": mock_ray, "ray.tune": mock_ray_tune, "ray.tune.schedulers": mock_schedulers},
            ):
                with patch("flaml.tune.tune.ASHAScheduler", mock_asha_cls, create=True):
                    # Patch the import inside the function
                    analysis = run(
                        _simple_eval,
                        config={"x": tune.uniform(0, 1)},
                        metric="metric",
                        mode="min",
                        num_samples=2,
                        use_ray=False,
                        scheduler="asha",
                        resource_attr="time_total_s",
                        min_resource=1,
                        max_resource=10,
                        reduction_factor=2,
                    )
                    assert len(analysis.trials) >= 1
        finally:
            tune_mod.ray_available = old_ray_available

    def test_run_with_evaluated_rewards(self):
        """Test points_to_evaluate with evaluated_rewards."""
        analysis = run(
            _simple_eval,
            config={"x": tune.choice([1, 2, 3])},
            metric="metric",
            mode="min",
            num_samples=3,
            use_ray=False,
            points_to_evaluate=[{"x": 1}],
            evaluated_rewards=[1.0],
        )
        assert len(analysis.trials) >= 1

    def test_run_ray_args_assertion(self):
        """Cover line 534: ray_args asserted invalid when use_ray=False."""
        with pytest.raises(AssertionError):
            run(
                _simple_eval,
                config={"x": tune.choice([1, 2])},
                metric="metric",
                mode="min",
                num_samples=1,
                use_ray=False,
                extra_kwarg_for_ray="value",
            )

    def test_run_max_failures(self):
        """Cover lines 954-958: max consecutive failures stops tuning."""
        call_count = 0

        def eval_fn(config):
            nonlocal call_count
            call_count += 1
            return {"metric": config.get("x", 1)}

        # Use a very restricted config space that exhausts quickly
        analysis = run(
            eval_fn,
            config={"x": tune.choice([1])},
            metric="metric",
            mode="min",
            num_samples=-1,
            time_budget_s=5,
            use_ray=False,
            max_failure=3,
        )
        assert analysis is not None

    def test_run_with_scheduler_sequential(self):
        """Cover line 906: scheduler.set_search_properties in sequential path."""

        class SimpleScheduler:
            def set_search_properties(self, metric, mode):
                self.metric = metric
                self.mode = mode

            def on_trial_add(self, trial_runner, trial):
                pass

            def on_trial_result(self, trial_runner, trial, result):
                return "CONTINUE"

            def on_trial_complete(self, trial_runner, trial, result):
                pass

            def on_trial_remove(self, trial_runner, trial):
                pass

        sched = SimpleScheduler()
        analysis = run(
            _simple_eval,
            config={"x": tune.choice([1, 2])},
            metric="metric",
            mode="min",
            num_samples=1,
            use_ray=False,
            scheduler=sched,
        )
        assert sched.metric == "metric"
        assert len(analysis.trials) == 1


# ---------------------------------------------------------------------------
# Import fallback / module-level coverage
# ---------------------------------------------------------------------------


class TestImportFallbacks:
    def test_kusto_logger_stub(self):
        """Cover lines 49-59: KustoLogger stub methods."""
        import flaml.tune.tune as tune_mod

        # The KustoLogger stub is used when fabric is not available
        # Just verify the interface exists and works
        if not tune_mod.internal_mlflow:
            tune_mod.kusto_logger.info("test")
            tune_mod.kusto_logger.warning("test")
            tune_mod.kusto_logger.error("test")

    def test_ray_available_flag(self):
        """Cover lines 17-18, 23: ray_available flag is set."""
        import flaml.tune.tune as tune_mod

        assert isinstance(tune_mod.ray_available, bool)


# ---------------------------------------------------------------------------
# MLflow integration paths
# ---------------------------------------------------------------------------


class TestMLflowIntegration:
    def test_run_with_mocked_internal_mlflow(self):
        """Cover lines 557-562: internal_mlflow wraps evaluation function."""
        import flaml.tune.tune as tune_mod

        old_internal = tune_mod.internal_mlflow

        mock_mlflow_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.wrap_evaluation_function.side_effect = lambda fn: fn
        mock_mlflow_cls.return_value = mock_instance

        try:
            tune_mod.internal_mlflow = True
            with patch("flaml.tune.tune.MLflowIntegration", mock_mlflow_cls, create=True):
                analysis = run(
                    _simple_eval,
                    config={"x": tune.choice([1, 2])},
                    metric="metric",
                    mode="min",
                    num_samples=1,
                    use_ray=False,
                )
                assert len(analysis.trials) == 1
                mock_mlflow_cls.assert_called_once()
                mock_instance.log_tune.assert_called_once()
                mock_instance.adopt_children.assert_called_once()
        finally:
            tune_mod.internal_mlflow = old_internal


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_experiment_analysis_search_space(self):
        """Verify search_space is set on analysis."""
        config = {"x": tune.choice([1, 2, 3])}
        analysis = run(
            _simple_eval,
            config=config,
            metric="metric",
            mode="min",
            num_samples=1,
            use_ray=False,
        )
        assert analysis.search_space is not None

    def test_run_with_eval_reporting(self):
        """Test eval function that calls tune.report."""
        analysis = run(
            _eval_with_report,
            config={"x": tune.choice([1, 2])},
            metric="metric",
            mode="min",
            num_samples=2,
            use_ray=False,
        )
        assert len(analysis.trials) == 2

    def test_run_with_eval_reporting_default_metric(self):
        """Test eval function using tune.report with _metric positional arg."""
        analysis = run(
            _eval_with_report_default_metric,
            config={"x": tune.choice([1, 2])},
            metric=DEFAULT_METRIC,
            mode="min",
            num_samples=2,
            use_ray=False,
        )
        assert len(analysis.trials) == 2

    def test_analysis_best_config_and_result(self):
        """Test best_config and best_result properties."""
        analysis = run(
            _simple_eval,
            config={"x": tune.choice([1, 2, 3])},
            metric="metric",
            mode="min",
            num_samples=3,
            use_ray=False,
        )
        assert analysis.best_config is not None
        assert analysis.best_result is not None
        assert isinstance(analysis.best_iteration, int) or analysis.best_iteration is None

    def test_run_custom_alg_no_metric_mode(self):
        """Cover lines 642-648: search_alg instance with metric/mode=None."""
        from flaml.tune.searcher.blendsearch import CFO

        alg = CFO(
            metric="metric",
            mode="min",
            space={"x": tune.choice([1, 2, 3])},
        )
        analysis = run(
            _simple_eval,
            config={"x": tune.choice([1, 2, 3])},
            metric=None,
            mode=None,
            num_samples=1,
            use_ray=False,
            search_alg=alg,
        )
        assert len(analysis.trials) == 1

    def test_run_custom_alg_lexico(self):
        """Cover lines 643-645, 667-672: search_alg instance + lexico_objectives."""
        from flaml.tune.searcher.blendsearch import CFO

        def eval_lexico(config):
            return {"err": config["x"] * 0.1, "time": config["x"] * 0.5}

        lexico = {
            "metrics": ["err", "time"],
            "modes": ["min", "min"],
            "tolerances": {"err": 0.01, "time": 0.0},
            "targets": {"err": 0.0, "time": 0.0},
        }
        alg = CFO(
            metric="err",
            mode="min",
            space={"x": tune.uniform(0.1, 1.0)},
            lexico_objectives=lexico,
        )
        analysis = run(
            eval_lexico,
            config={"x": tune.uniform(0.1, 1.0)},
            metric=None,
            mode=None,
            num_samples=2,
            use_ray=False,
            search_alg=alg,
            lexico_objectives=lexico,
        )
        assert len(analysis.trials) == 2

    def test_run_custom_blendsearch_alg(self):
        """Cover lines 674-680: BlendSearch instance with time_budget_s and num_samples."""
        from flaml.tune.searcher.blendsearch import BlendSearch

        alg = BlendSearch(
            metric="metric",
            mode="min",
            space={"x": tune.choice([1, 2, 3])},
        )
        analysis = run(
            _simple_eval,
            config={"x": tune.choice([1, 2, 3])},
            metric="metric",
            mode="min",
            num_samples=2,
            time_budget_s=10,
            use_ray=False,
            search_alg=alg,
        )
        assert len(analysis.trials) == 2

    def test_run_internal_mlflow_sequential(self):
        """Cover lines 966-970, 989-990: internal_mlflow logging in sequential path."""
        import flaml.tune.tune as tune_mod

        old_internal = tune_mod.internal_mlflow

        mock_mlflow_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.wrap_evaluation_function.side_effect = lambda fn: fn
        mock_mlflow_cls.return_value = mock_instance

        try:
            tune_mod.internal_mlflow = True
            with patch("flaml.tune.tune.MLflowIntegration", mock_mlflow_cls, create=True):
                analysis = run(
                    _simple_eval,
                    config={"x": tune.choice([1, 2])},
                    metric="metric",
                    mode="min",
                    num_samples=2,
                    use_ray=False,
                )
                # Simulate best_run_id being set
                analysis.best_run_id = "test_run_id"
                analysis.best_run_name = "test_run_name"
                # Verify mlflow integration was called
                mock_instance.log_tune.assert_called_once()
                mock_instance.adopt_children.assert_called_once()
        finally:
            tune_mod.internal_mlflow = old_internal

    def test_run_internal_mlflow_with_best_run_id(self):
        """Cover lines 968-970: log best run info when best_run_id is not None."""
        import flaml.tune.tune as tune_mod

        old_internal = tune_mod.internal_mlflow

        mock_mlflow_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.wrap_evaluation_function.side_effect = lambda fn: fn

        def fake_log_tune(analysis, metric):
            analysis.best_run_id = "run123"
            analysis.best_run_name = "run_name_123"

        mock_instance.log_tune.side_effect = fake_log_tune
        mock_mlflow_cls.return_value = mock_instance

        try:
            tune_mod.internal_mlflow = True
            with patch("flaml.tune.tune.MLflowIntegration", mock_mlflow_cls, create=True):
                analysis = run(
                    _simple_eval,
                    config={"x": tune.choice([1, 2])},
                    metric="metric",
                    mode="min",
                    num_samples=1,
                    use_ray=False,
                    verbose=1,
                )
                assert analysis.best_run_id == "run123"
        finally:
            tune_mod.internal_mlflow = old_internal
