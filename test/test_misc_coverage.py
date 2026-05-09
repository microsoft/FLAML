"""Tests for miscellaneous coverage gaps across the FLAML codebase.

Organized by source file. Focuses on easy-to-cover branches:
import fallbacks, simple error handlers, edge-case branches, deprecated shims.
"""

import importlib
import subprocess
import sys
import warnings
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. flaml/__init__.py – ImportError branches for automl and fabric.telemetry
#    Run in subprocesses to avoid corrupting the main process module cache.
# ---------------------------------------------------------------------------
class TestFlamlInit:
    def test_automl_import_error_branch(self):
        """Cover lines 8-9, 31: ImportError when flaml.automl is unavailable."""
        code = (
            "import sys, warnings; "
            "sys.modules['flaml.automl'] = None; "
            "warnings.simplefilter('always'); "
            "import flaml; "
            "assert not flaml.has_automl; "
            "print('OK')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        assert "OK" in result.stdout, result.stderr

    def test_telemetry_import_error_branch(self):
        """Cover lines 18-19: ImportError when flaml.fabric.telemetry is unavailable."""
        code = (
            "import sys; "
            "sys.modules['flaml.fabric'] = None; "
            "sys.modules['flaml.fabric.telemetry'] = None; "
            "import flaml; "
            "assert not flaml.is_log_telemetry; "
            "print('OK')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        assert "OK" in result.stdout, result.stderr


# ---------------------------------------------------------------------------
# 2. flaml/ml.py – deprecated shim
# ---------------------------------------------------------------------------
class TestFlamlMl:
    def test_import_deprecated_ml(self):
        """Cover lines 1, 3, 5: importing flaml.ml triggers DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import flaml.ml  # noqa: F401

            importlib.reload(flaml.ml)
            deprecation_msgs = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("flaml.ml" in str(m.message) for m in deprecation_msgs)


# ---------------------------------------------------------------------------
# 5. flaml/tune/scheduler/trial_scheduler.py – on_trial_add / on_trial_remove
# ---------------------------------------------------------------------------
class TestTrialScheduler:
    def test_on_trial_add_and_remove_are_noop(self):
        """Cover lines 30, 33: both methods are pass-through."""
        from flaml.tune.scheduler.trial_scheduler import TrialScheduler
        from flaml.tune.trial import Trial

        scheduler = TrialScheduler()
        mock_runner = MagicMock()
        mock_trial = MagicMock(spec=Trial)
        # Should not raise
        assert scheduler.on_trial_add(mock_runner, mock_trial) is None
        assert scheduler.on_trial_remove(mock_runner, mock_trial) is None


# ---------------------------------------------------------------------------
# 6. flaml/tune/searcher/search_thread.py
# ---------------------------------------------------------------------------
class TestSearchThread:
    def test_recursive_dict_update(self):
        """Cover the _recursive_dict_update helper."""
        from flaml.tune.searcher.search_thread import _recursive_dict_update

        target = {"a": {"x": 1, "y": 2}, "b": 3}
        source = {"a": {"z": 3}, "b": 10}
        _recursive_dict_update(target, source)
        assert target == {"a": {"x": 1, "y": 2, "z": 3}, "b": 10}

    def test_search_thread_suggest_floating_point_error(self):
        """Cover lines 97-99: FloatingPointError branch in suggest."""
        from flaml.tune.searcher.search_thread import SearchThread
        from flaml.tune.searcher.suggestion import Searcher

        mock_alg = MagicMock()
        mock_alg.cost_incumbent = 0
        mock_alg.best_obj = np.inf
        mock_alg.space = None
        mock_alg._space = {}
        mock_alg.suggest = MagicMock(side_effect=FloatingPointError("overflow"))

        thread = SearchThread(mode="min", search_alg=mock_alg)
        result = thread.suggest("trial_0")
        assert result is None

    def test_search_thread_on_trial_complete_runtime_error_reraised(self):
        """Cover lines 136, 138-139: RuntimeError that doesn't match optuna message is re-raised."""
        from flaml.tune.searcher.search_thread import SearchThread

        class FakeAlg:
            cost_incumbent = 0
            best_obj = np.inf
            space = None
            _space = {}
            metric = "loss"
            lexico_objectives = None

            def on_trial_complete(self, *args, **kwargs):
                raise RuntimeError("some other error")

        mock_alg = FakeAlg()

        thread = SearchThread(mode="min", search_alg=mock_alg)
        thread._is_ls = True
        thread._init_config = False
        thread.running = 1
        with pytest.raises(RuntimeError, match="some other error"):
            thread.on_trial_complete("t1", result={"loss": 0.5, "time_total_s": 1.0})

    def test_search_thread_on_trial_complete_runtime_error_suppressed(self):
        """Cover lines 136, 138-139: RuntimeError matching optuna finish message is suppressed."""
        from flaml.tune.searcher.search_thread import SearchThread

        class FakeAlg:
            cost_incumbent = 0
            best_obj = np.inf
            space = None
            _space = {}
            metric = "loss"
            lexico_objectives = None

            def on_trial_complete(self, *args, **kwargs):
                raise RuntimeError("Trial t1 has already finished and can not be updated.")

        mock_alg = FakeAlg()

        thread = SearchThread(mode="min", search_alg=mock_alg)
        thread._is_ls = True
        thread._init_config = False
        thread.running = 1
        # Should not raise
        thread.on_trial_complete("t1", result={"loss": 0.5, "time_total_s": 1.0})

    def test_search_thread_on_trial_result_runtime_error(self):
        """Cover lines 172, 174-175: RuntimeError in on_trial_result."""
        from flaml.tune.searcher.search_thread import SearchThread

        class FakeAlg:
            cost_incumbent = 0
            best_obj = np.inf
            space = None
            _space = {}
            _ot_trials = {"t1": True}

            def on_trial_result(self, *args, **kwargs):
                raise RuntimeError("unexpected error")

        mock_alg = FakeAlg()

        thread = SearchThread(mode="min", search_alg=mock_alg)
        with pytest.raises(RuntimeError, match="unexpected error"):
            thread.on_trial_result("t1", {"time_total_s": 1.0})

    def test_search_thread_on_trial_result_suppressed(self):
        """Cover lines 172, 174-175: RuntimeError with optuna finish message suppressed."""
        from flaml.tune.searcher.search_thread import SearchThread

        class FakeAlg:
            cost_incumbent = 0
            best_obj = np.inf
            space = None
            _space = {}
            _ot_trials = {"t1": True}

            def on_trial_result(self, *args, **kwargs):
                raise RuntimeError("Trial t1 has already finished and can not be updated.")

        mock_alg = FakeAlg()

        thread = SearchThread(mode="min", search_alg=mock_alg)
        # Should not raise
        thread.on_trial_result("t1", {"time_total_s": 1.0})

    def test_search_thread_suggest_with_unflatten(self):
        """Cover lines 94-96: suggest where _space is not a dict (define-by-run)."""
        from flaml.tune.searcher.search_thread import SearchThread
        from flaml.tune.searcher.suggestion import Searcher

        mock_alg = MagicMock()
        mock_alg.cost_incumbent = 0
        mock_alg.best_obj = np.inf
        mock_alg._space = "not_a_dict"
        mock_alg.space = {"x": 1}

        mock_alg.suggest = MagicMock(return_value={"x": 1})

        thread = SearchThread(mode="min", search_alg=mock_alg)
        thread._is_ls = False

        with patch("flaml.tune.searcher.search_thread.unflatten_hierarchical", return_value=({"x": 1}, {"x": 1})):
            config = thread.suggest("t1")
        assert config is not None


# ---------------------------------------------------------------------------
# 7. flaml/tune/analysis.py – ExperimentAnalysis validation
# ---------------------------------------------------------------------------
class TestExperimentAnalysis:
    def _make_analysis(self, default_metric=None, default_mode=None, trials=None):
        from flaml.tune.analysis import ExperimentAnalysis

        analysis = ExperimentAnalysis.__new__(ExperimentAnalysis)
        analysis.default_metric = default_metric
        analysis.default_mode = default_mode
        analysis.trials = trials or []
        return analysis

    def test_best_trial_no_metric(self):
        """Cover line 44: best_trial raises when no default_metric."""
        analysis = self._make_analysis(default_metric=None, default_mode="min")
        with pytest.raises(ValueError, match="best_trial"):
            _ = analysis.best_trial

    def test_best_config_no_metric(self):
        """Cover line 61: best_config raises when no default_metric."""
        analysis = self._make_analysis(default_metric=None, default_mode="min")
        with pytest.raises(ValueError, match="best_config"):
            _ = analysis.best_config

    def test_validate_metric_no_metric(self):
        """Cover line 76: _validate_metric raises."""
        analysis = self._make_analysis()
        with pytest.raises(ValueError, match="metric"):
            analysis._validate_metric("")

    def test_validate_mode_no_mode(self):
        """Cover line 84: _validate_mode raises when no mode."""
        analysis = self._make_analysis()
        with pytest.raises(ValueError, match="mode"):
            analysis._validate_mode("")

    def test_validate_mode_invalid(self):
        """Cover line 89: _validate_mode raises for invalid mode."""
        analysis = self._make_analysis()
        with pytest.raises(ValueError, match="min, max"):
            analysis._validate_mode("invalid")

    def test_get_best_trial_invalid_scope(self):
        """Cover line 126: invalid scope raises."""
        analysis = self._make_analysis(default_metric="m", default_mode="min")
        with pytest.raises(ValueError, match="scope"):
            analysis.get_best_trial("m", "min", scope="invalid")

    def test_get_best_trial_with_nan_filter(self):
        """Cover line 141: filter NaN metric scores."""
        mock_trial = MagicMock()
        mock_trial.metric_analysis = {"m": {"last": float("nan")}}
        analysis = self._make_analysis(default_metric="m", default_mode="min", trials=[mock_trial])
        result = analysis.get_best_trial("m", "min")
        assert result is None

    def test_get_best_trial_max_mode(self):
        """Cover the max mode comparison branch."""
        t1 = MagicMock()
        t1.metric_analysis = {"m": {"last": 0.5}}
        t2 = MagicMock()
        t2.metric_analysis = {"m": {"last": 0.9}}
        analysis = self._make_analysis(default_metric="m", default_mode="max", trials=[t1, t2])
        best = analysis.get_best_trial("m", "max")
        assert best is t2

    def test_get_best_trial_scope_all(self):
        """Cover line 141: scope='all' uses mode key."""
        t1 = MagicMock()
        t1.metric_analysis = {"m": {"min": 0.1, "max": 0.9}}
        analysis = self._make_analysis(default_metric="m", default_mode="min", trials=[t1])
        best = analysis.get_best_trial("m", "min", scope="all")
        assert best is t1

    def test_best_result_no_metric(self):
        """Cover lines 199-200: best_result raises when no metric/mode."""
        analysis = self._make_analysis()
        with pytest.raises(ValueError, match="best_result"):
            _ = analysis.best_result

    def test_best_result_success(self):
        """Cover line 206: best_result returns last_result."""
        t1 = MagicMock()
        t1.metric_analysis = {"m": {"last": 0.5}}
        t1.last_result = {"m": 0.5}
        analysis = self._make_analysis(default_metric="m", default_mode="min", trials=[t1])
        assert analysis.best_result == {"m": 0.5}

    def test_results_property(self):
        """Cover line 72: results property."""
        t1 = MagicMock()
        t1.trial_id = "t1"
        t1.last_result = {"m": 0.5}
        analysis = self._make_analysis(default_metric="m", default_mode="min", trials=[t1])
        assert analysis.results == {"t1": {"m": 0.5}}


# ---------------------------------------------------------------------------
# 8. flaml/tune/sample.py – Domain, Sampler, and helper classes
# ---------------------------------------------------------------------------
class TestSample:
    def test_np_random_generator_exists(self):
        """Cover lines 28-29: np_random_generator attribute check (numpy >= 1.17)."""
        from flaml.tune.sample import LEGACY_RNG

        assert LEGACY_RNG is False  # numpy >= 1.17

    def test_backwards_compatible_numpy_rng_with_seed(self):
        """Cover lines 62-64: _BackwardsCompatibleNumpyRng with int seed."""
        from flaml.tune.sample import _BackwardsCompatibleNumpyRng

        rng = _BackwardsCompatibleNumpyRng(42)
        assert rng._rng is not None

    def test_backwards_compatible_numpy_rng_legacy_rng(self):
        """Cover lines 67-68: legacy_rng property."""
        from flaml.tune.sample import _BackwardsCompatibleNumpyRng

        rng = _BackwardsCompatibleNumpyRng(np.random.RandomState(42))
        assert rng.legacy_rng is True

    def test_backwards_compatible_numpy_rng_property(self):
        """Cover lines 72-73: rng property when _rng is None."""
        from flaml.tune.sample import _BackwardsCompatibleNumpyRng

        rng = _BackwardsCompatibleNumpyRng(None)
        assert rng.rng is np.random

    def test_backwards_compatible_numpy_rng_getattr_legacy(self):
        """Cover lines 76-82: __getattr__ with legacy rng name mapping."""
        from flaml.tune.sample import _BackwardsCompatibleNumpyRng

        rng = _BackwardsCompatibleNumpyRng(np.random.RandomState(42))
        # 'integers' should map to 'randint' for legacy
        func = rng.integers
        assert callable(func)
        # 'random' should map to 'rand'
        func2 = rng.random
        assert callable(func2)

    def test_domain_cast(self):
        """Cover line 102: Domain.cast is identity."""
        from flaml.tune.sample import Domain

        d = Domain()
        assert d.cast(42) == 42

    def test_domain_set_sampler_override(self):
        """Cover lines 106-112: set_sampler with override."""
        from flaml.tune.sample import Domain, Uniform

        d = Domain()
        d.set_sampler(Uniform())
        with pytest.raises(ValueError, match="one sampler"):
            d.set_sampler(Uniform())
        d.set_sampler(Uniform(), allow_override=True)

    def test_domain_is_grid(self):
        """Cover line 132: is_grid."""
        from flaml.tune.sample import Domain, Grid

        d = Domain()
        assert d.is_grid() is False
        d.sampler = Grid()
        assert d.is_grid() is True

    def test_domain_is_function(self):
        """Cover line 135: is_function."""
        from flaml.tune.sample import Domain

        d = Domain()
        assert d.is_function() is False

    def test_domain_is_valid_raises(self):
        """Cover line 139: is_valid raises NotImplementedError."""
        from flaml.tune.sample import Domain

        d = Domain()
        with pytest.raises(NotImplementedError):
            d.is_valid(1)

    def test_domain_str(self):
        """Cover line 143: domain_str."""
        from flaml.tune.sample import Domain

        d = Domain()
        assert d.domain_str == "(unknown)"

    def test_sampler_sample_raises(self):
        """Cover line 154: Sampler.sample raises."""
        from flaml.tune.sample import Sampler

        s = Sampler()
        with pytest.raises(NotImplementedError):
            s.sample(None)

    def test_base_sampler_str(self):
        """Cover line 159: BaseSampler.__str__."""
        from flaml.tune.sample import BaseSampler

        assert str(BaseSampler()) == "Base"

    def test_uniform_str(self):
        """Cover line 164: Uniform.__str__."""
        from flaml.tune.sample import Uniform

        assert str(Uniform()) == "Uniform"

    def test_loguniform_str(self):
        """Cover line 173: LogUniform.__str__."""
        from flaml.tune.sample import LogUniform

        assert str(LogUniform(10)) == "LogUniform"

    def test_normal_str(self):
        """Cover line 184: Normal.__str__."""
        from flaml.tune.sample import Normal

        assert str(Normal(0, 1)) == "Normal"

    def test_grid_sample(self):
        """Cover line 197: Grid.sample returns RuntimeError."""
        from flaml.tune.sample import Grid

        g = Grid()
        result = g.sample(None)
        assert isinstance(result, RuntimeError)

    def test_float_domain_str(self):
        """Cover line 311: Float.domain_str."""
        from flaml.tune.sample import Float

        f = Float(0.0, 1.0)
        assert f.domain_str == "(0.0, 1.0)"

    def test_float_is_valid(self):
        """Cover line 307: Float.is_valid."""
        from flaml.tune.sample import Float

        f = Float(0.0, 1.0)
        assert f.is_valid(0.5) is True
        assert f.is_valid(1.5) is False

    def test_float_cast(self):
        """Cover line 261: Float.cast."""
        from flaml.tune.sample import Float

        f = Float(0, 1)
        assert isinstance(f.cast(1), float)

    def test_float_uniform_no_lower_bound(self):
        """Cover line 265: Float.uniform with no lower bound."""
        from flaml.tune.sample import Float

        with pytest.raises(ValueError, match="lower bound"):
            Float(None, 1.0).uniform()

    def test_float_uniform_no_upper_bound(self):
        """Cover line 267: Float.uniform with no upper bound."""
        from flaml.tune.sample import Float

        with pytest.raises(ValueError, match="upper bound"):
            Float(0.0, None).uniform()

    def test_float_loguniform_negative_lower(self):
        """Cover line 274: Float.loguniform negative lower."""
        from flaml.tune.sample import Float

        with pytest.raises(ValueError, match="lower bound greater than 0"):
            Float(-1.0, 1.0).loguniform()

    def test_float_loguniform_inf_upper(self):
        """Cover line 281: Float.loguniform infinite upper."""
        from flaml.tune.sample import Float

        with pytest.raises(ValueError, match="upper bound greater than 0"):
            Float(0.1, float("inf")).loguniform()

    def test_float_quantized_lower_not_divisible(self):
        """Cover line 298: Float.quantized lower not divisible."""
        from flaml.tune.sample import Float

        with pytest.raises(ValueError, match="not divisible"):
            Float(0.3, 1.0).uniform().quantized(0.2)

    def test_float_quantized_upper_not_divisible(self):
        """Cover line 300: Float.quantized upper not divisible."""
        from flaml.tune.sample import Float

        with pytest.raises(ValueError, match="not divisible"):
            Float(0.0, 0.3).uniform().quantized(0.2)

    def test_integer_cast(self):
        """Cover line 354: Integer.cast."""
        from flaml.tune.sample import Integer

        i = Integer(0, 10)
        assert isinstance(i.cast(5.5), int)

    def test_integer_is_valid(self):
        """Cover line 386: Integer.is_valid."""
        from flaml.tune.sample import Integer

        i = Integer(0, 10)
        assert i.is_valid(5) is True
        assert i.is_valid(11) is False

    def test_integer_domain_str(self):
        """Cover line 390: Integer.domain_str."""
        from flaml.tune.sample import Integer

        i = Integer(0, 10)
        assert i.domain_str == "(0, 10)"

    def test_integer_loguniform_bad_lower(self):
        """Cover line 368: Integer.loguniform negative lower."""
        from flaml.tune.sample import Integer

        with pytest.raises(ValueError, match="lower bound"):
            Integer(-1, 10).loguniform()

    def test_integer_loguniform_bad_upper(self):
        """Cover line 375: Integer.loguniform bad upper."""
        from flaml.tune.sample import Integer

        with pytest.raises(ValueError, match="upper bound"):
            Integer(1, float("inf")).loguniform()

    def test_categorical_is_valid(self):
        """Cover line 432: Categorical.is_valid."""
        from flaml.tune.sample import Categorical

        c = Categorical(["a", "b", "c"])
        assert c.is_valid("a") is True
        assert c.is_valid("d") is False

    def test_categorical_domain_str(self):
        """Cover line 436: Categorical.domain_str."""
        from flaml.tune.sample import Categorical

        c = Categorical(["a", "b"])
        assert c.domain_str == "['a', 'b']"

    def test_categorical_len_and_getitem(self):
        """Cover lines 426, 429: __len__ and __getitem__."""
        from flaml.tune.sample import Categorical

        c = Categorical(["x", "y", "z"])
        assert len(c) == 3
        assert c[1] == "y"

    def test_categorical_grid(self):
        """Cover lines 421-423: Categorical.grid."""
        from flaml.tune.sample import Categorical, Grid

        c = Categorical(["a", "b"]).grid()
        assert c.is_grid() is True

    def test_quantized_get_sampler(self):
        """Cover line 447: Quantized.get_sampler."""
        from flaml.tune.sample import Quantized, Uniform

        u = Uniform()
        q = Quantized(u, 0.1)
        assert q.get_sampler() is u

    def test_quantized_q_equals_1(self):
        """Cover line 460: Quantized with q=1 delegates directly."""
        from flaml.tune.sample import Float

        f = Float(0, 10).uniform().quantized(1)
        val = f.sample(random_state=42)
        assert isinstance(val, float)

    def test_polynomial_expansion_set(self):
        """Cover lines 486, 490, 494: PolynomialExpansionSet properties."""
        from flaml.tune.sample import PolynomialExpansionSet

        pes = PolynomialExpansionSet({"a", "b"}, highest_poly_order=3, allow_self_inter=True)
        assert pes.init_monomials == {"a", "b"}
        assert pes.highest_poly_order == 3
        assert pes.allow_self_inter is True
        assert str(pes) == "PolynomialExpansionSet"

    def test_polynomial_expansion_set_default_order(self):
        """Cover line 481: default highest_poly_order."""
        from flaml.tune.sample import PolynomialExpansionSet

        pes = PolynomialExpansionSet({"a", "b"})
        assert pes.highest_poly_order == 2

    def test_polynomial_expansion_set_function(self):
        """Cover line 613: polynomial_expansion_set function."""
        from flaml.tune.sample import polynomial_expansion_set

        pes = polynomial_expansion_set({"x", "y"}, highest_poly_order=2)
        assert pes.init_monomials == {"x", "y"}

    def test_float_normal_sampling(self):
        """Cover Float._Normal.sample."""
        from flaml.tune.sample import Float

        f = Float(None, None).normal(0, 1)
        val = f.sample(random_state=42)
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# 9. flaml/tune/trial_runner.py
# ---------------------------------------------------------------------------
class TestTrialRunner:
    def test_nologger(self):
        """Cover line 22: Nologger.on_result."""
        from flaml.tune.trial_runner import Nologger

        n = Nologger()
        assert n.on_result({"x": 1}) is None

    def test_base_trial_runner_add_trial_with_scheduler(self):
        """Cover line 81: scheduler.on_trial_add called."""
        from flaml.tune.trial_runner import BaseTrialRunner, SimpleTrial

        scheduler = MagicMock()
        runner = BaseTrialRunner(scheduler=scheduler)
        trial = SimpleTrial({"x": 1})
        runner.add_trial(trial)
        scheduler.on_trial_add.assert_called_once_with(runner, trial)

    def test_base_trial_runner_process_trial_result_stop(self):
        """Cover lines 89-91: scheduler returns STOP."""
        from flaml.tune.trial import Trial
        from flaml.tune.trial_runner import BaseTrialRunner, SimpleTrial

        scheduler = MagicMock()
        scheduler.on_trial_result.return_value = "STOP"
        search_alg = MagicMock()
        runner = BaseTrialRunner(search_alg=search_alg, scheduler=scheduler)
        trial = SimpleTrial({"x": 1})
        trial.set_status(Trial.RUNNING)
        runner.process_trial_result(trial, {"metric": 0.5, "training_iteration": 1})
        assert trial.status == Trial.TERMINATED

    def test_base_trial_runner_process_trial_result_pause(self):
        """Cover lines 92-93: scheduler returns PAUSE."""
        from flaml.tune.trial import Trial
        from flaml.tune.trial_runner import BaseTrialRunner, SimpleTrial

        scheduler = MagicMock()
        scheduler.on_trial_result.return_value = "PAUSE"
        search_alg = MagicMock()
        runner = BaseTrialRunner(search_alg=search_alg, scheduler=scheduler)
        trial = SimpleTrial({"x": 1})
        trial.set_status(Trial.RUNNING)
        runner.process_trial_result(trial, {"metric": 0.5, "training_iteration": 1})
        assert trial.status == Trial.PAUSED

    def test_base_trial_runner_stop_trial_with_scheduler(self):
        """Cover line 99: scheduler.on_trial_complete called on stop."""
        from flaml.tune.trial import Trial
        from flaml.tune.trial_runner import BaseTrialRunner, SimpleTrial

        scheduler = MagicMock()
        search_alg = MagicMock()
        runner = BaseTrialRunner(search_alg=search_alg, scheduler=scheduler)
        trial = SimpleTrial({"x": 1})
        trial.set_status(Trial.RUNNING)
        runner.stop_trial(trial)
        scheduler.on_trial_complete.assert_called_once()

    def test_base_trial_runner_stop_error_trial(self):
        """Cover lines 103-105: stop_trial with ERROR status."""
        from flaml.tune.trial import Trial
        from flaml.tune.trial_runner import BaseTrialRunner, SimpleTrial

        scheduler = MagicMock()
        search_alg = MagicMock()
        runner = BaseTrialRunner(search_alg=search_alg, scheduler=scheduler)
        trial = SimpleTrial({"x": 1})
        trial.set_status(Trial.ERROR)
        runner.stop_trial(trial)
        scheduler.on_trial_remove.assert_called_once()
        search_alg.on_trial_complete.assert_called_once_with(trial.trial_id, trial.last_result, error=True)


# ---------------------------------------------------------------------------
# 10. flaml/tune/searcher/suggestion.py – Searcher and ConcurrencyLimiter
# ---------------------------------------------------------------------------
class TestSuggestion:
    def test_searcher_init_with_mode_and_metric(self):
        """Cover Searcher init validation logic."""
        from flaml.tune.searcher.suggestion import Searcher

        s = Searcher(metric="m", mode="min")
        assert s.metric == "m"
        assert s.mode == "min"

    def test_searcher_init_list_mode(self):
        """Cover Searcher with list mode."""
        from flaml.tune.searcher.suggestion import Searcher

        s = Searcher(metric=["m1", "m2"], mode=["min", "max"])
        assert s.metric == ["m1", "m2"]

    def test_searcher_set_search_properties(self):
        """Cover Searcher.set_search_properties returns False by default."""
        from flaml.tune.searcher.suggestion import Searcher

        s = Searcher()
        assert s.set_search_properties("m", "min", {}) is False

    def test_searcher_on_trial_result_noop(self):
        """Cover Searcher.on_trial_result is a no-op."""
        from flaml.tune.searcher.suggestion import Searcher

        s = Searcher()
        assert s.on_trial_result("t1", {}) is None

    def test_concurrency_limiter_at_capacity(self):
        """Cover ConcurrencyLimiter.suggest returns None at capacity."""
        from flaml.tune.searcher.suggestion import ConcurrencyLimiter, Searcher

        inner = MagicMock()
        inner.metric = "m"
        inner.mode = "min"
        inner._metric = "m"
        inner._mode = "min"
        limiter = ConcurrencyLimiter(inner, max_concurrent=1)
        inner.suggest.return_value = {"x": 1}
        limiter.suggest("t1")
        result = limiter.suggest("t2")
        assert result is None

    def test_concurrency_limiter_batch_mode(self):
        """Cover ConcurrencyLimiter batch completion."""
        from flaml.tune.searcher.suggestion import ConcurrencyLimiter, Searcher

        inner = MagicMock()
        inner.metric = "m"
        inner.mode = "min"
        inner._metric = "m"
        inner._mode = "min"
        inner.suggest.return_value = {"x": 1}

        limiter = ConcurrencyLimiter(inner, max_concurrent=2, batch=True)
        limiter.suggest("t1")
        limiter.suggest("t2")
        # Complete one – should cache
        limiter.on_trial_complete("t1", result={"m": 0.5})
        assert "t1" in limiter.cached_results
        # Complete second – should flush
        limiter.on_trial_complete("t2", result={"m": 0.3})
        assert len(limiter.cached_results) == 0

    def test_concurrency_limiter_get_set_state(self):
        """Cover ConcurrencyLimiter.get_state / set_state."""
        from flaml.tune.searcher.suggestion import ConcurrencyLimiter, Searcher

        inner = MagicMock()
        inner.metric = "m"
        inner.mode = "min"
        inner._metric = "m"
        inner._mode = "min"
        limiter = ConcurrencyLimiter(inner, max_concurrent=2)
        state = limiter.get_state()
        assert "searcher" not in state
        limiter.set_state(state)

    def test_concurrency_limiter_delegated_methods(self):
        """Cover ConcurrencyLimiter delegate methods: save, restore, on_pause, on_unpause."""
        from flaml.tune.searcher.suggestion import ConcurrencyLimiter, Searcher

        inner = MagicMock()
        inner.metric = "m"
        inner.mode = "min"
        inner._metric = "m"
        inner._mode = "min"
        limiter = ConcurrencyLimiter(inner, max_concurrent=2)
        limiter.save("path")
        inner.save.assert_called_with("path")
        limiter.restore("path")
        inner.restore.assert_called_with("path")
        limiter.on_pause("t1")
        inner.on_pause.assert_called_with("t1")
        limiter.on_unpause("t1")
        inner.on_unpause.assert_called_with("t1")
        limiter.set_search_properties("m", "min", {})
        inner.set_search_properties.assert_called_with("m", "min", {})

    def test_validate_warmstart(self):
        """Cover validate_warmstart error paths."""
        from flaml.tune.searcher.suggestion import validate_warmstart

        # Type error for points_to_evaluate
        with pytest.raises(TypeError, match="points_to_evaluate expected to be a list"):
            validate_warmstart(["x"], "not_a_list", None)

        # Type error for individual point
        with pytest.raises(TypeError, match="include list or dict"):
            validate_warmstart(["x"], ["not_list_or_dict"], None)

        # Length mismatch
        with pytest.raises(ValueError, match="do not match"):
            validate_warmstart(["x", "y"], [[1]], None)

        # evaluated_rewards type error
        with pytest.raises(TypeError, match="evaluated_rewards expected to be a list"):
            validate_warmstart(["x"], [[1]], "not_a_list")

        # evaluated_rewards length mismatch
        with pytest.raises(ValueError, match="do not match"):
            validate_warmstart(["x"], [[1], [2]], [1.0])


# ---------------------------------------------------------------------------
# 11. flaml/tune/searcher/variant_generator.py
# ---------------------------------------------------------------------------
class TestVariantGenerator:
    def test_tune_error(self):
        """Cover TuneError class."""
        from flaml.tune.searcher.variant_generator import TuneError

        with pytest.raises(TuneError):
            raise TuneError("test error")

    def test_grid_search(self):
        """Cover grid_search function."""
        from flaml.tune.searcher.variant_generator import grid_search

        assert grid_search([1, 2, 3]) == {"grid_search": [1, 2, 3]}

    def test_generate_variants_no_vars(self):
        """Cover generate_variants with fully resolved spec."""
        from flaml.tune.searcher.variant_generator import generate_variants

        results = list(generate_variants({"x": 1, "y": 2}))
        assert len(results) == 1
        assert results[0][1] == {"x": 1, "y": 2}

    def test_generate_variants_with_grid(self):
        """Cover generate_variants with grid search."""
        from flaml.tune.searcher.variant_generator import generate_variants

        results = list(generate_variants({"x": {"grid_search": [1, 2]}}))
        assert len(results) == 2

    def test_has_unresolved_values(self):
        """Cover has_unresolved_values."""
        from flaml.tune.searcher.variant_generator import has_unresolved_values

        assert has_unresolved_values({"x": 1}) is False
        assert has_unresolved_values({"x": {"grid_search": [1, 2]}}) is True

    def test_assign_value(self):
        """Cover assign_value."""
        from flaml.tune.searcher.variant_generator import assign_value

        spec = {"a": {"b": 1}}
        assign_value(spec, ("a", "b"), 42)
        assert spec["a"]["b"] == 42

    def test_split_resolved_with_list(self):
        """Cover lines 281-289: _split_resolved_unresolved_values with lists."""
        from flaml.tune.searcher.variant_generator import (
            _split_resolved_unresolved_values,
        )

        spec = {"items": [1, 2, 3]}
        resolved, unresolved = _split_resolved_unresolved_values(spec)
        assert len(unresolved) == 0
        assert len(resolved) > 0

    def test_try_resolve_grid_not_list(self):
        """Cover TuneError when grid_search value is not a list."""
        from flaml.tune.searcher.variant_generator import TuneError, _try_resolve

        with pytest.raises(TuneError, match="expected list"):
            _try_resolve({"grid_search": "not_a_list"})

    def test_unresolved_access_guard(self):
        """Cover _UnresolvedAccessGuard and RecursiveDependencyError."""
        from flaml.tune.searcher.variant_generator import (
            RecursiveDependencyError,
            _UnresolvedAccessGuard,
        )

        guard = _UnresolvedAccessGuard({"x": 1, "y": {"grid_search": [1, 2]}})
        assert guard.x == 1
        with pytest.raises(RecursiveDependencyError):
            _ = guard.y

    def test_unresolved_access_guard_nested_dict(self):
        """Cover _UnresolvedAccessGuard with nested dict (returns guard)."""
        from flaml.tune.searcher.variant_generator import _UnresolvedAccessGuard

        guard = _UnresolvedAccessGuard({"nested": {"a": 1}})
        result = guard.nested
        assert isinstance(result, _UnresolvedAccessGuard)
        assert result["a"] == 1

    def test_generate_variants_domain(self):
        """Cover generate_variants with Domain variables."""
        from flaml.tune.sample import Float
        from flaml.tune.searcher.variant_generator import generate_variants

        spec = {"x": Float(0, 1).uniform()}
        results = list(generate_variants(spec, random_state=42))
        assert len(results) == 1
        assert 0 <= results[0][1]["x"] <= 1


# ---------------------------------------------------------------------------
# 13. flaml/fabric/logger.py – KustoLogger (no synapse branch)
# ---------------------------------------------------------------------------
class TestFabricLogger:
    def test_kusto_logger_methods(self):
        """Cover lines 17-30: KustoLogger no-op methods."""
        from flaml.fabric.logger import KustoLogger, init_kusto_logger

        kl = KustoLogger()
        assert kl.debug("msg") is None
        assert kl.info("msg") is None
        assert kl.warning("msg") is None
        assert kl.error("msg") is None
        assert kl.exception("msg") is None

    def test_init_kusto_logger(self):
        """Cover lines 34-35: init_kusto_logger returns KustoLogger instance."""
        from flaml.fabric.logger import KustoLogger, init_kusto_logger

        logger = init_kusto_logger("test")
        assert isinstance(logger, KustoLogger)

    def test_init_kusto_logger_default(self):
        """Cover line 35: init_kusto_logger with empty name."""
        from flaml.fabric.logger import KustoLogger, init_kusto_logger

        logger = init_kusto_logger()
        assert isinstance(logger, KustoLogger)


# ---------------------------------------------------------------------------
# 16. flaml/automl/nlp/huggingface/training_args.py – import fallback
# ---------------------------------------------------------------------------
class TestTrainingArgs:
    def test_training_args_import_fallback(self):
        """Cover lines 9-10: when transformers is not installed, TrainingArguments = object."""
        # We can test that TrainingArgumentsForAuto is importable
        # regardless of transformers presence
        from flaml.automl.nlp.huggingface.training_args import (
            TrainingArgumentsForAuto,
        )

        assert TrainingArgumentsForAuto is not None


# ---------------------------------------------------------------------------
# 17. flaml/automl/nlp/huggingface/trainer.py – import fallback
# ---------------------------------------------------------------------------
class TestTrainer:
    def test_trainer_import_fallback(self):
        """Cover lines 5-6: TrainerForAuto is importable."""
        from flaml.automl.nlp.huggingface.trainer import TrainerForAuto

        assert TrainerForAuto is not None


# ---------------------------------------------------------------------------
# flaml/tune/utils.py – choice function
# ---------------------------------------------------------------------------
class TestTuneUtils:
    def test_choice_ordered_numeric(self):
        """Cover choice with numeric categories (ordered=True auto)."""
        from flaml.tune.utils import choice

        domain = choice([1, 2, 3])
        assert domain.ordered is True

    def test_choice_ordered_string(self):
        """Cover choice with string categories (ordered=False auto)."""
        from flaml.tune.utils import choice

        domain = choice(["a", "b", "c"])
        assert domain.ordered is False

    def test_choice_explicit_order(self):
        """Cover choice with explicit order parameter."""
        from flaml.tune.utils import choice

        domain = choice(["a", "b"], order=True)
        assert domain.ordered is True


# ---------------------------------------------------------------------------
# flaml/tune/analysis.py – is_nan_or_inf helper
# ---------------------------------------------------------------------------
class TestAnalysisHelpers:
    def test_is_nan_or_inf(self):
        """Cover line 29: is_nan_or_inf."""
        from flaml.tune.analysis import is_nan_or_inf

        assert is_nan_or_inf(float("nan"))
        assert is_nan_or_inf(float("inf"))
        assert not is_nan_or_inf(1.0)


# ---------------------------------------------------------------------------
# flaml/tune/sample.py – top-level convenience functions
# ---------------------------------------------------------------------------
class TestSampleConvenienceFunctions:
    def test_uniform(self):
        from flaml.tune.sample import uniform

        d = uniform(0, 1)
        assert d.lower == 0.0

    def test_quniform(self):
        from flaml.tune.sample import quniform

        d = quniform(0, 10, 2)
        val = d.sample(random_state=42)
        assert isinstance(val, float)

    def test_loguniform(self):
        from flaml.tune.sample import loguniform

        d = loguniform(1e-4, 1e-2)
        val = d.sample(random_state=42)
        assert val > 0

    def test_qloguniform(self):
        from flaml.tune.sample import qloguniform

        d = qloguniform(1e-4, 1e-2, 1e-4)
        val = d.sample(random_state=42)
        assert val > 0

    def test_choice(self):
        from flaml.tune.sample import choice

        d = choice([1, 2, 3])
        val = d.sample(random_state=42)
        assert val in [1, 2, 3]

    def test_randint(self):
        from flaml.tune.sample import randint

        d = randint(0, 10)
        val = d.sample(random_state=42)
        assert isinstance(val, int)

    def test_lograndint(self):
        from flaml.tune.sample import lograndint

        d = lograndint(1, 100)
        val = d.sample(random_state=42)
        assert isinstance(val, int)

    def test_qrandint(self):
        from flaml.tune.sample import qrandint

        d = qrandint(0, 10, 2)
        val = d.sample(random_state=42)
        assert isinstance(val, (int, np.integer))

    def test_qlograndint(self):
        from flaml.tune.sample import qlograndint

        d = qlograndint(1, 100, 1)
        val = d.sample(random_state=42)
        assert isinstance(val, (int, np.integer))

    def test_randn(self):
        from flaml.tune.sample import randn

        d = randn(0, 1)
        val = d.sample(random_state=42)
        assert isinstance(val, float)

    def test_qrandn(self):
        from flaml.tune.sample import qrandn

        d = qrandn(0, 1, 0.1)
        val = d.sample(random_state=42)
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# flaml/tune/trial.py – Trial helpers
# ---------------------------------------------------------------------------
class TestTrial:
    def test_flatten_dict(self):
        """Cover flatten_dict."""
        from flaml.tune.trial import flatten_dict

        result = flatten_dict({"a": {"b": 1, "c": 2}})
        assert result == {"a/b": 1, "a/c": 2}

    def test_flatten_dict_prevent_delimiter(self):
        """Cover flatten_dict with prevent_delimiter."""
        from flaml.tune.trial import flatten_dict

        with pytest.raises(ValueError, match="delimiter"):
            flatten_dict({"a/b": 1}, prevent_delimiter=True)

    def test_unflatten_dict(self):
        """Cover unflatten_dict."""
        from flaml.tune.trial import unflatten_dict

        result = unflatten_dict({"a/b": 1, "a/c": 2})
        assert result == {"a": {"b": 1, "c": 2}}

    def test_trial_generate_id(self):
        """Cover Trial.generate_id."""
        from flaml.tune.trial import Trial

        tid = Trial.generate_id()
        assert isinstance(tid, str) and len(tid) == 8

    def test_trial_is_finished(self):
        """Cover Trial.is_finished."""
        from flaml.tune.trial_runner import SimpleTrial

        t = SimpleTrial({"x": 1})
        assert t.is_finished() is False
        t.set_status("TERMINATED")
        assert t.is_finished() is True


# ---------------------------------------------------------------------------
# flaml/tune/spark/utils.py – non-Spark paths
# ---------------------------------------------------------------------------
class TestSparkUtils:
    def test_get_broadcast_data_non_spark(self):
        """Cover get_broadcast_data when input is not a Broadcast."""
        from flaml.tune.spark.utils import get_broadcast_data

        data = [1, 2, 3]
        assert get_broadcast_data(data) == [1, 2, 3]

    def test_with_parameters_not_callable(self):
        """Cover with_parameters raises for non-callable."""
        from flaml.tune.spark.utils import with_parameters

        with pytest.raises(ValueError, match="only works with function"):
            with_parameters("not_callable")
