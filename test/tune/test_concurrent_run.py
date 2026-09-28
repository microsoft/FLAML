"""Regression test for #996: two threads calling flaml.tune.run() concurrently
must not corrupt each other's trial state.

flaml.tune.run()/tune.report() used to coordinate through module-level globals
(_runner/_running_trial/_training_iteration/_use_ray/_verbose in flaml/tune/tune.py).
Those globals are shared by every thread, so whichever thread's run() call
executes _runner = SequentialTrialRunner(...) most recently "wins" the global
for every thread's subsequent report()/stop_trial() calls, including resetting
it to None (or another thread's runner) out from under a still-running call.

This forces the exact interleaving that corrupts state, deterministically (no
sleep-based timing): thread A steps its own trial and pauses inside its
evaluation function; thread B starts while A is paused, creates its own runner
(overwriting the shared state A is still relying on), steps its own trial, and
also pauses. A is released and runs to completion, all the way through
report(), stop_trial(), building its analysis, and its own finally-restore,
entirely while B is still paused. B is then released: on main, B's report()
silently drops its result
(the runner it reads has already been reset to whatever A's finally restored),
and B's stop_trial() call raises AttributeError on that stale/None runner.
"""

import concurrent.futures
import logging
import queue
import threading
from unittest import mock

import pytest

import flaml.tune.tune as tune_module
from flaml import tune
from flaml.tune.logger import logger
from flaml.tune.trial import Trial


def test_concurrent_tune_run_does_not_corrupt_state():
    a_paused = threading.Event()
    b_paused = threading.Event()
    release_a = threading.Event()
    release_b = threading.Event()
    a_done = threading.Event()
    results = {}

    def eval_a(config):
        a_paused.set()
        assert release_a.wait(timeout=5), "thread A was never released"
        return {"metric": 1.0}

    def eval_b(config):
        b_paused.set()
        assert release_b.wait(timeout=5), "thread B was never released"
        return {"metric": 2.0}

    def run_a():
        try:
            analysis = tune.run(
                eval_a,
                config={"x": tune.uniform(0, 1)},
                metric="metric",
                mode="min",
                num_samples=1,
                verbose=0,
            )
            results["A"] = ("ok", [t.last_result for t in analysis.trials])
        except Exception as e:  # noqa: BLE001 (captured for the test assertion below)
            results["A"] = ("error", f"{type(e).__name__}: {e}")
        finally:
            a_done.set()

    def run_b():
        try:
            analysis = tune.run(
                eval_b,
                config={"x": tune.uniform(0, 1)},
                metric="metric",
                mode="min",
                num_samples=1,
                verbose=0,
            )
            results["B"] = ("ok", [t.last_result for t in analysis.trials])
        except Exception as e:  # noqa: BLE001 (captured for the test assertion below)
            results["B"] = ("error", f"{type(e).__name__}: {e}")

    thread_a = threading.Thread(target=run_a)
    thread_b = threading.Thread(target=run_b)

    thread_a.start()
    assert a_paused.wait(timeout=5), "thread A never reached its evaluation function"

    thread_b.start()
    assert b_paused.wait(timeout=5), "thread B never reached its evaluation function"

    # Let A run to full completion, including its own finally-restore, while
    # B is still paused inside its evaluation function.
    release_a.set()
    assert a_done.wait(timeout=5), "thread A never finished"
    thread_a.join(timeout=5)

    release_b.set()
    thread_b.join(timeout=5)

    def metric_of(key):
        status, trials = results.get(key, ("missing", None))
        if status != "ok" or trials is None or len(trials) != 1:
            return None
        return trials[0].get("metric")

    assert results.get("A", ("missing",))[0] == "ok", f"thread A did not complete cleanly: {results.get('A')}"
    assert results.get("B", ("missing",))[0] == "ok", f"thread B did not complete cleanly: {results.get('B')}"
    assert metric_of("A") == 1.0, f"thread A's own trial did not receive thread A's own metric: {results}"
    assert metric_of("B") == 2.0, f"thread B's own trial did not receive thread B's own metric: {results}"


def test_concurrent_tune_run_logging_does_not_cross_contaminate(tmp_path):
    """Follow-up to #996: the five _TuneState fields are per-thread now, but
    tune.run() was still swapping the shared flaml.tune.logger logger's
    `.handlers`/level wholesale (`logger.handlers = []`, then re-add), which
    is the same corruption class on a sixth piece of shared state. Two
    concurrent tune.run() calls with different log_file_name used to lose
    each other's FileHandler mid-run and restore whichever handler list
    happened to be current when each one's finally block ran.

    Forces the same deterministic pause/release overlap as
    test_concurrent_tune_run_does_not_corrupt_state above: both threads have
    their own FileHandler simultaneously attached to the shared logger at
    the same time, not just simultaneously "in tune.run()".
    """
    a_paused = threading.Event()
    b_paused = threading.Event()
    release_a = threading.Event()
    release_b = threading.Event()

    log_a = str(tmp_path / "a.log")
    log_b = str(tmp_path / "b.log")

    handlers_before = list(logger.handlers)
    level_before = logger.getEffectiveLevel()

    def eval_a(config):
        logger.info("MARKER_FROM_A_PRE")
        a_paused.set()
        assert release_a.wait(timeout=5), "thread A was never released"
        logger.info("MARKER_FROM_A_POST")
        return {"metric": 1.0}

    def eval_b(config):
        logger.info("MARKER_FROM_B_PRE")
        b_paused.set()
        assert release_b.wait(timeout=5), "thread B was never released"
        logger.info("MARKER_FROM_B_POST")
        return {"metric": 2.0}

    def run_a():
        tune.run(
            eval_a,
            config={"x": tune.uniform(0, 1)},
            metric="metric",
            mode="min",
            num_samples=1,
            verbose=2,
            log_file_name=log_a,
        )

    def run_b():
        tune.run(
            eval_b,
            config={"x": tune.uniform(0, 1)},
            metric="metric",
            mode="min",
            num_samples=1,
            verbose=2,
            log_file_name=log_b,
        )

    thread_a = threading.Thread(target=run_a)
    thread_b = threading.Thread(target=run_b)

    thread_a.start()
    assert a_paused.wait(timeout=5), "thread A never reached its evaluation function"

    thread_b.start()
    assert b_paused.wait(timeout=5), "thread B never reached its evaluation function"

    # Both threads' own FileHandlers are attached to the shared logger right
    # now. Release A first and let it run to full completion, including its
    # own finally-restore, while B is still paused: a proper LIFO nesting
    # (innermost starts and finishes first) restores correctly even with the
    # old wholesale logger.handlers swap, so this crossing order is the one
    # that actually exercises the corruption (same shape as
    # test_concurrent_tune_run_does_not_corrupt_state above).
    release_a.set()
    thread_a.join(timeout=5)
    release_b.set()
    thread_b.join(timeout=5)

    text_a = open(log_a).read()
    text_b = open(log_b).read()

    for marker in ("MARKER_FROM_A_PRE", "MARKER_FROM_A_POST"):
        assert marker in text_a, f"thread A's own log file is missing {marker}: {text_a!r}"
        assert marker not in text_b, f"{marker} leaked into thread B's log file: {text_b!r}"
    for marker in ("MARKER_FROM_B_PRE", "MARKER_FROM_B_POST"):
        assert marker in text_b, f"thread B's own log file is missing {marker}: {text_b!r}"
        assert marker not in text_a, f"{marker} leaked into thread A's log file: {text_a!r}"

    assert logger.handlers == handlers_before, f"logger.handlers was not fully restored: {logger.handlers}"
    assert (
        logger.getEffectiveLevel() == level_before
    ), f"logger level was not restored: {logger.getEffectiveLevel()} != {level_before}"


def test_tune_run_setup_failure_restores_state():
    """Follow-up to #996: _state.use_ray/_state.verbose and the logger
    handler/level are mutated by the `if not use_ray:` block near the top of
    run(), before the try/finally that used to be the only thing restoring
    them. A failure later in setup (searcher/scheduler construction) used to
    skip straight past that try/finally and leak the mutated state into
    whatever tune.run() this thread calls next, nested or not.

    `search_alg="not-a-real-search-alg"` fails the validity assert inside
    run()'s searcher setup, after the logger/state mutation has already
    happened, which is exactly the gap being tested.
    """
    handlers_before = list(logger.handlers)
    level_before = logger.getEffectiveLevel()
    use_ray_before = tune.tune._state.use_ray
    verbose_before = tune.tune._state.verbose

    with pytest.raises(AssertionError):
        tune.run(
            lambda config: {"metric": 1.0},
            config={"x": tune.uniform(0, 1)},
            metric="metric",
            mode="min",
            num_samples=1,
            verbose=2,
            search_alg="not-a-real-search-alg",
        )

    assert logger.handlers == handlers_before, f"logger.handlers leaked past the setup failure: {logger.handlers}"
    assert logger.getEffectiveLevel() == level_before, "logger level leaked past the setup failure"
    assert tune.tune._state.use_ray == use_ray_before, "_state.use_ray leaked past the setup failure"
    assert tune.tune._state.verbose == verbose_before, "_state.verbose leaked past the setup failure"

    # And state genuinely was not left corrupted: an ordinary run right after
    # still works, rather than inheriting whatever the failed setup left behind.
    analysis = tune.run(
        lambda config: {"metric": 1.0},
        config={"x": tune.uniform(0, 1)},
        metric="metric",
        mode="min",
        num_samples=1,
        verbose=0,
    )
    assert len(analysis.trials) == 1
    assert analysis.trials[0].last_result.get("metric") == 1.0


def test_tune_report_from_trainable_spawned_thread():
    """Follow-up to #996, reviewer point 1 (second review): report() reads
    _state.runner, which is thread-local (per-thread by design, so
    concurrent tune.run() calls do not see each other's runner). A
    worker/callback thread that a trainable spawns on its own therefore
    starts from fresh _TuneState defaults, with no runner attached.

    An earlier version of this fix required the trainable to call
    get_run_context()/use_run_context() itself, and the reviewer asked for
    that to work automatically instead, for existing trainables that never
    call either. First half now checks exactly that: a plain
    threading.Thread with no FLAML-specific code in it still reports
    correctly, because run() sets _propagated_context around the
    evaluation_function() call and a patched threading.Thread.start()
    carries that ambient context into any thread spawned during it (see
    _install_thread_context_propagation() in tune.py). Second half checks
    the explicit get_run_context()/use_run_context() API still works too,
    for callers who want it (a persistent thread-pool executor, for
    instance, where the automatic patch can't reach individual submissions).
    """

    def eval_automatic_propagation(config):
        def worker():
            tune.report(metric=42.0)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)
        return None  # the worker thread's report() is the only result

    analysis = tune.run(
        eval_automatic_propagation,
        config={"x": tune.uniform(0, 1)},
        metric="metric",
        mode="min",
        num_samples=1,
        verbose=0,
    )
    assert analysis.trials[0].last_result is not None, (
        "report() from a plain worker thread with no explicit context call was dropped; "
        "expected automatic propagation via _propagated_context"
    )
    assert analysis.trials[0].last_result.get("metric") == 42.0

    def eval_with_explicit_propagation(config):
        ctx = tune.get_run_context()

        def worker():
            with tune.use_run_context(ctx):
                tune.report(metric=7.0)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)
        return None

    analysis2 = tune.run(
        eval_with_explicit_propagation,
        config={"x": tune.uniform(0, 1)},
        metric="metric",
        mode="min",
        num_samples=1,
        verbose=0,
    )
    assert analysis2.trials[0].last_result is not None, "propagated report() from the worker thread was dropped"
    assert analysis2.trials[0].last_result.get("metric") == 7.0


def test_training_iteration_shared_across_worker_threads():
    """Follow-up to #996, reviewer point 2 (second review): training_iteration
    used to be a plain int copied onto _RunContext by get_run_context() and
    discarded when the receiving thread's use_run_context() block exited.
    The driving thread's own copy was never updated by a worker thread's
    report(), so each new propagated handoff restarted counting from
    whatever the driving thread's stale copy held (0 here, since the
    driving thread itself never reports directly) instead of continuing
    the trial's real count. Every one of the three handoffs below would
    report training_iteration=1 on unfixed code, not 0, 1, 2.

    Three SEPARATE worker threads report for the SAME trial, one at a
    time (joined before the next starts, so this is deterministic, not a
    race): each is a brand-new threading.Thread with its own fresh
    _TuneState, so this exercises whether the iteration counter is
    actually shared through the trial, not just accidentally continuous
    because it stayed on one thread.
    """

    def eval_multi_handoff(config):
        ctx = tune.get_run_context()
        for _ in range(3):

            def worker():
                with tune.use_run_context(ctx):
                    tune.report(metric=1.0)

            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=5)
        return None

    analysis = tune.run(
        eval_multi_handoff,
        config={"x": tune.uniform(0, 1)},
        metric="metric",
        mode="min",
        num_samples=1,
        verbose=0,
    )
    last_result = analysis.trials[0].last_result
    assert last_result is not None
    assert last_result.get("training_iteration") == 2, (
        "expected the third propagated report to record training_iteration=2 (0-indexed, "
        f"monotonically increasing across the three handoffs); got {last_result}"
    )


def test_tune_run_spark_setup_failure_restores_state():
    """Follow-up to #996, reviewer point 3 (second review): Spark backend
    initialization (check_spark(), constructing the SparkSession) used to
    run before the try/finally that calls _restore_tune_state(), so a
    failure there (here: forced via check_spark(), the same failure a user
    hits from a bad environment) raised straight out of run() without
    restoring the _state/logger mutations the earlier common-setup section
    had already made, leaking them into whatever this thread does next.

    Patches check_spark() itself rather than relying on PySpark being
    absent: CI installs real pyspark on some legs, where check_spark()
    would otherwise succeed and this test would never exercise the
    failure path at all.
    """
    handlers_before = list(logger.handlers)
    level_before = logger.getEffectiveLevel()
    use_ray_before = tune.tune._state.use_ray
    verbose_before = tune.tune._state.verbose

    with mock.patch(
        "flaml.tune.tune.check_spark",
        return_value=(False, ImportError("simulated: pyspark unavailable")),
    ):
        with pytest.raises(ImportError):
            tune.run(
                lambda config: {"metric": 1.0},
                config={"x": tune.uniform(0, 1)},
                metric="metric",
                mode="min",
                num_samples=1,
                verbose=2,
                use_spark=True,
            )

    assert logger.handlers == handlers_before, f"logger.handlers leaked past the spark setup failure: {logger.handlers}"
    assert logger.getEffectiveLevel() == level_before, "logger level leaked past the spark setup failure"
    assert tune.tune._state.use_ray == use_ray_before, "_state.use_ray leaked past the spark setup failure"
    assert tune.tune._state.verbose == verbose_before, "_state.verbose leaked past the spark setup failure"

    # State genuinely was not left corrupted: an ordinary run right after
    # still works, rather than inheriting whatever the failed setup left.
    analysis = tune.run(
        lambda config: {"metric": 1.0},
        config={"x": tune.uniform(0, 1)},
        metric="metric",
        mode="min",
        num_samples=1,
        verbose=0,
    )
    assert len(analysis.trials) == 1
    assert analysis.trials[0].last_result.get("metric") == 1.0


def test_tune_report_from_prewarmed_threadpool_executor_worker():
    """Follow-up to #996, third review point 1: a ThreadPoolExecutor's
    worker threads call Thread.start() once, when the pool spins them up,
    not once per submitted task. _install_thread_context_propagation()
    captures the ambient _RunContext at start() time, so a worker warmed up
    BEFORE any tune.run() call exists would previously capture nothing, and
    every later task submitted to that same (reused) worker would silently
    lose report() the same way an un-patched plain Thread used to.

    The pool is created and its worker warmed up (submit + result()) before
    tune.run() is ever called, so this cannot pass by accident just because
    the worker happened to start during a run.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(lambda: None).result()  # warm up the one worker thread

    def eval_via_prewarmed_pool(config):
        future = executor.submit(tune.report, metric=99.0)
        future.result(timeout=5)
        return None  # the pool worker's report() is the only result

    try:
        analysis = tune.run(
            eval_via_prewarmed_pool,
            config={"x": tune.uniform(0, 1)},
            metric="metric",
            mode="min",
            num_samples=1,
            verbose=0,
        )
    finally:
        executor.shutdown(wait=True)

    assert analysis.trials[0].last_result is not None, (
        "report() submitted to an already-warmed-up ThreadPoolExecutor worker was dropped; "
        "expected _install_executor_context_propagation() to attach the context per task"
    )
    assert analysis.trials[0].last_result.get("metric") == 99.0


def test_tune_report_from_prewarmed_executor_reused_across_trials():
    """Same mechanism as above, exercised across TWO trials sharing the SAME
    pre-warmed worker thread (the "cross-trial executor reuse" half of
    third review point 1): each trial's task must land on ITS OWN trial,
    not get pinned to whichever trial warmed the worker up first.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(lambda: None).result()

    def eval_via_prewarmed_pool(config):
        future = executor.submit(tune.report, metric=config["x"])
        future.result(timeout=5)
        return None

    try:
        analysis = tune.run(
            eval_via_prewarmed_pool,
            config={"x": tune.uniform(0, 1)},
            points_to_evaluate=[{"x": 11.0}, {"x": 22.0}],
            metric="metric",
            mode="min",
            num_samples=2,
            verbose=0,
        )
    finally:
        executor.shutdown(wait=True)

    reported = sorted(t.last_result.get("metric") for t in analysis.trials if t.last_result is not None)
    assert reported == [
        11.0,
        22.0,
    ], f"expected each of the two trials to land its own report() through the reused pool worker, got {reported}"


def test_tune_report_from_thread_subclass_overriding_run():
    """Follow-up to #996, third review point 2: the first version of this
    patch replaced threading.Thread.run directly, which a Thread SUBCLASS
    overriding its own run() (idiomatic, common) shadows, so the patch's
    context-attaching code never executed for it. The fix wraps
    Thread._bootstrap_inner instead, which CPython calls internally and
    which invokes self.run() regardless of which run() that resolves to.
    """

    class ReportingWorker(threading.Thread):
        def run(self):
            tune.report(metric=123.0)

    def eval_subclassed_thread(config):
        t = ReportingWorker()
        t.start()
        t.join(timeout=5)
        return None

    analysis = tune.run(
        eval_subclassed_thread,
        config={"x": tune.uniform(0, 1)},
        metric="metric",
        mode="min",
        num_samples=1,
        verbose=0,
    )
    assert analysis.trials[0].last_result is not None, (
        "report() from a Thread SUBCLASS overriding run() was dropped; expected "
        "_install_thread_context_propagation() to wrap _bootstrap_inner, not run(), "
        "so an overridden run() is still covered"
    )
    assert analysis.trials[0].last_result.get("metric") == 123.0


def test_tune_log_records_from_worker_thread_reach_run_log(tmp_path):
    """Follow-up to #996, third review point 4: automatic context
    propagation (a plain worker Thread, or a ThreadPoolExecutor task)
    carried report()'s runner/trial routing, but not `_state.log_run_id`,
    which `_RunScopedFilter` matches against to decide whether a log record
    belongs in THIS run's log file. A worker thread's own _TuneState starts
    with log_run_id=None, so its log records were silently filtered out of
    the run log even though its report() call correctly reached the right
    trial.
    """
    log_path = str(tmp_path / "worker_thread.log")

    def eval_logging_worker(config):
        def worker():
            logger.info("MARKER_FROM_WORKER_THREAD")

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)
        return {"metric": 1.0}

    tune.run(
        eval_logging_worker,
        config={"x": tune.uniform(0, 1)},
        metric="metric",
        mode="min",
        num_samples=1,
        verbose=2,
        log_file_name=log_path,
    )

    text = open(log_path).read()
    assert "MARKER_FROM_WORKER_THREAD" in text, (
        "a worker thread's log record was filtered out of its own run's log file; "
        f"expected log_run_id propagation to carry it through, got: {text!r}"
    )


def test_late_report_via_stale_context_does_not_corrupt_finished_trial():
    """Follow-up to #996, fourth review point 2: a _RunContext captured for
    a trial stays usable after that trial finishes. A worker thread that
    captured get_run_context() and reports late, after tune.run() has
    already returned and the trial is TERMINATED, used to still write
    through: process_trial_result() overwrote the trial's already-final
    metric_analysis/last_result with the late value, and report()'s own
    trailing `if trial.is_finished(): raise StopIteration` (the normal
    scheduler-stop signal for the CURRENT report) then raised into the late
    caller too, which has no reason to expect it the way a trainable's own
    control-flow loop does.
    """
    captured = {}

    def eval_capturing(config):
        captured["ctx"] = tune.get_run_context()
        return {"metric": 1.0}

    analysis = tune.run(
        eval_capturing,
        config={"x": tune.uniform(0, 1)},
        metric="metric",
        mode="min",
        num_samples=1,
        verbose=0,
    )
    trial = analysis.trials[0]
    assert trial.is_finished(), "expected the trial to be TERMINATED once tune.run() returns"
    last_result_before = dict(trial.last_result)
    metric_analysis_before = {k: dict(v) for k, v in trial.metric_analysis.items()}

    with tune.use_run_context(captured["ctx"]):
        tune.report(metric=999.0)

    assert (
        trial.last_result == last_result_before
    ), f"a late report corrupted the finished trial's last_result: {trial.last_result}"
    assert (
        trial.metric_analysis == metric_analysis_before
    ), f"a late report corrupted the finished trial's metric_analysis: {trial.metric_analysis}"


def test_tune_log_records_from_executor_worker_reach_run_log(tmp_path):
    """Same as above (third review point 4), for a task submitted to a
    ThreadPoolExecutor rather than a plain Thread, since the two are
    separate propagation paths (_install_thread_context_propagation() vs
    _install_executor_context_propagation()).
    """
    log_path = str(tmp_path / "executor_worker.log")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(lambda: None).result()  # warm up before any run exists

    def emit():
        logger.info("MARKER_FROM_EXECUTOR_WORKER")

    def eval_logging_executor(config):
        future = executor.submit(emit)
        future.result(timeout=5)
        return {"metric": 1.0}

    try:
        tune.run(
            eval_logging_executor,
            config={"x": tune.uniform(0, 1)},
            metric="metric",
            mode="min",
            num_samples=1,
            verbose=2,
            log_file_name=log_path,
        )
    finally:
        executor.shutdown(wait=True)

    text = open(log_path).read()
    assert "MARKER_FROM_EXECUTOR_WORKER" in text, (
        "an executor worker's log record was filtered out of its own run's log file; "
        f"expected log_run_id propagation to carry it through, got: {text!r}"
    )


def test_concurrent_reports_for_same_trial_admit_atomically():
    """Follow-up to #996, fifth review point 2: trial.is_finished() and
    runner.process_trial_result() were not atomic. Two worker threads
    reporting for the SAME trial through a shared propagated context (the
    same handoff every test above uses, just two of them at once instead of
    one) could both pass the is_finished() check while the trial was still
    running; whichever one's process_trial_result() call landed AFTER the
    other one's scheduler decision had already finished the trial still
    wrote through, silently replacing the trial's real final result with a
    stale one.

    Forced deterministically, with events rather than a sleep: worker B is
    paused right after its own is_finished() check returns False, before it
    enters the critical section, until worker A's entire report() call,
    including the scheduler decision that terminates the trial, has
    completed. B is released only then, so it always reaches admission with
    an already-finished trial, which is exactly the window the fix closes
    with a second, lock-protected check immediately before
    process_trial_result().

    Gate point: `_admission_lock_for(trial)`, patched to pause AFTER
    fetching the real per-trial lock but BEFORE returning it, i.e. before
    the `with` statement that follows ever calls `.acquire()` on it, so B
    is paused without holding the lock. This used to gate on
    `_next_training_iteration` instead (also called between the fast check
    and the `with` block, at the time); #996 follow-up sixth review point 3
    moved that call INSIDE the locked section (allocation and admission
    must be one atomic step, not two), so gating there now would mean B
    pauses WHILE HOLDING the lock A's own report() needs, deadlocking both
    threads instead of testing anything.
    """

    class StopOnFirstResult:
        """A minimal TrialScheduler, the same pluggable interface
        SequentialTrialRunner feeds any real scheduler through: STOPs the
        trial the first time a result is admitted, so whichever worker
        reaches admission first legitimately finishes the trial.
        """

        def set_search_properties(self, metric=None, mode=None, **spec):
            pass

        def on_trial_add(self, runner, trial):
            pass

        def on_trial_result(self, runner, trial, result):
            return "STOP"

        def on_trial_complete(self, runner, trial_id, result=None, error=False):
            pass

        def on_trial_remove(self, runner, trial):
            pass

    b_ready = threading.Event()
    a_finished = threading.Event()
    orig_admission_lock_for = tune.tune._admission_lock_for

    def gated_admission_lock_for(trial_obj):
        lock = orig_admission_lock_for(trial_obj)
        if threading.current_thread().name == "B_WORKER":
            b_ready.set()
            assert a_finished.wait(timeout=5), "worker A never finished while B was paused"
        return lock

    def eval_concurrent_reporters(config):
        ctx = tune.get_run_context()

        def worker_a():
            try:
                with tune.use_run_context(ctx):
                    tune.report(metric=1.0)
            except StopIteration:
                pass
            finally:
                a_finished.set()

        def worker_b():
            try:
                with tune.use_run_context(ctx):
                    tune.report(metric=2.0)
            except StopIteration:
                pass

        thread_b = threading.Thread(target=worker_b, name="B_WORKER")
        thread_b.start()
        assert b_ready.wait(timeout=5), "worker B never reached the admission gap"
        thread_a = threading.Thread(target=worker_a, name="A_WORKER")
        thread_a.start()
        thread_a.join(timeout=5)
        thread_b.join(timeout=5)
        return None

    with mock.patch("flaml.tune.tune._admission_lock_for", side_effect=gated_admission_lock_for):
        analysis = tune.run(
            eval_concurrent_reporters,
            config={"x": tune.uniform(0, 1)},
            metric="metric",
            mode="min",
            num_samples=1,
            scheduler=StopOnFirstResult(),
            verbose=0,
        )

    trial = analysis.trials[0]
    assert trial.is_finished()
    assert trial.last_result.get("metric") == 1.0, (
        "worker B's report, paused mid-flight and released only after worker A's report "
        "had already terminated the trial via the scheduler's STOP decision, still "
        f"overwrote the trial's real final result; got {trial.last_result}"
    )


def test_tune_report_from_prestarted_reused_queue_worker_via_explicit_context():
    """Follow-up to #996, fifth review point 1 (restating fourth review
    point 1): a generic persistent queue-consumer thread started BEFORE any
    tune.run() call exists has nothing for _install_thread_context_
    propagation() to capture at its Thread.start() time, and it is not a
    ThreadPoolExecutor either, so _install_executor_context_propagation()
    does not apply.

    Automatic per-dispatch propagation for an arbitrary queue worker is not
    being added: report()'s only handle on which trial a dequeued item
    belongs to is whatever context was captured when it was produced, and
    nothing at report()-call time can reconstruct that after the fact.
    SequentialTrialRunner.step() reassigns runner.running_trial to a new
    trial every step, so a fallback that resolved the trial live at
    report() time would attribute a delayed item to whichever trial happens
    to be running when it is finally dequeued, not the one it was produced
    for, which is a silent wrong-trial write and strictly worse than today.

    The compatibility route the review also names already exists:
    get_run_context()/use_run_context(), captured by the producer at
    enqueue time and carried on the queue item itself instead of resolved
    at dequeue time. This is that route, on exactly the shape described: a
    worker thread started before any tune.run() call exists, reused
    unmodified across two separate trials.
    """
    work_queue = queue.Queue()
    stop = object()

    def worker():
        while True:
            item = work_queue.get()
            try:
                if item is stop:
                    return
                ctx, value = item
                with tune.use_run_context(ctx):
                    tune.report(metric=value)
            finally:
                work_queue.task_done()

    worker_thread = threading.Thread(target=worker)
    worker_thread.start()  # started before any tune.run() call exists

    def eval_via_prestarted_queue(config):
        ctx = tune.get_run_context()
        work_queue.put((ctx, config["x"]))
        work_queue.join()  # wait for the pre-started worker to drain this trial's item
        return None

    try:
        analysis = tune.run(
            eval_via_prestarted_queue,
            config={"x": tune.uniform(0, 1)},
            points_to_evaluate=[{"x": 11.0}, {"x": 22.0}],
            metric="metric",
            mode="min",
            num_samples=2,
            verbose=0,
        )
    finally:
        work_queue.put(stop)
        worker_thread.join(timeout=5)

    reported = sorted(t.last_result.get("metric") for t in analysis.trials if t.last_result is not None)
    assert reported == [
        11.0,
        22.0,
    ], (
        "a worker thread started before tune.run() and reused across both trials, using "
        "the documented get_run_context()/use_run_context() API, did not route each "
        f"trial's report to its own trial; got {reported}"
    )


def test_stop_trial_shares_lifecycle_lock_with_late_report():
    """Follow-up to #996, sixth review point 1: report()'s admission
    (process_trial_result(), guarded by tune.py's per-trial
    _admission_lock_for) and trial_runner.py's stop_trial() did not share a
    lock, even though both mutate the same trial: last_result and
    metric_analysis (via Trial.update_last_result()), status, and the
    search_alg/scheduler on_trial_* callbacks. A trainable can report from
    a background thread it does not wait for (every worker-thread test
    above uses exactly this shape; here evaluation_function() just does not
    join() it before returning), and run()'s own sequential loop calls
    runner.stop_trial(trial_to_run) immediately after evaluation_function()
    returns, whether or not that straggling report has finished.

    Forced deterministically, with events rather than a sleep:
    Trial.update_last_result() (called from inside process_trial_result(),
    itself inside the admission lock) is patched to signal it has been
    entered and then block. Once that signal arrives, stop_trial() is
    called directly from a separate thread, not through run()'s own loop:
    doing it through the loop would block the driving thread on the very
    lock this test needs to release the worker to unblock, deadlocking the
    test itself rather than exercising the race. Before the fix,
    stop_trial() had nothing to wait on: it read trial.last_result while
    the straggling report's update_last_result() was still paused mid-write
    (last_result still the trial's pre-report default), and handed that
    stale value to the search algorithm's on_trial_complete() as the
    trial's supposedly final result. After the fix, stop_trial() blocks on
    the same per-trial lock until the report's entire critical section has
    completed, so it always sees the trial's real last_result.
    """
    worker_in_update = threading.Event()
    release_worker = threading.Event()
    orig_update_last_result = Trial.update_last_result

    def gated_update_last_result(self, result):
        worker_in_update.set()
        assert release_worker.wait(timeout=5), "test never released the in-flight report"
        return orig_update_last_result(self, result)

    class RecordingSearchAlg:
        """Minimal search_alg double: suggests exactly one trial, records
        every result stop_trial() hands its on_trial_complete().
        """

        def __init__(self):
            self._suggested = False
            self.complete_calls = []

        def set_search_properties(self, metric=None, mode=None, config=None, **spec):
            return True

        def suggest(self, trial_id):
            if self._suggested:
                return None
            self._suggested = True
            return {"x": 0.5}

        def on_trial_result(self, trial_id, result):
            pass

        def on_trial_complete(self, trial_id, result=None, error=False):
            self.complete_calls.append(dict(result) if result else result)

    search_alg = RecordingSearchAlg()
    worker_errors = []

    def eval_fire_and_forget(config):
        ctx = tune.get_run_context()
        trial = ctx.running_trial
        runner = ctx.runner

        def worker():
            try:
                with tune.use_run_context(ctx):
                    tune.report(metric=1.0)
            except StopIteration:
                pass
            except Exception as exc:  # pragma: no cover - failure diagnostics only
                worker_errors.append(exc)

        report_thread = threading.Thread(target=worker, name="STRAGGLER")
        report_thread.start()
        assert worker_in_update.wait(timeout=5), "worker never reached update_last_result"

        stop_started = threading.Event()

        def call_stop_trial():
            stop_started.set()
            runner.stop_trial(trial)

        stop_thread = threading.Thread(target=call_stop_trial, name="STOPPER")
        stop_thread.start()
        assert stop_started.wait(timeout=5), "stop_trial() thread never started"
        release_worker.set()
        report_thread.join(timeout=5)
        stop_thread.join(timeout=5)
        return None  # run()'s own post-eval stop_trial() call becomes a harmless no-op

    with mock.patch.object(Trial, "update_last_result", gated_update_last_result):
        analysis = tune.run(
            eval_fire_and_forget,
            config={"x": tune.uniform(0, 1)},
            metric="metric",
            mode="min",
            num_samples=1,
            search_alg=search_alg,
            verbose=0,
        )

    assert not worker_errors, f"straggling report thread raised: {worker_errors}"
    trial = analysis.trials[0]
    assert trial.last_result is not None and trial.last_result.get("metric") == 1.0
    assert search_alg.complete_calls, "stop_trial() never called on_trial_complete"
    assert search_alg.complete_calls[0] is not None and search_alg.complete_calls[0].get("metric") == 1.0, (
        "stop_trial() handed the search algorithm a stale/incomplete last_result while the "
        f"straggling report's update_last_result() was still in flight; got {search_alg.complete_calls[0]}"
    )


def test_training_iteration_allocated_inside_admission_lock():
    """Follow-up to #996, sixth review point 3: training_iteration used to
    be allocated (_next_training_iteration()) BEFORE the per-trial
    admission lock (_admission_lock_for) was acquired, not inside it. Two
    truly concurrent reports for the same trial could then be handed
    iteration numbers in one order and reach runner.process_trial_result(),
    which is what a scheduler/searcher that orders trials by
    training_iteration (ASHA and similar) actually sees, in the OTHER
    order, if the thread that allocated the LATER number happened to reach
    the lock first.

    Rather than trying to force that exact reordering (which the fix makes
    impossible to construct at all, since allocation now only happens while
    already holding the lock), this proves the mechanism directly: a
    report's iteration allocation is patched to pause mid-call, and a
    second, independent attempt to enter the SAME trial's admission section
    is made concurrently. If allocation and admission share one critical
    section, that second attempt must block for as long as allocation is
    paused; if they are two separate critical sections (the bug), the
    second attempt sails through immediately, since nothing is held during
    allocation.
    """
    allocating = threading.Event()
    release_allocation = threading.Event()
    orig_next_iteration = tune.tune._next_training_iteration

    def gated_next_iteration(trial_obj):
        allocating.set()
        assert release_allocation.wait(timeout=5), "test never released the paused allocation"
        return orig_next_iteration(trial_obj)

    def eval_probe(config):
        ctx = tune.get_run_context()
        trial = ctx.running_trial

        def worker():
            with tune.use_run_context(ctx):
                tune.report(metric=1.0)

        report_thread = threading.Thread(target=worker, name="ALLOCATOR")
        report_thread.start()
        assert allocating.wait(timeout=5), "worker never reached iteration allocation"

        probe_acquired = threading.Event()

        def probe():
            with tune.tune._admission_lock_for(trial):
                probe_acquired.set()

        probe_thread = threading.Thread(target=probe, name="PROBE")
        probe_thread.start()
        acquired_while_allocation_paused = probe_acquired.wait(timeout=0.5)

        release_allocation.set()
        report_thread.join(timeout=5)
        probe_thread.join(timeout=5)

        assert not acquired_while_allocation_paused, (
            "a second, independent attempt to admit a report for the same trial acquired "
            "the admission lock while training_iteration allocation for an in-flight report "
            "was still paused: allocation is not happening inside the same critical section "
            "as process_trial_result(), so a concurrent report can be allocated a later "
            "iteration and still be admitted first"
        )
        return None

    with mock.patch("flaml.tune.tune._next_training_iteration", side_effect=gated_next_iteration):
        tune.run(
            eval_probe,
            config={"x": tune.uniform(0, 1)},
            metric="metric",
            mode="min",
            num_samples=1,
            verbose=0,
        )


def test_logger_level_restored_as_inherited_not_explicit():
    """Follow-up to #996, sixth review point 4: _logger_level_enter() saved
    logger.getEffectiveLevel(), the RESOLVED level after walking up the
    logger hierarchy when this logger has no level of its own (logger.level
    == logging.NOTSET), and _logger_level_exit() restored that resolved
    number via logger.setLevel(). That gives the logger a permanent
    explicit level it never had before: a logger that was inheriting must
    go back to inheriting once every active run has exited, not end up
    pinned to whatever the ancestor chain resolved to during the run.

    Not a contrived setup: flaml/__init__.py sets the "flaml" logger to
    INFO at import time, so flaml.tune.logger's own getEffectiveLevel()
    already resolves to INFO (via that ancestor) in any unconfigured
    process. No level manipulation beyond resetting flaml.tune.logger's own
    level to NOTSET is needed to construct "this logger is inheriting", the
    precondition the fix is about; the assertion just below confirms it.
    """
    saved_level = logger.level
    try:
        logger.setLevel(logging.NOTSET)  # construct the precondition: inheriting, not explicit
        assert logger.getEffectiveLevel() != logging.NOTSET, (
            "test assumes an ancestor logger (flaml/__init__.py sets 'flaml' to INFO) resolves "
            "to a non-NOTSET effective level here"
        )

        tune.run(
            lambda config: {"metric": 1.0},
            config={"x": tune.uniform(0, 1)},
            metric="metric",
            mode="min",
            num_samples=1,
            verbose=1,
        )

        assert logger.level == logging.NOTSET, (
            "tune.run() left flaml.tune's logger pinned to an explicit level instead of "
            f"restoring it to NOTSET (inheriting); got {logging.getLevelName(logger.level)}"
        )
    finally:
        logger.setLevel(saved_level)


def test_context_less_report_from_unchanged_legacy_queue_worker_warns_without_corrupting(tmp_path):
    """Follow-up to #996, seventh review: a generic queue/callback worker
    started before any tune.run() call exists, that never calls
    get_run_context()/use_run_context() itself (an unchanged legacy
    caller), still has its tune.report() silently dropped: automatic
    attribution is not being added, for the same reason given in the
    fourth, fifth and sixth review rounds (SequentialTrialRunner.step()
    reassigns running_trial every step, so a live-resolved fallback would
    attribute a delayed report to whichever trial happens to be running by
    the time it is handled, not the one it was produced for).

    What changed this round: the drop is no longer silent. It logs once,
    and the message survives even while a verbose run is active with its
    own log-file handler attached, which is the exact case
    _RunScopedFilter would otherwise swallow it in (a handler whose filter
    rejects a record still counts toward Python's `found` handler count,
    so `logging.lastResort` never fires either; verified directly while
    building this fix).
    """
    work_queue = queue.Queue()
    stop = object()

    def worker():
        while True:
            item = work_queue.get()
            try:
                if item is stop:
                    return
                # unchanged legacy caller: no get_run_context()/use_run_context(),
                # exactly the shape the review describes.
                tune.report(metric=item)
            finally:
                work_queue.task_done()

    def eval_via_legacy_queue(config):
        work_queue.put(config["x"])
        work_queue.join()  # wait for the pre-started worker to (fail to) report it
        return None

    log_path = str(tmp_path / "legacy_worker.log")
    worker_thread = threading.Thread(target=worker)
    try:
        worker_thread.start()  # started before any tune.run() call exists
        with mock.patch.object(tune_module, "_context_less_report_warned", False):
            analysis = tune.run(
                eval_via_legacy_queue,
                config={"x": tune.uniform(0, 1)},
                points_to_evaluate=[{"x": 5.0}],
                metric="metric",
                mode="min",
                num_samples=1,
                verbose=1,
                log_file_name=log_path,
            )
    finally:
        # Put `stop` and join unconditionally, including if the mock.patch
        # setup itself raised (as it does against a tune.py that predates
        # this fix, which has no `_context_less_report_warned` attribute to
        # patch): the worker thread is a plain non-daemon Thread blocked on
        # queue.get() with nothing else able to release it, and leaving it
        # running hangs the whole interpreter at process exit.
        work_queue.put(stop)
        worker_thread.join(timeout=5)

    assert analysis.trials[0].last_result is None, (
        "the context-less report should have been dropped, leaving the trial's result "
        f"untouched (None), not corrupted with a value; got {analysis.trials[0].last_result!r}"
    )

    log_text = open(log_path).read()
    assert (
        "no active run to attribute it to" in log_text
    ), f"expected the context-less-report warning in the run's own log file, got: {log_text!r}"
    assert (
        log_text.count("no active run to attribute it to") == 1
    ), "the warning should fire once per process, not once per dropped report"
