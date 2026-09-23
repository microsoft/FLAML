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
import threading
from unittest import mock

import pytest

from flaml import tune
from flaml.tune.logger import logger


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
