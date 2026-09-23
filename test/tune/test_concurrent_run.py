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

import threading

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
    failure there (here: PySpark not installed, the same failure a user
    hits from a bad environment) raised straight out of run() without
    restoring the _state/logger mutations the earlier common-setup section
    had already made, leaking them into whatever this thread does next.

    PySpark is not installed in this test environment, so check_spark()
    deterministically returns unavailable; no real Spark cluster needed.
    """
    handlers_before = list(logger.handlers)
    level_before = logger.getEffectiveLevel()
    use_ray_before = tune.tune._state.use_ray
    verbose_before = tune.tune._state.verbose

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
