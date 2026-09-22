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

from flaml import tune


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
