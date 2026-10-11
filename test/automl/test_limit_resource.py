import signal
import threading
import time

import numpy as np
import pytest

from flaml.automl.model import LRL2Classifier, TimeoutHandler, limit_resource

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="requires POSIX interval timers")


@pytest.fixture(autouse=True)
def restore_alarm():
    handler = signal.getsignal(signal.SIGALRM)
    timer = signal.setitimer(signal.ITIMER_REAL, 0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, *timer)


@pytest.mark.parametrize("raises", [False, True])
def test_restore_caller_alarm(raises):
    def handler(sig, frame):
        pass

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 60, 5)
    try:
        with limit_resource(-1, 30):
            time.sleep(0.02)
            if raises:
                raise RuntimeError("training failed")
    except RuntimeError:
        assert raises
    assert signal.getsignal(signal.SIGALRM) is handler
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    assert 0 < remaining < 59.99
    assert interval == 5


def test_fit_restores_caller_alarm():
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    signal.setitimer(signal.ITIMER_REAL, 60)
    model = LRL2Classifier()
    model.fit(np.array([[0], [1], [2], [3]]), np.array([0, 0, 1, 1]), budget=30)
    assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN
    assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 60


def test_nested_limit_restores_outer_alarm():
    with limit_resource(-1, 30):
        handler = signal.getsignal(signal.SIGALRM)
        with limit_resource(-1, 60):
            time.sleep(0.02)
        assert signal.getsignal(signal.SIGALRM) is handler
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] < 29.99
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_earlier_caller_deadline_is_not_delayed():
    signal.signal(signal.SIGALRM, TimeoutHandler)
    signal.setitimer(signal.ITIMER_REAL, 0.02)
    with pytest.raises(TimeoutError):
        with limit_resource(-1, 30):
            time.sleep(1)
    assert signal.getsignal(signal.SIGALRM) is TimeoutHandler
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_repeating_caller_alarm_remains_active():
    calls = []

    def handler(sig, frame):
        calls.append(sig)

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 0.02, 0.02)
    with limit_resource(-1, 30):
        time.sleep(0.07)
    assert len(calls) >= 2
    assert signal.getsignal(signal.SIGALRM) is handler
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    assert 0 < remaining <= 0.02
    assert interval == 0.02


def test_timeout_restores_handler():
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    with pytest.raises(TimeoutError):
        with limit_resource(-1, 1):
            time.sleep(2)
    assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


@pytest.mark.parametrize("worker_thread", [False, True])
def test_no_alarm_without_budget_or_in_worker(worker_thread):
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    signal.setitimer(signal.ITIMER_REAL, 60, 5)

    def run():
        with limit_resource(-1, 30 if worker_thread else None):
            pass

    if worker_thread:
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
    else:
        run()
    assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    assert 0 < remaining <= 60
    assert interval == 5


def test_nested_limits_preserve_repeating_caller_alarm():
    calls = []

    def handler(sig, frame):
        calls.append(sig)

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 0.02, 0.02)
    with limit_resource(-1, 30):
        with limit_resource(-1, 10):
            time.sleep(0.07)
        time.sleep(0.04)
    assert len(calls) >= 4
    assert signal.getsignal(signal.SIGALRM) is handler
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    assert 0 < remaining <= 0.02
    assert interval == 0.02


def test_nested_limit_honors_outer_timeout():
    with pytest.raises(TimeoutError):
        with limit_resource(-1, 1):
            with limit_resource(-1, 30):
                time.sleep(2)
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_caller_alarm_does_not_cancel_training_timeout():
    calls = []
    signal.signal(signal.SIGALRM, lambda sig, frame: calls.append(sig))
    signal.setitimer(signal.ITIMER_REAL, 0.02)
    with pytest.raises(TimeoutError):
        with limit_resource(-1, 1):
            time.sleep(2)
    assert calls == [signal.SIGALRM]
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_invalid_timeout_does_not_replace_caller_alarm():
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    signal.setitimer(signal.ITIMER_REAL, 60, 5)
    with pytest.raises(OverflowError):
        with limit_resource(-1, -1):
            pytest.fail("negative time limits must be rejected")
    assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    assert 0 < remaining <= 60
    assert interval == 5


def test_restore_alarm_if_memory_cleanup_raises(monkeypatch):
    from types import SimpleNamespace

    def setrlimit(resource, limits):
        if limits == (-1, -1):
            assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
            raise OSError("memory cleanup failed")

    monkeypatch.setattr(
        "flaml.automl.model.resource",
        SimpleNamespace(RLIMIT_AS=0, getrlimit=lambda resource: (-1, -1), setrlimit=setrlimit),
    )
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    signal.setitimer(signal.ITIMER_REAL, 60, 5)
    with pytest.raises(OSError, match="memory cleanup failed"):
        with limit_resource(123, 30):
            pass
    assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    assert 0 < remaining <= 60
    assert interval == 5
