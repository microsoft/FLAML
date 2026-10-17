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


@pytest.fixture
def caller_alarm(monkeypatch):
    def install(handler, delay=0.02):
        # Supply the entry snapshot without a live timer that can fire during test setup.
        # All timers armed by the context and subsequent reads use the real OS timer.
        signal.signal(signal.SIGALRM, handler)
        getitimer = signal.getitimer
        pending = True

        def initial_timer(timer):
            nonlocal pending
            if pending:
                pending = False
                return delay, 0
            return getitimer(timer)

        monkeypatch.setattr(signal, "getitimer", initial_timer)

    return install


@pytest.mark.parametrize("exception_type", [TimeoutError, MemoryError])
def test_fit_propagates_caller_exception(monkeypatch, exception_type, caller_alarm):
    error = exception_type("caller deadline")

    def handler(sig, frame):
        raise error

    caller_alarm(handler)
    model = LRL2Classifier()
    monkeypatch.setattr(model, "_fit", lambda *args, **kwargs: time.sleep(1))
    with pytest.raises(exception_type) as caught:
        model.fit(np.array([[0], [1]]), np.array([0, 1]), budget=1)
    assert caught.value is error
    assert caught.traceback[-1].name == "handler"
    assert model.model is None
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_fit_propagates_outer_budget_timeout(monkeypatch):
    model = LRL2Classifier()
    monkeypatch.setattr(model, "_fit", lambda *args, **kwargs: time.sleep(2))
    with pytest.raises(TimeoutError):
        with limit_resource(-1, 1):
            model.fit(np.array([[0], [1]]), np.array([0, 1]), budget=30)
    assert model.model is None


def test_fit_still_handles_own_budget_timeout(monkeypatch):
    from sklearn.dummy import DummyClassifier

    model = LRL2Classifier()
    monkeypatch.setattr(model, "_fit", lambda *args, **kwargs: time.sleep(2))
    with limit_resource(-1, 30):
        model.fit(np.array([[0], [1]]), np.array([0, 1]), budget=1)
        assert isinstance(model.model, DummyClassifier)
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0


@pytest.mark.parametrize("replacement", ["SIG_IGN", "SIG_DFL"])
def test_changed_caller_handler_does_not_disable_budget(replacement):
    import subprocess
    import sys

    # Isolate SIG_DFL: a regression must fail the test, not terminate pytest.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import signal
import time
from flaml.automl.model import limit_resource
replacement = signal.{replacement}
def handler(sig, frame):
    signal.signal(signal.SIGALRM, replacement)
signal.signal(signal.SIGALRM, handler)
getitimer = signal.getitimer
pending = True
def initial_timer(timer):
    global pending
    if pending:
        pending = False
        return 0.02, 0
    return getitimer(timer)
signal.getitimer = initial_timer
try:
    with limit_resource(-1, 1):
        time.sleep(2)
except TimeoutError:
    assert signal.getsignal(signal.SIGALRM) == replacement
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
else:
    raise AssertionError("training deadline was ignored")
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_nested_limits_preserve_changed_caller_handler(caller_alarm):
    calls = []

    def replacement(sig, frame):
        calls.append("replacement")

    def handler(sig, frame):
        calls.append("original")
        signal.signal(signal.SIGALRM, replacement)
        signal.setitimer(signal.ITIMER_REAL, 0.02)

    caller_alarm(handler)
    with limit_resource(-1, 30):
        with limit_resource(-1, 10):
            wait_deadline = time.monotonic() + 2
            while len(calls) < 2 and time.monotonic() < wait_deadline:
                time.sleep(0.01)
    assert calls == ["original", "replacement"]
    assert signal.getsignal(signal.SIGALRM) is replacement


def test_delayed_repeating_alarm_keeps_phase(monkeypatch):
    from types import SimpleNamespace

    # A fake clock makes the phase check independent of scheduler delays in CI.
    calls = []
    state = SimpleNamespace(now=0.0, deadline=0.04, interval=0.2, handler=lambda *args: calls.append(state.now))

    def getitimer(timer):
        return (max(state.deadline - state.now, 0) if state.deadline else 0, state.interval)

    def setitimer(timer, delay, interval=0):
        state.deadline = state.now + delay if delay else None
        state.interval = interval

    def set_handler(sig, handler):
        state.handler = handler

    # Keep the real signal module intact for the alarm-restoration fixture.
    monkeypatch.setattr(
        "flaml.automl.model.signal",
        SimpleNamespace(
            SIGALRM=signal.SIGALRM,
            SIG_DFL=signal.SIG_DFL,
            ITIMER_REAL=signal.ITIMER_REAL,
            getsignal=lambda sig: state.handler,
            signal=set_handler,
            getitimer=getitimer,
            setitimer=setitimer,
        ),
    )
    monkeypatch.setattr("flaml.automl.model.time", SimpleNamespace(monotonic=lambda: state.now))
    with limit_resource(-1, 2):
        state.now = 0.28
        state.handler(signal.SIGALRM, None)
        assert calls == [0.28]
        assert state.deadline == pytest.approx(0.44)
        state.now = 0.44
        state.handler(signal.SIGALRM, None)
        assert calls == [0.28, 0.44]
        assert state.deadline == pytest.approx(0.64)
    assert state.interval == 0.2


def test_early_signal_does_not_expire_budget():
    with limit_resource(-1, 30):
        signal.raise_signal(signal.SIGALRM)
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 29


def test_caught_caller_exception_does_not_disable_budget(caller_alarm):
    error = RuntimeError("caller deadline")

    def handler(sig, frame):
        raise error

    caller_alarm(handler)
    with pytest.raises(TimeoutError):
        with limit_resource(-1, 1):
            try:
                time.sleep(0.5)
            except RuntimeError as caught:
                assert caught is error
            assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
            time.sleep(1.2)


@pytest.mark.parametrize("exception_type", [TimeoutError, MemoryError, RuntimeError])
def test_expired_budget_takes_precedence_over_slow_caller(monkeypatch, exception_type, caller_alarm):
    from sklearn.dummy import DummyClassifier

    def handler(sig, frame):
        time.sleep(1.05)
        raise exception_type("late caller exception")

    caller_alarm(handler)
    model = LRL2Classifier()
    monkeypatch.setattr(model, "_fit", lambda *args, **kwargs: time.sleep(2))
    model.fit(np.array([[0], [1]]), np.array([0, 1]), budget=1)
    assert isinstance(model.model, DummyClassifier)


def test_caught_late_caller_cannot_disable_expired_budget(caller_alarm):
    def handler(sig, frame):
        time.sleep(1.05)
        raise RuntimeError("late caller exception")

    caller_alarm(handler)
    with pytest.raises(TimeoutError):
        with limit_resource(-1, 1):
            try:
                time.sleep(2)
            except RuntimeError:
                pass
            time.sleep(0.2)


def test_overlapping_outer_budget_keeps_its_owner(monkeypatch, caller_alarm):
    def handler(sig, frame):
        time.sleep(1.05)
        raise RuntimeError("late caller exception")

    caller_alarm(handler)
    model = LRL2Classifier()
    monkeypatch.setattr(model, "_fit", lambda *args, **kwargs: time.sleep(2))
    with pytest.raises(TimeoutError):
        with limit_resource(-1, 1):
            model.fit(np.array([[0], [1]]), np.array([0, 1]), budget=1)
    assert model.model is None
