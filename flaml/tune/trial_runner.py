# !
#  * Copyright (c) Microsoft Corporation. All rights reserved.
#  * Licensed under the MIT License. See LICENSE file in the
#  * project root for license information.
import logging
from typing import Optional

# try:
#     from ray import __version__ as ray_version
#     assert ray_version >= '1.0.0'
#     from ray.tune.trial import Trial
# except (ImportError, AssertionError):
from .trial import Trial

logger = logging.getLogger(__name__)


class Nologger:
    """Logger without logging."""

    def on_result(self, result):
        pass


class SimpleTrial(Trial):
    """A simple trial class."""

    def __init__(self, config, trial_id=None):
        self.trial_id = Trial.generate_id() if trial_id is None else trial_id
        self.config = config or {}
        self.status = Trial.PENDING
        self.start_time = None
        self.last_result = None
        self.last_update_time = -float("inf")
        self.custom_trial_name = None
        self.trainable_name = "trainable"
        self.experiment_tag = "exp"
        self.verbose = False
        self.result_logger = Nologger()
        self.metric_analysis = {}
        self.n_steps = [5, 10]
        self.metric_n_steps = {}
        self.metric_n_reports = {}


class BaseTrialRunner:
    """Implementation of a simple trial runner.

    Note that the caller usually should not mutate trial state directly.
    """

    def __init__(
        self,
        search_alg=None,
        scheduler=None,
        metric: Optional[str] = None,
        mode: Optional[str] = "min",
    ):
        self._search_alg = search_alg
        self._scheduler_alg = scheduler
        self._trials = []
        self._metric = metric
        self._mode = mode

    def get_trials(self):
        """Returns the list of trials managed by this TrialRunner.

        Note that the caller usually should not mutate trial state directly.
        """
        return self._trials

    def add_trial(self, trial):
        """Adds a new trial to this TrialRunner.

        Trials may be added at any time.

        Args:
            trial (Trial): Trial to queue.
        """
        self._trials.append(trial)
        if self._scheduler_alg:
            self._scheduler_alg.on_trial_add(self, trial)

    def process_trial_result(self, trial, result):
        trial.update_last_result(result)
        if "time_total_s" not in result.keys():
            result["time_total_s"] = trial.last_update_time - trial.start_time
        self._search_alg.on_trial_result(trial.trial_id, result)
        if self._scheduler_alg:
            decision = self._scheduler_alg.on_trial_result(self, trial, result)
            if decision == "STOP":
                trial.set_status(Trial.TERMINATED)
            elif decision == "PAUSE":
                trial.set_status(Trial.PAUSED)

    def stop_trial(self, trial):
        """Stops trial.

        Holds the same per-trial lock tune.py's report() takes before its
        own admission (tune.py's `_admission_lock_for`), so this cannot run
        at the same instant a report is inside its locked
        process_trial_result() section for the SAME trial (#996 follow-up,
        sixth review point 1). Before this, a straggling background
        thread's report (see the propagation machinery in tune.py: a
        trainable can hand a worker thread a run context and keep reporting
        through it after evaluation_function() has already returned) could
        be mutating `trial.last_result`/`trial.status` and calling into the
        search_alg/scheduler for this trial via process_trial_result() at
        the same time run()'s own loop called stop_trial() on it directly,
        unsynchronized: the two paths shared a trial but not a lock. Local
        import: trial_runner is only ever imported lazily from inside
        tune.run() (or after `flaml.tune`'s own package init has already
        finished), specifically so tune.py has no load-time dependency on
        this module; importing back here only inside the call keeps that
        one-directional at import time while both places key the same lock
        off the same trial object.
        """
        from .tune import _admission_lock_for

        with _admission_lock_for(trial):
            if trial.status not in [Trial.ERROR, Trial.TERMINATED]:
                if self._scheduler_alg:
                    self._scheduler_alg.on_trial_complete(self, trial.trial_id, trial.last_result)
                self._search_alg.on_trial_complete(trial.trial_id, trial.last_result)
                trial.set_status(Trial.TERMINATED)
            elif self._scheduler_alg:
                self._scheduler_alg.on_trial_remove(self, trial)
                if trial.status == Trial.ERROR:
                    self._search_alg.on_trial_complete(trial.trial_id, trial.last_result, error=True)


class SequentialTrialRunner(BaseTrialRunner):
    """Implementation of the sequential trial runner."""

    def step(self) -> Trial:
        """Runs one step of the trial event loop.

        Callers should typically run this method repeatedly in a loop. They
        may inspect or modify the runner's state in between calls to step().

        Returns:
            a trial to run.
        """
        trial_id = Trial.generate_id()
        config = self._search_alg.suggest(trial_id)
        if config is not None:
            trial = SimpleTrial(config, trial_id)
            self.add_trial(trial)
            trial.set_status(Trial.RUNNING)
        else:
            trial = None
        self.running_trial = trial
        return trial

    def stop_trial(self, trial):
        super().stop_trial(trial)
        self.running_trial = None


class SparkTrialRunner(BaseTrialRunner):
    """Implementation of the spark trial runner."""

    def __init__(
        self,
        search_alg=None,
        scheduler=None,
        metric: Optional[str] = None,
        mode: Optional[str] = "min",
    ):
        super().__init__(search_alg, scheduler, metric, mode)
        self.running_trials = []

    def step(self) -> Trial:
        """Runs one step of the trial event loop.

        Callers should typically run this method repeatedly in a loop. They
        may inspect or modify the runner's state in between calls to step().

        Returns:
            a trial to run.
        """
        trial_id = Trial.generate_id()
        config = self._search_alg.suggest(trial_id)
        if config is not None:
            trial = SimpleTrial(config, trial_id)
            self.add_trial(trial)
            trial.set_status(Trial.RUNNING)
            self.running_trials.append(trial)
        else:
            trial = None
        return trial

    def stop_trial(self, trial):
        super().stop_trial(trial)
        self.running_trials.remove(trial)
