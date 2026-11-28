import os
import shutil
import tempfile
import unittest

import numpy as np
import pytest

from flaml import CFO, tune
from flaml.tune.searcher.suggestion import ConcurrencyLimiter


@pytest.mark.parametrize(
    "algorithm,completed,lexicographic",
    [
        ("CFO", False, False),
        ("CFO", True, False),
        ("BlendSearch", False, False),
        ("BlendSearch", True, False),
        ("CFO", True, True),
    ],
)
def test_legacy_checkpoint_restores_reservations_without_resetting_completed_results(
    tmp_path, algorithm, completed, lexicographic
):
    import copy

    from flaml.tune.searcher import blendsearch
    from flaml.tune.searcher.flow2 import FLOW2

    space = {"x": tune.uniform(0, 1)} if completed or algorithm == "BlendSearch" else {"x": 1}
    objectives = (
        {
            "metrics": ["loss", "cost"],
            "modes": ["min", "min"],
            "tolerances": {"loss": 0.0, "cost": 0.0},
            "targets": {"loss": -float("inf"), "cost": -float("inf")},
        }
        if lexicographic
        else None
    )
    cls = getattr(blendsearch, algorithm)
    searcher = cls(space=space, metric="loss", mode="min", lexico_objectives=objectives)
    if completed:
        winner = searcher.suggest("completed")
        searcher.on_trial_complete(
            "completed", {"loss": 5.0, "cost": 1.0, "config": winner, "time_total_s": 1.0, "training_iteration": 0}
        )
    saved_results = copy.deepcopy(searcher._result)
    pending = searcher.suggest("pending")
    assert pending is not None
    if completed:
        searcher.on_trial_result(
            "pending", {"loss": 1.0, "cost": 1.0, "config": pending, "time_total_s": 1.0, "training_iteration": 0}
        )
    searcher.__dict__.pop("_trial_signatures")
    local_searchers = [searcher._ls] + [
        thread._search_alg for thread in searcher._search_thread_pool.values() if isinstance(thread._search_alg, FLOW2)
    ]
    for local in local_searchers:
        for field in ("_observation_order", "_next_observation_order", "_committed_order"):
            local.__dict__.pop(field, None)
    checkpoint = str(tmp_path / "legacy.pkl")
    searcher.save(checkpoint)
    restored = cls(space=space, metric="loss", mode="min", lexico_objectives=objectives)
    restored.restore(checkpoint)
    assert all(restored._result[key] == result for key, result in saved_results.items())
    restored.on_trial_complete("pending", error=True)
    assert restored.suggest("retry") is not None
    if completed:
        local = next(thread._search_alg for thread in restored._search_thread_pool.values() if thread._is_ls)
        assert local.best_config == winner
        assert local.best_obj == ({"loss": 5.0, "cost": 1.0} if lexicographic else 5.0)


class AbstractWarmStartTest:
    def setUp(self):
        # ray.init(num_cpus=1, local_mode=True)
        self.tmpdir = tempfile.mkdtemp()
        self.experiment_name = "searcher-state-Test.pkl"

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
        # ray.shutdown()

    def set_basic_conf(self):
        raise NotImplementedError

    def run_part_from_scratch(self):
        np.random.seed(162)
        search_alg, cost = self.set_basic_conf()
        search_alg = ConcurrencyLimiter(search_alg, 1)
        results_exp_1 = tune.run(cost, num_samples=5, search_alg=search_alg, verbose=0, local_dir=self.tmpdir)
        checkpoint_path = os.path.join(self.tmpdir, self.experiment_name)
        search_alg.save(checkpoint_path)
        return results_exp_1, np.random.get_state(), checkpoint_path

    def run_explicit_restore(self, random_state, checkpoint_path):
        search_alg2, cost = self.set_basic_conf()
        search_alg2 = ConcurrencyLimiter(search_alg2, 1)
        search_alg2.restore(checkpoint_path)
        return tune.run(cost, num_samples=5, search_alg=search_alg2, verbose=0)

    def run_full(self):
        np.random.seed(162)
        search_alg3, cost = self.set_basic_conf()
        search_alg3 = ConcurrencyLimiter(search_alg3, 1)
        return tune.run(cost, num_samples=10, search_alg=search_alg3, verbose=0)

    def testReproduce(self):
        results_exp_1, _, _ = self.run_part_from_scratch()
        results_exp_2, _, _ = self.run_part_from_scratch()
        trials_1_config = [trial.config for trial in results_exp_1.trials]
        trials_2_config = [trial.config for trial in results_exp_2.trials]
        self.assertEqual(trials_1_config, trials_2_config)

    def testWarmStart(self):
        results_exp_1, r_state, checkpoint_path = self.run_part_from_scratch()
        results_exp_2 = self.run_explicit_restore(r_state, checkpoint_path)
        results_exp_3 = self.run_full()
        trials_1_config = [trial.config for trial in results_exp_1.trials]
        trials_2_config = [trial.config for trial in results_exp_2.trials]
        trials_3_config = [trial.config for trial in results_exp_3.trials]
        self.assertEqual(trials_1_config + trials_2_config, trials_3_config)


class CFOWarmStartTest(AbstractWarmStartTest, unittest.TestCase):
    def set_basic_conf(self):
        space = {
            "height": tune.uniform(-100, 100),
            "width": tune.randint(0, 100),
        }

        def cost(param):
            tune.report(loss=(param["height"] - 14) ** 2 - abs(param["width"] - 3))

        search_alg = CFO(
            space=space,
            metric="loss",
            mode="min",
            seed=20,
        )

        return search_alg, cost


# class BlendsearchWarmStartTest(AbstractWarmStartTest, unittest.TestCase):
#     def set_basic_conf(self):
#         from flaml import BlendSearch
#         space = {
#             "height": tune.uniform(-100, 100),
#             "width": tune.randint(0, 100),
#         }

#         def cost(param):
#             tune.report(loss=(param["height"] - 14) ** 2 - abs(param["width"] - 3))

#         search_alg = BlendSearch(
#             space=space,
#             metric="loss",
#             mode="min",
#             seed=20,
#         )

#         return search_alg, cost
