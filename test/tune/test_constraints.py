def test_config_constraint():
    from flaml import tune

    # Test dict return value
    def evaluate_config_dict(config):
        metric = (round(config["x"]) - 85000) ** 2 - config["x"] / config["y"]
        return {"metric": metric}

    def config_constraint(config):
        if config["y"] >= config["x"]:
            return 1
        else:
            return 0

    analysis = tune.run(
        evaluate_config_dict,
        config={
            "x": tune.qloguniform(lower=1, upper=100000, q=1),
            "y": tune.qrandint(lower=2, upper=100000, q=2),
        },
        config_constraints=[(config_constraint, "<", 0.5)],
        metric="metric",
        mode="max",
        num_samples=100,
        log_file_name="logs/config_constraint.log",
    )

    assert analysis.best_config["x"] > analysis.best_config["y"]
    assert analysis.trials[0].config["x"] > analysis.trials[0].config["y"]


def test_config_constraint_strict_less_than_excludes_threshold():
    from flaml import CFO, tune

    evaluated = []

    def evaluate_config(config):
        evaluated.append(config["n"])
        return {"metric": config["n"]}

    space = {"n": tune.randint(0, 11)}
    tune.run(
        evaluate_config,
        config=space,
        search_alg=CFO(
            space=space,
            metric="metric",
            mode="min",
            config_constraints=[(lambda config: config["n"], "<", 5)],
            points_to_evaluate=[{"n": 5}, {"n": 3}],
        ),
        metric="metric",
        mode="min",
        num_samples=2,
        use_ray=False,
        verbose=0,
    )
    # the boundary point is skipped and the next feasible point is evaluated
    assert evaluated[0] == 3
    assert max(evaluated) < 5


def test_blendsearch_fallback_config_respects_strict_less_than():
    from unittest import mock

    from flaml import BlendSearch, tune

    space = {"n": tune.randint(0, 11)}
    searcher = BlendSearch(
        space=space,
        metric="metric",
        mode="min",
        low_cost_partial_config={"n": 5},
        points_to_evaluate=[{"n": 0}],
        config_constraints=[(lambda config: config["n"], "<", 5)],
    )
    # the first trial stays outstanding
    assert searcher.suggest("t1") == {"n": 0}
    # a global proposal that is not valid makes BlendSearch fall back to the low cost init point,
    # which sits on the threshold of the strict '<' constraint
    with (
        mock.patch.object(searcher._search_thread_pool[0], "suggest", return_value={"n": 1}),
        mock.patch.object(searcher, "_valid", return_value=False),
    ):
        assert searcher.suggest("t2") is None


def test_blendsearch_rejected_fallback_retires_global_proposal():
    from unittest import mock

    from optuna.trial import TrialState

    from flaml import BlendSearch, tune

    space = {"n": tune.randint(0, 11)}
    searcher = BlendSearch(
        space=space,
        metric="metric",
        mode="min",
        low_cost_partial_config={"n": 5},
        points_to_evaluate=[{"n": 0}],
        config_constraints=[(lambda config: config["n"], "<", 5)],
    )
    # the first trial stays outstanding
    assert searcher.suggest("t1") == {"n": 0}
    thread = searcher._search_thread_pool[0]
    study = thread._search_alg._ot_study

    def running_trials():
        return [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.RUNNING]

    baseline_running = thread.running
    baseline_trials = len(running_trials())
    # only the admissibility check is forced to fail: the proposals and their accounting stay real
    with mock.patch.object(searcher, "_valid", return_value=False):
        for i in range(3):
            study.enqueue_trial({"n": 1 + i})
            assert searcher.suggest(f"t{i + 2}") is None
    assert thread.running == baseline_running
    assert len(running_trials()) == baseline_trials
    assert not any(t in searcher._trial_proposed_by for t in ("t2", "t3", "t4"))
