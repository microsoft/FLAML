from flaml import BlendSearch, tune


def test_metric_constraint_is_checked_when_the_metric_is_zero():
    def evaluate_config(config):
        # recall is exactly 0.0 above x=5, which violates recall >= 0.5
        return {"score": config["x"], "recall": 0.0 if config["x"] > 5 else 1.0}

    analysis = tune.run(
        evaluate_config,
        config={"x": tune.uniform(0, 10)},
        metric="score",
        mode="max",
        metric_constraints=[("recall", ">=", 0.5)],
        # one feasible and one violating point are given explicitly, so no search proposal is needed
        points_to_evaluate=[{"x": 2.0}, {"x": 8.0}],
        num_samples=2,
        use_ray=False,
        verbose=0,
    )
    assert len(analysis.trials) == 2
    penalized = unpenalized = 0
    for trial in analysis.trials:
        result = trial.last_result
        modified = result["score" + BlendSearch.lagrange]
        if result["recall"] == 0.0:
            # the violated constraint must lower the (maximized) objective
            assert modified < result["score"]
            penalized += 1
        else:
            assert modified == result["score"]
            unpenalized += 1
    assert penalized == 1 and unpenalized == 1
