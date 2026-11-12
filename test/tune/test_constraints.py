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
            points_to_evaluate=[{"n": 5}],
        ),
        metric="metric",
        mode="min",
        num_samples=30,
        use_ray=False,
        verbose=0,
    )
    assert evaluated
    assert max(evaluated) < 5
