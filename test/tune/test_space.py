def test_define_by_run():
    from flaml import BlendSearch, tune
    from flaml.tune.space import unflatten_hierarchical

    config = {
        # Sample a float uniformly between -5.0 and -1.0
        "uniform": tune.uniform(-5, -1),
        # Sample a float uniformly between 3.2 and 5.4,
        # rounding to increments of 0.2
        "quniform": tune.quniform(3.2, 5.4, 0.2),
        # Sample a float uniformly between 0.0001 and 0.01, while
        # sampling in log space
        "loguniform": tune.loguniform(1e-4, 1e-2),
        # Sample a float uniformly between 0.0001 and 0.1, while
        # sampling in log space and rounding to increments of 0.00005
        "qloguniform": tune.qloguniform(1e-4, 1e-1, 5e-5),
        # Sample a random float from a normal distribution with
        # mean=10 and sd=2
        # "randn": tune.randn(10, 2),
        # Sample a random float from a normal distribution with
        # mean=10 and sd=2, rounding to increments of 0.2
        # "qrandn": tune.qrandn(10, 2, 0.2),
        # Sample a integer uniformly between -9 (inclusive) and 15 (exclusive)
        "randint": tune.randint(-9, 15),
        # Sample a random uniformly between -21 (inclusive) and 12 (inclusive (!))
        # rounding to increments of 3 (includes 12)
        "qrandint": tune.qrandint(-21, 12, 3),
        # Sample a integer uniformly between 1 (inclusive) and 10 (exclusive),
        # while sampling in log space
        "lograndint": tune.lograndint(1, 10),
        # Sample a integer uniformly between 1 (inclusive) and 10 (inclusive (!)),
        # while sampling in log space and rounding to increments of 2
        "qlograndint": tune.qlograndint(1, 10, 2),
        # Sample an option uniformly from the specified choices
        "choice": tune.choice(["a", "b", "c"]),
    }
    bs = BlendSearch(
        space={"c": tune.choice([{"nested": config}])}, metric="metric", mode="max"
    )
    for i in range(1):
        config = bs._gs.suggest(f"{i}")
        print(config)
        print(unflatten_hierarchical(config, bs._gs.space)[0])
