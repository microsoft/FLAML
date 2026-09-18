"""Smoke-test an installed public package from outside its source checkout."""

import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--automl", action="store_true")
    args = parser.parse_args()

    import flaml
    from flaml import tune

    assert not Path(flaml.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])
    assert flaml.has_automl is args.automl
    assert importlib.util.find_spec("mlflow") is None
    assert importlib.util.find_spec("pyspark") is None
    analysis = tune.run(
        lambda config: {"loss": 1.0},
        config={"x": 1},
        metric="loss",
        mode="min",
        num_samples=1,
        use_ray=False,
        verbose=0,
    )
    assert len(analysis.trials) == 1

    if args.automl:
        import numpy as np

        from flaml import AutoML
        from flaml.automl import register_automl_pipeline

        automl = AutoML(estimator_list=["rf"], max_iter=1, n_jobs=1, verbose=0)
        assert automl._settings["model_history"] is False
        assert automl._settings["featurization"] == "off"
        X = np.arange(80).reshape(40, 2)
        automl.fit(X, np.tile([0, 1], 20))
        assert len(automl.predict(X[:3])) == 3
        assert automl.mlflow_integration is None
        assert callable(register_automl_pipeline)


if __name__ == "__main__":
    main()
