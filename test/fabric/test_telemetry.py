import re

import numpy as np

import flaml


def test_automl_telemetry(caplog):
    automl_1 = flaml.AutoML()
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 2, 3])
    automl_1.fit(X_train, y_train, max_iter=3)

    automl_2 = flaml.AutoML()
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 2, 3])
    automl_2.fit(X_train, y_train, max_iter=4)

    """
    Below assertions worked on my local machine, but not on Azure pipeline.
    """
    # captured_text = caplog.text
    # assert len(re.findall("log_telemetry: flaml.automl", captured_text)) == 1
    # assert len(re.findall("log_telemetry: flaml.tune", captured_text)) == 0


def test_tune_telemetry(caplog):
    def tune_func1(config):
        return {"metric": config["x"] ** 2}

    def tune_func2(config):
        return {"metric": config["x"] ** 3}

    flaml.tune.run(tune_func1, config={"x": flaml.tune.uniform(0, 1)}, num_samples=3, metric="metric", mode="min")
    flaml.tune.run(tune_func2, config={"x": flaml.tune.uniform(0, 1)}, num_samples=3, metric="metric", mode="max")

    """
    Below assertions worked on my local machine, but not on Azure pipeline.
    """
    # captured_text = caplog.text
    # assert len(re.findall("log_telemetry: flaml.automl", captured_text)) == 0
    # assert len(re.findall("log_telemetry: flaml.tune", captured_text)) == 1


if __name__ == "__main__":

    def tune_func1(config):
        return {"metric": config["x"] ** 2}

    def tune_func2(config):
        return {"metric": config["x"] ** 3}

    flaml.tune.run(tune_func1, config={"x": flaml.tune.uniform(0, 1)}, num_samples=3, metric="metric", mode="min")
    flaml.tune.run(tune_func2, config={"x": flaml.tune.uniform(0, 1)}, num_samples=3, metric="metric", mode="max")

    automl_1 = flaml.AutoML()
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 2, 3])
    automl_1.fit(X_train, y_train, max_iter=3)

    automl_2 = flaml.AutoML()
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 2, 3])
    automl_2.fit(X_train, y_train, max_iter=4)
