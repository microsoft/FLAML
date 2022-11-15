from flaml import AutoML, CFO, tune
from collections import defaultdict
import argparse
import pickle
import os
import math
import sys
import numpy as np
import pandas as pd


def _BraninCurrin(config):
    # Rescale brain
    x_1 = 15 * config["x1"] - 5
    x_2 = 15 * config["x2"]
    # Brain function
    t1 = x_2 - 5.1 / (4 * math.pi**2) * x_1**2 + 5 / math.pi * x_1 - 6
    t2 = 10 * (1 - 1 / (8 * math.pi)) * math.cos(x_1)
    brain_result = t1**2 + t2 + 10
    # Currin function
    xc_1 = config["x1"]
    xc_2 = config["x2"]
    factor1 = 1 - math.exp(-1 / (2 * xc_2))
    numer = 2300 * pow(xc_1, 3) + 1900 * pow(xc_1, 2) + 2092 * xc_1 + 60
    denom = 100 * pow(xc_1, 3) + 500 * pow(xc_1, 2) + 4 * xc_1 + 20
    currin_result = factor1 * numer / denom
    return {"brain": brain_result, "currin": currin_result}


def test_reproducibility():
    lexico_objectives = {}
    lexico_objectives["metrics"] = ["brain", "currin"]
    lexico_objectives["tolerances"] = {"brain": 10.0, "currin": 0.0}
    lexico_objectives["targets"] = {"brain": 0.0, "currin": 0.0}
    lexico_objectives["modes"] = ["min", "min"]

    search_space = {
        "x1": tune.uniform(lower=0.000001, upper=1.0),
        "x2": tune.uniform(lower=0.000001, upper=1.0),
    }

    analysis = tune.run(
        _BraninCurrin,
        num_samples=1000,
        config=search_space,
        use_ray=False,
        lexico_objectives=lexico_objectives,
    )

    print(analysis.best_trial)
    print(analysis.best_config)
    print(analysis.best_result)

    assert (
        analysis.best_result["currin"] <= 2.2
    ), "lexicographic optimization not reproducible"


if __name__ == "__main__":
    test_reproducibility()
