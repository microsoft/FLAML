import itertools
import re
from importlib.metadata import requires
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def requirements_for(extra, python_version="3.13", machine="AMD64"):
    environment = {
        **default_environment(),
        "extra": canonicalize_name(extra),
        "python_version": python_version,
        "python_full_version": python_version + ".0",
        "sys_platform": "win32",
        "platform_machine": machine,
    }
    requirements = [Requirement(value) for value in requires("FLAML")]
    return {
        canonicalize_name(requirement.name): requirement
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate(environment)
    }


def test_public_install_dependencies_stay_optional():
    assert set(requirements_for("")) == {"numpy"}
    assert not {"mlflow", "mlflow-skinny", "pyspark", "synapseml"} & requirements_for("automl").keys()


@pytest.mark.parametrize("extra", ["synapse", "fabric_python"])
def test_fabric_extras_use_lightweight_notebook_dependencies(extra):
    dependencies = requirements_for(extra)
    assert {"ipython", "nbformat", "mlflow-skinny"} <= dependencies.keys()
    assert not {"jupyter", "rouge-score"} & dependencies.keys()


@pytest.mark.parametrize("python_version,maximum", [("3.12", "1.3.2"), ("3.13", "1.4.2")])
def test_synapse_joblib_constraints(python_version, maximum):
    joblib = requirements_for("synapse", python_version)["joblib"]
    assert joblib.specifier.contains(maximum)
    assert not joblib.specifier.contains("1.5.0")
    assert joblib.specifier.contains("1.4.2") is (python_version == "3.13")


def test_arm64_dependencies_keep_plotting_and_async_support():
    dependencies = requirements_for("test", machine="ARM64")
    assert {"plotly", "pytest-asyncio", "pytest-xdist"} <= dependencies.keys()
    assert "catboost" not in dependencies


def test_ci_matrix_preserves_arm64_and_split_spark_jobs():
    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-package.yml"
    jobs = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]
    matrix = jobs["build"]["strategy"]["matrix"]
    entries = [
        dict(zip(("os", "python-version", "test-type"), values))
        for values in itertools.product(matrix["os"], matrix["python-version"], matrix["test-type"])
    ]
    included = [
        entry
        for entry in entries
        if not any(all(entry[key] == value for key, value in exclusion.items()) for exclusion in matrix["exclude"])
    ]
    assert len(included) == 14
    assert sum(entry["os"] == "windows-11-arm" for entry in included) == 3
    assert all(entry["os"] == "ubuntu-latest" for entry in included if entry["test-type"] == "spark")
    assert "public-install" in jobs
    for job in jobs.values():
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])
