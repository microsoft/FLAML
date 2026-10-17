import itertools
import re
from importlib.metadata import requires
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def requirements_for(extra, python_version="3.13", machine="AMD64", platform="win32"):
    environment = {
        **default_environment(),
        "extra": canonicalize_name(extra),
        "python_version": python_version,
        "python_full_version": python_version + ".0",
        "sys_platform": platform,
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


def test_autofe_extra_declares_its_import_dependencies():
    dependencies = requirements_for("autofe")
    assert {"pandas", "scikit-learn", "xgboost"} <= dependencies.keys()
    assert not {"mlflow", "mlflow-skinny", "pyspark", "synapseml"} & dependencies.keys()


@pytest.mark.parametrize("python_version", ["3.10", "3.11", "3.12"])
@pytest.mark.parametrize("platform,machine", [("linux", "x86_64"), ("win32", "AMD64")])
def test_numpy_one_test_profile_uses_a_compatible_arrow_release(python_version, platform, machine):
    dependencies = requirements_for("test", python_version, machine, platform)
    assert "1.26.4" in dependencies["numpy"].specifier
    assert "2.0.0" not in dependencies["numpy"].specifier
    assert "25.0.1" in dependencies["pyarrow"].specifier
    assert "26.0.0" not in dependencies["pyarrow"].specifier


def test_arrow_compatibility_cap_does_not_expand_other_profiles():
    assert "pyarrow" not in requirements_for("test", "3.13")
    assert "pyarrow" not in requirements_for("test", "3.12", "ARM64")
    assert "pyarrow" not in requirements_for("automl", "3.12")


def test_fabric_notebooks_do_not_install_internal_release_wheels():
    notebook_root = Path(__file__).resolve().parents[1] / "notebook" / "trident"
    files = [*notebook_root.glob("*.ipynb"), notebook_root / "generate_test.py"]
    private_feed = re.compile(r"https://(?:automlsaeastus|synapsemldatascience)\.blob\.core\.windows\.net/releases/")
    assert not [str(path.name) for path in files if private_feed.search(path.read_text(encoding="utf-8"))]


def test_combined_notebook_generator_uses_public_source(tmp_path, monkeypatch):
    import ast
    import runpy
    import shutil

    nbformat = pytest.importorskip("nbformat")
    root = Path(__file__).resolve().parents[1]
    script = root / "notebook" / "trident" / "generate_test.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "all_notebooks" for target in node.targets)
    )
    for name in ast.literal_eval(assignment.value):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name="__main__")
    generated = nbformat.read(tmp_path / "notebook" / "trident" / "all_in_one_test.ipynb", as_version=4)
    nbformat.validate(generated)
    assert "git+https://github.com/microsoft/FLAML.git@main" in generated.cells[0].source
    committed = nbformat.read(root / "notebook" / "trident" / "all_in_one_test.ipynb", as_version=4)
    assert "git+https://github.com/microsoft/FLAML.git@main" in committed.cells[0].source
    assert all(not cell.get("outputs") for cell in generated.cells if cell.cell_type == "code")


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


def test_spark_installation_and_pandas_cap_are_limited_to_spark_jobs():
    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-package.yml"
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["build"]["steps"]
    affected = [
        step
        for step in steps
        if "pip install pyspark==" in step.get("run", "") or 'pip install "pandas>=2.0,<3"' in step.get("run", "")
    ]
    assert len(affected) == 4
    assert all("matrix.test-type == 'spark'" in step["if"] for step in affected)


def test_offline_nlp_coverage_has_a_maintained_ci_step():
    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-package.yml"
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["build"]["steps"]
    nlp = next(step for step in steps if step.get("name") == "Test offline NLP utilities and classification heads")
    assert "matrix.test-type == 'notspark'" in nlp["if"]
    assert "test/nlp/test_hf_utils_coverage.py" in nlp["run"]
    assert "test/nlp/test_autohf_classificationhead.py" in nlp["run"]
    assert nlp["env"]["HF_HUB_OFFLINE"] == "1"
    assert nlp["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_nlp_execution_is_not_excluded_from_coverage():
    import configparser
    import fnmatch

    root = Path(__file__).resolve().parents[1]
    config = configparser.ConfigParser()
    config.read(root / ".coveragerc", encoding="utf-8")
    source = (root / "flaml" / "automl" / "nlp" / "huggingface" / "utils.py").as_posix()
    assert not any(fnmatch.fnmatch(source, pattern.strip()) for pattern in config["run"]["omit"].splitlines())


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("upstream_status", ["success", "failure", "cancelled"])
def test_missing_coverage_fails_successful_jobs(tmp_path, aggregate, upstream_status):
    import os
    import shutil
    import subprocess

    if os.name == "nt" or shutil.which("bash") is None:
        pytest.skip("Coverage workflow shell guards execute on Linux CI.")
    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-package.yml"
    jobs = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]
    name = "Combine coverage data and produce XML" if aggregate else "Combine per-job coverage data"
    step = next(item for item in jobs["coverage" if aggregate else "build"]["steps"] if item.get("name") == name)
    variable = "BUILD_MATRIX_RESULT" if aggregate else "TEST_JOB_RESULT"
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=tmp_path,
        env={**os.environ, variable: upstream_status},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (1 if upstream_status == "success" else 0), result.stdout + result.stderr
