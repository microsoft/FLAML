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


@pytest.mark.parametrize("python_version", ["3.10", "3.13"])
def test_mlflow_test_profile_uses_compatible_sqlalchemy(python_version):
    dependencies = requirements_for("test", python_version)
    assert "2.0.58" in dependencies["sqlalchemy"].specifier
    assert "2.1.0" not in dependencies["sqlalchemy"].specifier
    assert "sqlalchemy" not in requirements_for("automl", python_version)


def test_legacy_arm64_xgboost_build_has_an_explicit_native_target():
    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-package.yml"
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["build"]["steps"]
    index = next(
        i for i, step in enumerate(steps) if step.get("name") == "Build native XGBoost for Windows ARM64 Python 3.11"
    )
    build = steps[index]
    assert build["if"] == "matrix.os == 'windows-11-arm' && matrix.python-version == '3.11'"
    assert "-G 'Visual Studio 18 2026' -A ARM64" in build["run"]
    assert "--branch v3.2.0 --recurse-submodules" in build["run"]
    assert "-DUSE_CUDA=OFF -DUSE_OPENMP=OFF" in build["run"]
    assert "machine == 0xaa64" in build["run"]
    assert "python -m pip install --no-cache-dir (Join-Path $source 'python-package')" in build["run"]
    assert "xgboost.train(" in build["run"]
    assert "XGBClassifier" not in build["run"]
    assert index < next(i for i, step in enumerate(steps) if step.get("name") == "Install packages and dependencies")


def test_native_xgboost_smoke_does_not_require_sklearn():
    import subprocess
    import sys

    pytest.importorskip("xgboost")
    code = (
        "import sys; sys.modules['sklearn'] = None; "
        "import numpy as np, xgboost; "
        "X=np.arange(16).reshape(8,2); "
        "model=xgboost.train({'objective':'binary:logistic','nthread':1}, "
        "xgboost.DMatrix(X,label=np.tile([0,1],4)),num_boost_round=2); "
        "assert len(model.predict(xgboost.DMatrix(X[:2]))) == 2"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


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
    import shlex

    requirement = next(
        Requirement(package)
        for package in shlex.split(generated.cells[0].source)[2:]
        if package.lower().startswith("flaml")
    )
    assert "forecast" in requirement.extras
    committed = nbformat.read(root / "notebook" / "trident" / "all_in_one_test.ipynb", as_version=4)
    assert "git+https://github.com/microsoft/FLAML.git@main" in committed.cells[0].source
    assert all(not cell.get("outputs") for cell in generated.cells if cell.cell_type == "code")
    assert generated == committed


@pytest.mark.parametrize("extra", ["synapse", "fabric_python"])
def test_fabric_extras_use_lightweight_notebook_dependencies(extra):
    dependencies = requirements_for(extra)
    assert {"ipython", "nbformat", "mlflow-skinny"} <= dependencies.keys()
    assert not {"jupyter", "rouge-score"} & dependencies.keys()
    assert "1.3.0" in dependencies["scikit-learn"].specifier
    assert "1.2.0" not in dependencies["scikit-learn"].specifier


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
    spark_job = jobs["spark"]
    spark_entries = spark_job["strategy"]["matrix"]["include"]
    included.extend({"os": spark_job["runs-on"], "test-type": "spark", **entry} for entry in spark_entries)
    assert len(included) == 14
    assert sum(entry["os"] == "windows-11-arm" for entry in included) == 3
    assert all(entry["os"] == "ubuntu-latest" for entry in included if entry["test-type"] == "spark")
    assert {(entry["python-version"], entry["spark-version"]) for entry in spark_entries} == {
        ("3.11", "3.5.1"),
        ("3.12", "4.0.1"),
        ("3.13", "4.1.0"),
    }
    assert set(jobs["coverage"]["needs"]) == {"build", "spark"}
    assert "public-install" in jobs
    test_step = next(step for step in jobs["build"]["steps"] if step.get("name") == "Test with pytest (notspark)")
    assert test_step["env"]["PYTEST_FAULTHANDLER_TIMEOUT"] == (
        "${{ runner.os == 'Windows' && matrix.python-version == '3.11' && '0' || '120' }}"
    )
    assert '-o "faulthandler_timeout=$PYTEST_FAULTHANDLER_TIMEOUT"' in test_step["run"]
    assert "-p no:faulthandler" not in test_step["run"]
    assert test_step["timeout-minutes"] == 180
    for job in jobs.values():
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])


def test_spark_installation_and_pandas_cap_are_limited_to_spark_jobs():
    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[1]
    workflow = root / ".github" / "workflows" / "python-package.yml"
    jobs = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]
    assert all(
        "pyspark==" not in step.get("run", "") and "pandas>=2.0,<3" not in step.get("run", "")
        for step in jobs["build"]["steps"]
    )
    spark_job = jobs["spark"]
    assert any("docker compose build" in step.get("run", "") for step in spark_job["steps"])
    assert spark_job["env"]["SPARK_VERSION"] == "${{ matrix.spark-version }}"
    compose_path = root / spark_job["env"]["COMPOSE_FILE"]
    services = yaml.safe_load(compose_path.read_text(encoding="utf-8"))["services"]
    assert services["master"]["image"] == services["worker"]["image"] == services["driver"]["image"]
    build = services["master"]["build"]
    dockerfile = compose_path.parent / build["context"] / build["dockerfile"]
    content = dockerfile.read_text(encoding="utf-8")
    assert '"pyspark==${SPARK_VERSION}"' in content
    assert '"pandas>=2.0,<3"' in content


def test_declared_ray_compatibility_uses_real_worker_ci_profiles():
    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-package.yml"
    job = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["ray"]
    assert job["runs-on"] == "ubuntu-latest"
    assert set(job["strategy"]["matrix"]["ray-version"]) == {"1.13.0", "2.4.0"}
    assert job["permissions"] == {"contents": "read"}
    setup = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/setup-python@"))
    assert setup["with"]["python-version"] == "3.10"
    install = next(step for step in job["steps"] if step.get("name") == "Install declared Ray compatibility profile")
    assert '"sqlalchemy<2.1"' in install["run"]
    tests = next(step for step in job["steps"] if step.get("name") == "Verify real Ray reporting and trial ownership")
    assert "test/fabric/test_ray_reporting.py" in tests["run"]
    assert tests["timeout-minutes"] <= job["timeout-minutes"]


def test_offline_nlp_coverage_has_a_maintained_ci_step():
    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-package.yml"
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["build"]["steps"]
    nlp = next(step for step in steps if step.get("name") == "Test offline NLP utilities and classification heads")
    assert "matrix.test-type == 'notspark'" in nlp["if"]
    assert "matrix.python-version == '3.10'" not in nlp["if"]
    assert "test/nlp/test_hf_utils_coverage.py" in nlp["run"]
    assert "test/nlp/test_autohf_classificationhead.py" in nlp["run"]
    for selector in ("test_build_portfolio", "test_starting_point_not_in_search_space", "test_build_error_portfolio"):
        assert f"test/nlp/test_default.py::{selector}" in nlp["run"]
    assert nlp["env"]["HF_HUB_OFFLINE"] == "1"
    assert nlp["env"]["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("outcome", ["passed", "skipped", "missing"])
def test_offline_nlp_gate_rejects_missing_training_execution(tmp_path, outcome):
    import subprocess
    import sys
    import xml.etree.ElementTree as ET

    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-package.yml"
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["build"]["steps"]
    command = next(
        step["run"] for step in steps if step.get("name") == "Test offline NLP utilities and classification heads"
    )
    gate = command.split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    root = ET.Element("testsuite")
    if outcome != "missing":
        for task in ("seq-classification", "seq-regression"):
            case = ET.SubElement(
                root, "testcase", name=f"test_offline_automl_fit_and_predict_with_local_checkpoint[{task}]"
            )
            if outcome == "skipped":
                ET.SubElement(case, "skipped")
    ET.ElementTree(root).write(tmp_path / "offline-nlp.xml")
    result = subprocess.run([sys.executable, "-c", gate], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) is (outcome == "passed")
    if outcome != "passed":
        assert "Offline AutoML classification and regression" in result.stderr


def test_ci_write_permissions_are_limited_to_publication_jobs():
    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-package.yml"
    configuration = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert configuration["permissions"] == {"contents": "read"}
    jobs = configuration["jobs"]
    assert jobs["coverage"]["permissions"] == {"contents": "read", "pull-requests": "write"}
    assert jobs["save-dependencies"]["permissions"] == {"contents": "write"}
    assert jobs["save-dependencies"]["if"] == "github.ref == 'refs/heads/main'"
    assert jobs["save-dependencies"]["needs"] == "build"
    assert "permissions" not in jobs["build"]
    assert "permissions" not in jobs["public-install"]


def test_nlp_execution_is_not_excluded_from_coverage():
    import configparser
    import fnmatch

    root = Path(__file__).resolve().parents[1]
    config = configparser.ConfigParser()
    config.read(root / ".coveragerc", encoding="utf-8")
    source = (root / "flaml" / "automl" / "nlp" / "huggingface" / "utils.py").as_posix()
    assert not any(fnmatch.fnmatch(source, pattern.strip()) for pattern in config["run"]["omit"].splitlines())


def test_public_overview_uses_explicit_final_test_evaluation_and_no_editorial_metadata():
    import ast
    import json

    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "notebook" / "trident" / "FLAML Demo - Overview.ipynb").read_text(encoding="utf-8"))
    assert not any("ms_comments" in cell.get("metadata", {}) for cell in source["cells"])
    cells = ["".join(cell.get("source", [])) for cell in source["cells"] if cell["cell_type"] == "code"]
    definition = ast.parse(next(cell for cell in cells if "def predict(" in cell)).body[0]
    assert not definition.args.defaults
    for metric in ("default_metrics", "automl_metrics"):
        tree = ast.parse(next(cell for cell in cells if f"{metric} = predict(" in cell))
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == metric for target in node.targets)
        )
        assert isinstance(assignment.value.args[1], ast.Name)
        assert assignment.value.args[1].id == "test_data"
    combined = json.loads((root / "notebook" / "trident" / "all_in_one_test.ipynb").read_text(encoding="utf-8"))
    assert not any("ms_comments" in cell.get("metadata", {}) for cell in combined["cells"])


@pytest.mark.parametrize("job_name", ["build", "spark", "coverage"])
@pytest.mark.parametrize("upstream_status", ["success", "failure", "cancelled"])
def test_missing_coverage_fails_successful_jobs(tmp_path, job_name, upstream_status):
    import os
    import shutil
    import subprocess

    if os.name == "nt" or shutil.which("bash") is None:
        pytest.skip("Coverage workflow shell guards execute on Linux CI.")
    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-package.yml"
    jobs = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]
    names = {
        "build": "Combine per-job coverage data",
        "spark": "Combine container coverage",
        "coverage": "Combine coverage data and produce XML",
    }
    step = next(item for item in jobs[job_name]["steps"] if item.get("name") == names[job_name])
    variable = "BUILD_MATRIX_RESULT" if job_name == "coverage" else "TEST_JOB_RESULT"
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            variable: upstream_status,
            "SPARK_TEST_OUTPUT": str(tmp_path),
            "COVERAGE_FILE": str(tmp_path / ".coverage"),
            "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (1 if upstream_status == "success" else 0), result.stdout + result.stderr
    if upstream_status == "success":
        assert "::error::Successful" in result.stdout
