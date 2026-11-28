import subprocess
import sys
from pathlib import Path
from test._artifact_paths import artifact_path

import pytest


def test_artifact_paths_are_owned_by_module_and_process():
    first = artifact_path(__file__, "automl.pkl")
    other_module = str(Path(__file__).with_name("another_test.py"))
    second = artifact_path(other_module, "automl.pkl")
    assert first != second
    assert artifact_path(__file__, "automl.pkl") == first
    assert not Path(first).is_relative_to(Path(__file__).resolve().parents[1])
    child = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from test._artifact_paths import artifact_path; import sys; print(artifact_path(sys.argv[1], 'automl.pkl'))",
            __file__,
        ],
        text=True,
    ).strip()
    assert child != first


def test_artifact_path_rejects_escape():
    with pytest.raises(ValueError, match="owning module"):
        artifact_path(__file__, "../escape.pkl")


def test_shared_nlp_settings_allocate_distinct_logs():
    import ast

    path = Path(__file__).resolve().parent / "nlp" / "utils.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    definition = next(
        node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "get_automl_settings"
    )
    namespace = {"__file__": str(path)}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    first = namespace["get_automl_settings"]()["log_file_name"]
    second = namespace["get_automl_settings"]()["log_file_name"]
    assert first != second
    assert Path(first).parent == Path(second).parent
