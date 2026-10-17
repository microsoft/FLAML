"""Process- and module-owned paths for legacy test artifacts."""

from pathlib import Path
from tempfile import TemporaryDirectory

_temporary_directory = TemporaryDirectory(prefix="flaml_test_artifacts_")
_test_root = Path(__file__).resolve().parent


def artifact_path(module_file, relative_path):
    module = Path(module_file).resolve().relative_to(_test_root).with_suffix("")
    directory = Path(_temporary_directory.name) / module
    path = (directory / relative_path).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Test artifacts must stay within their owning module directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
