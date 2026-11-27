import importlib
import os
import re
import sys
from types import ModuleType

from flaml.tune.spark.utils import (
    broadcast_code,
    check_spark,
    get_n_cpus,
    with_parameters,
)

_CODE_ARCHIVE = re.compile(r"^(\d{20})_(.+)_([0-9a-f]{64})_flaml_custom_code\.zip$")


def _archive_publication(path):
    match = _CODE_ARCHIVE.fullmatch(os.path.basename(path)) if isinstance(path, str) else None
    return (int(match[1]), match[2], match[3]) if match else None


def _resolve_code_alias(module):
    from flaml.tune.spark.utils import _broadcast_code_lock, _published_code_revisions

    with _broadcast_code_lock:
        name = module.__name__.rsplit(".", 1)[-1]
        if name in _published_code_revisions:
            publication = _published_code_revisions[name]
            if publication is None:
                raise ModuleNotFoundError(f"Custom module {name!r} has no successful publication.")
            revision = f"_flaml_custom_{name}_{publication[0]}"
        else:
            revision = vars(module)["_flaml_initial_revision"]
            sequence = vars(module)["_flaml_publication_sequence"]
            for path in sys.path:
                publication = _archive_publication(path)
                if publication and publication[1] == name and publication[0] > sequence:
                    sequence = publication[0]
                    revision = f"_flaml_custom_{name}_{publication[2]}"
        return importlib.import_module(f"{__name__}.{revision}")


class _CodeAlias(ModuleType):
    def __getattr__(self, name):
        revision = _resolve_code_alias(self)
        if name == "_flaml_revision":
            return revision
        if name == "__all__":
            return getattr(revision, "__all__", [key for key in vars(revision) if not key.startswith("_")])
        return getattr(revision, name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(_resolve_code_alias(self))))


def _install_code_alias(name, revision, sequence):
    module = sys.modules[name]
    module._flaml_initial_revision = revision
    module._flaml_publication_sequence = sequence
    module.__class__ = _CodeAlias
    _resolve_code_alias(module)


class _SparkCodePaths(list):
    def __iter__(self):
        yield from super().__iter__()
        # Spark may add archives after this package was imported by a reused worker.
        # Preserve qualified module names used by standard pickle in broadcast data.
        archives = [path for path in sys.path if isinstance(path, str) and path.endswith("_flaml_custom_code.zip")]
        archives.sort(key=lambda path: (_archive_publication(path) or (0,))[0], reverse=True)
        for path in archives:
            yield os.path.join(path, "_flaml_code")
            yield path  # Preserve older revision archives used by existing pickles.


__path__ = _SparkCodePaths(__path__)

__all__ = ["check_spark", "get_n_cpus", "with_parameters", "broadcast_code"]
