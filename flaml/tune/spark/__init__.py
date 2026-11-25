import os
import sys

from flaml.tune.spark.utils import (
    broadcast_code,
    check_spark,
    get_n_cpus,
    with_parameters,
)


class _SparkCodePaths(list):
    def __iter__(self):
        yield from super().__iter__()
        # Spark may add archives after this package was imported by a reused worker.
        # Preserve qualified module names used by standard pickle in broadcast data.
        for path in sys.path:
            if isinstance(path, str) and path.endswith("_flaml_custom_code.zip"):
                yield os.path.join(path, "_flaml_code")
                yield path  # Preserve older revision archives used by existing pickles.


__path__ = _SparkCodePaths(__path__)

__all__ = ["check_spark", "get_n_cpus", "with_parameters", "broadcast_code"]
