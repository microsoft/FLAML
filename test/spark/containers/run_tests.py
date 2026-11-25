"""Run the Spark suite only after proving execution on separate containers."""

import json
import os
import socket
import sys
import time
from urllib.request import urlopen

import pytest


def _wait_for_workers(count):
    deadline = time.monotonic() + 120
    workers = []
    error = None
    while time.monotonic() < deadline:
        try:
            with urlopen("http://master:8080/json/", timeout=5) as response:
                workers = [worker for worker in json.load(response)["workers"] if worker["state"] == "ALIVE"]
            error = None
            if len(workers) == count and len({worker["host"] for worker in workers}) == count:
                return
        except (OSError, ValueError) as exc:
            error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"Expected {count} distinct registered Spark workers; found {workers}; last error: {error}")


def _executor_identity(value):
    import socket
    import sys

    import pyspark
    from pyspark import TaskContext

    import flaml

    if TaskContext.get() is None:
        raise RuntimeError("Spark work ran outside an executor task")
    return value, socket.gethostname(), sys.version_info[:2], pyspark.__version__, flaml.__version__


def _barrier_identity(values):
    from pyspark import BarrierTaskContext

    BarrierTaskContext.get().barrier()
    return [_executor_identity(sum(values))]


def verify_cluster(spark, master, worker_count):
    context = spark.sparkContext
    if not master.startswith("spark://") or context.master != master:
        raise RuntimeError(f"Container tests require master {master!r}, not {context.master!r}")
    _wait_for_workers(worker_count)
    results = (
        context.parallelize(range(worker_count), worker_count).barrier().mapPartitions(_barrier_identity).collect()
    )
    hosts = {result[1] for result in results}
    if len(hosts) != worker_count or hosts.intersection({socket.gethostname(), "master", "driver"}):
        raise RuntimeError(f"Expected execution on {worker_count} worker containers, got {results}")

    import pyspark
    from joblib import Parallel, delayed, parallel_backend
    from joblibspark import register_spark

    import flaml

    expected_runtime = (sys.version_info[:2], pyspark.__version__, flaml.__version__)
    if any(result[2:] != expected_runtime for result in results):
        raise RuntimeError(f"Driver/executor runtime mismatch: expected {expected_runtime}, got {results}")

    register_spark()
    with parallel_backend("spark"):
        joblib_results = Parallel(n_jobs=worker_count, batch_size=1)(
            delayed(_executor_identity)(value) for value in range(worker_count)
        )
    if (
        [result[0] for result in joblib_results] != list(range(worker_count))
        or not {result[1] for result in joblib_results}.issubset(hosts)
        or any(result[2:] != expected_runtime for result in joblib_results)
    ):
        raise RuntimeError(f"Joblib did not execute on the verified Spark workers: {joblib_results}")
    return {"spark": results, "joblib": joblib_results}


class ClusterTests:
    def __init__(self, master, worker_count):
        self.master = master
        self.worker_count = worker_count
        self.spark = None

    def pytest_sessionstart(self, session):
        from test.spark._init_spark import init_spark_session

        self.spark = init_spark_session("flaml-container-tests")
        evidence = verify_cluster(self.spark, self.master, self.worker_count)
        session.config.pluginmanager.get_plugin("terminalreporter").write_line(
            "Container executor proof: " + json.dumps(evidence)
        )


def main():
    master = os.environ.get("FLAML_TEST_SPARK_MASTER", "")
    worker_count = int(os.environ.get("FLAML_TEST_SPARK_WORKERS", "0"))
    if not master.startswith("spark://") or worker_count < 2:
        raise RuntimeError("Container tests require FLAML_TEST_SPARK_MASTER and at least two FLAML_TEST_SPARK_WORKERS")
    guard = ClusterTests(master, worker_count)
    try:
        return pytest.main(
            sys.argv[1:]
            or [
                "test/",
                "--ignore=test/autogen",
                "--ignore=test/nlp",
                "--reruns",
                "2",
                "--reruns-delay",
                "10",
                "--junitxml=/coverage/pytest.xml",
                "-m",
                "spark",
            ],
            plugins=[guard],
        )
    finally:
        if guard.spark is not None:
            guard.spark.stop()
            guard.spark.sparkContext._gateway.shutdown_callback_server()


if __name__ == "__main__":
    raise SystemExit(main())
