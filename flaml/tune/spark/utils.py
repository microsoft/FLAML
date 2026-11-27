import ast
import errno
import hashlib
import importlib
import json
import keyword
import logging
import os
import sys
import tempfile
import textwrap
import threading
import time
from contextlib import contextmanager
from functools import lru_cache, partial
from types import ModuleType
from zipfile import ZIP_DEFLATED, ZipFile

logger = logging.getLogger(__name__)
logger_formatter = logging.Formatter(
    "[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S"
)
logger.propagate = False
_broadcast_code_lock = threading.RLock()
_published_code_aliases = set()
_published_code_revisions = {}
_publication_sequence = 0
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
try:
    import py4j
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.util import VersionUtils
except ImportError:
    _have_spark = False
    py4j = None
    _spark_major_minor_version = (0, 0)
else:
    _have_spark = True
    _spark_major_minor_version = VersionUtils.majorMinorVersion(pyspark.__version__)


@lru_cache(maxsize=2)
def check_spark():
    """Check if Spark is installed and running.
    Result of the function will be cached since test once is enough. As lru_cache will not
    cache exceptions, we don't raise exceptions here but only log a warning message.

    Returns:
        Return (True, None) if the check passes, otherwise log the exception message and
        return (False, Exception(msg)). The exception can be raised by the caller.
    """
    logger.debug("\nchecking Spark installation...This line should appear only once.\n")
    if not _have_spark:
        msg = """use_spark=True requires installation of PySpark. Please run pip install flaml[spark]
        and check [here](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
        for more details about installing Spark."""
        return False, ImportError(msg)

    if _spark_major_minor_version[0] < 3:
        msg = "Spark version must be >= 3.0 to use flaml[spark]"
        return False, ImportError(msg)

    try:
        SparkSession.builder.getOrCreate()
    except RuntimeError as e:
        return False, RuntimeError(e)

    return True, None


def get_n_cpus(node="driver"):
    """Get the number of CPU cores of the given type of node.

    Args:
        node: string | The type of node to get the number of cores. Can be 'driver' or 'executor'.
            Default is 'driver'.

    Returns:
        An int of the number of CPU cores.
    """
    assert node in ["driver", "executor"]
    try:
        n_cpus = int(SparkSession.builder.getOrCreate().sparkContext.getConf().get(f"spark.{node}.cores"))
    except (TypeError, RuntimeError):
        n_cpus = os.cpu_count()
    return n_cpus


def with_parameters(trainable, **kwargs):
    """Wrapper for trainables to pass arbitrary large data objects.

    This wrapper function will store all passed parameters in the Spark
    Broadcast variable.

    Args:
        trainable: Trainable to wrap.
        **kwargs: parameters to store in object store.

    Returns:
        A new function with partial application of the given arguments
        and keywords. The given arguments and keywords will be broadcasted
        to all the executors.


    ```python
    import pyspark
    import flaml
    from sklearn.datasets import load_iris
    def train(config, data=None):
        if isinstance(data, pyspark.broadcast.Broadcast):
            data = data.value
        print(config, data)

    data = load_iris()
    with_parameters_train = flaml.tune.spark.utils.with_parameters(train, data=data)
    with_parameters_train(config=1)
    train(config={"metric": "accuracy"})
    ```
    """

    if not callable(trainable):
        raise ValueError(
            f"`with_parameters() only works with function trainables`. " f"Got type: " f"{type(trainable)}."
        )

    spark_available, spark_error_msg = check_spark()
    if not spark_available:
        raise spark_error_msg
    spark = SparkSession.builder.getOrCreate()

    bc_kwargs = dict()
    for k, v in kwargs.items():
        bc_kwargs[k] = spark.sparkContext.broadcast(v)

    return partial(trainable, **bc_kwargs)


def _publish_code_file(path, contents):
    descriptor, temporary = tempfile.mkstemp(prefix=".flaml-code-", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(contents)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _publication_file(directory, name):
    return os.path.join(directory, ".flaml-code-state", name + ".json")


def _latest_archive_sequence(directory, name):
    if directory is None:
        return -1
    from flaml.tune.spark import _archive_publication

    latest = -1
    with os.scandir(directory) as entries:
        for entry in entries:
            publication = _archive_publication(entry.name)
            if publication and publication[1] == name and entry.is_file():
                latest = max(latest, publication[0])
    return latest


def _validate_publication(publication, name):
    if publication is None:
        return None
    prefix = f"_flaml_custom_{name}_"
    if (
        not isinstance(publication, dict)
        or not isinstance(publication.get("revision"), str)
        or not publication["revision"].startswith(prefix)
        or len(publication["revision"]) != len(prefix) + 64
        or set(publication["revision"][len(prefix) :]) - set("0123456789abcdef")
        or type(publication.get("sequence")) is not int
        or not 0 <= publication["sequence"] < 10**20
    ):
        raise ValueError(f"Invalid committed publication for custom module {name!r}.")
    return publication


def _read_publication(directory, name):
    if not directory:
        return False, None
    try:
        with open(_publication_file(directory, name), encoding="utf-8") as source:
            return True, _validate_publication(json.load(source), name)
    except (FileNotFoundError, NotADirectoryError):
        return False, None


def _legacy_publication(source, name):
    if source is None:
        return None
    try:
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_flaml_install_alias"
                and len(node.args) == 3
            ):
                return _validate_publication(
                    {"revision": ast.literal_eval(node.args[1]), "sequence": ast.literal_eval(node.args[2])},
                    name,
                )
    except (SyntaxError, TypeError, ValueError):
        pass
    return None


def _validate_code_namespace(file_name):
    package = sys.modules.get("flaml.tune.spark")
    if package is None:
        return
    existing = vars(package).get(file_name)
    existing_revision = vars(existing).get("_flaml_revision") if isinstance(existing, ModuleType) else None
    initial_revision = vars(existing).get("_flaml_initial_revision", "") if isinstance(existing, ModuleType) else ""
    owned_alias = (
        isinstance(existing, ModuleType)
        and existing.__name__ == f"flaml.tune.spark.{file_name}"
        and (
            isinstance(initial_revision, str)
            and initial_revision.startswith(f"_flaml_custom_{file_name}_")
            or isinstance(existing_revision, ModuleType)
            and existing_revision.__name__.startswith(f"flaml.tune.spark._flaml_custom_{file_name}_")
        )
    )
    reserved = {
        name.casefold()
        for name in vars(package)
        if name not in _published_code_aliases and not (name == file_name and owned_alias)
    }
    if file_name.casefold() in reserved or (
        existing is not None
        and not (isinstance(existing, ModuleType) and existing.__name__ == f"flaml.tune.spark.{file_name}")
    ):
        raise ValueError("file_name must not replace the Spark package's existing module API.")


@contextmanager
def _publication_file_lock(path, name):
    # Writers share a file lock; readers only need the atomically replaced state.
    _validate_code_namespace(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "a+b", buffering=0) as stream:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    os.lseek(stream.fileno(), 0, os.SEEK_SET)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(0.05)

            def unlock():
                os.lseek(stream.fileno(), 0, os.SEEK_SET)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)

            def unlock():
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

        primary = None
        try:
            yield
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                unlock()
            except OSError:
                if primary is None:
                    raise
                logger.exception("Failed to release the custom-code publication lock.")


def broadcast_code(custom_code="", file_name="mylearner"):
    """Write customized learner/metric code contents to a file for importing.
    It is necessary for using the customized learner/metric in spark backend.
    The path of the immutable learner/metric source file will be returned.
    Reimport the public module after publishing changed code; its classes retain
    revision-specific identities so existing and future Spark workers agree.
    The driver installation must be writable, as in previous versions.

    Args:
        custom_code: str, default="" | code contents of the custom learner/metric.
        file_name: str, default="mylearner" | file name of the custom learner/metric.

    Returns:
        The path of the custom code file.
    ```python
    from flaml.tune.spark.utils import broadcast_code
    from flaml.automl.model import LGBMEstimator

    custom_code = '''
    from flaml.automl.model import LGBMEstimator
    from flaml import tune

    class MyLargeLGBM(LGBMEstimator):
        @classmethod
        def search_space(cls, **params):
            return {
                "n_estimators": {
                    "domain": tune.lograndint(lower=4, upper=32768),
                    "init_value": 32768,
                    "low_cost_init_value": 4,
                },
                "num_leaves": {
                    "domain": tune.lograndint(lower=4, upper=32768),
                    "init_value": 32768,
                    "low_cost_init_value": 4,
                },
            }
    '''

    broadcast_code(custom_code=custom_code)
    from flaml.tune.spark.mylearner import MyLargeLGBM
    assert isinstance(MyLargeLGBM(), LGBMEstimator)
    ```
    """
    global _publication_sequence
    if (
        not isinstance(file_name, str)
        or not file_name.isidentifier()
        or keyword.iskeyword(file_name)
        or file_name.casefold() in {"__init__", "utils", "sys", "os"}
        or file_name.startswith("__")
        or file_name.casefold().startswith("_flaml_")
    ):
        raise ValueError("file_name must be a non-reserved Python module name.")
    spark_available, _ = check_spark()
    flaml_path = os.path.dirname(os.path.abspath(__file__))
    custom_code = textwrap.dedent(custom_code)
    digest = hashlib.sha256(("flaml-code-v4\0" + file_name + "\0" + custom_code).encode("utf-8")).hexdigest()
    revision = f"_flaml_custom_{file_name}_{digest}"
    custom_path = os.path.join(flaml_path, revision + ".py")
    alias_path = os.path.join(flaml_path, file_name + ".py")
    state_path = _publication_file(flaml_path, file_name)
    with _broadcast_code_lock, _publication_file_lock(state_path, file_name):
        previous_alias = None
        if os.path.exists(alias_path):
            with open(alias_path, encoding="utf-8", newline="") as source:
                previous_alias = source.read()
        state_exists, committed = _read_publication(flaml_path, file_name)
        if not state_exists:
            committed = _legacy_publication(previous_alias, file_name)
            _publish_code_file(state_path, json.dumps(committed) + "\n")
        previous = _published_code_revisions.get(file_name)
        if committed is None:
            previous = None
        elif previous is None or (previous[0], previous[2]) != (
            committed["revision"].rsplit("_", 1)[-1],
            committed["sequence"],
        ):
            previous = (committed["revision"].rsplit("_", 1)[-1], None, committed["sequence"], None)
        _published_code_revisions[file_name] = previous
        spark = SparkSession.builder.getOrCreate() if spark_available else None
        root = pyspark.SparkFiles.getRootDirectory() if spark_available else None
        latest_delivery = _latest_archive_sequence(root, file_name)
        if (
            previous
            and previous[:2] == (digest, root)
            and previous[2] >= latest_delivery
            and (previous[3] is None or os.path.exists(previous[3]))
        ):
            sequence, archive_path = previous[2:]
        else:
            _publication_sequence = max(
                _publication_sequence + 1,
                time.time_ns(),
                committed["sequence"] + 1 if committed else 0,
                latest_delivery + 1,
            )
            sequence = _publication_sequence
            archive_path = (
                os.path.join(root, f"{sequence:020d}_{file_name}_{digest}_flaml_custom_code.zip")
                if root is not None
                else None
            )
        alias = (
            "from . import _install_code_alias as _flaml_install_alias\n"
            f"_flaml_install_alias(__name__, {revision!r}, {sequence})\n"
        )
        publication = _validate_publication({"revision": revision, "sequence": sequence}, file_name)
        cached_alias = importlib.util.cache_from_source(alias_path)
        alias_prepared = False
        archive_created = False
        delivery_attempted = False
        try:
            if not os.path.exists(custom_path):
                _publish_code_file(custom_path, custom_code)
            if spark_available and not os.path.exists(archive_path):
                descriptor, temporary = tempfile.mkstemp(dir=os.path.dirname(archive_path), suffix=".zip")
                os.close(descriptor)
                try:
                    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
                        archive.writestr(f"_flaml_code/{revision}.py", custom_code)
                        archive.writestr(f"_flaml_code/{file_name}.py", alias)
                    os.replace(temporary, archive_path)
                    archive_created = True
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            if os.path.exists(cached_alias):
                os.unlink(cached_alias)
            _publish_code_file(alias_path, alias)
            alias_prepared = True
            importlib.invalidate_caches()
            # Complete fallible local preparation before making the revision visible to workers.
            if spark_available:
                delivery_attempted = True
                spark.sparkContext.addPyFile(archive_path)
        except BaseException:
            if alias_prepared:
                try:
                    if previous_alias is None:
                        os.unlink(alias_path)
                    else:
                        _publish_code_file(alias_path, previous_alias)
                    if os.path.exists(cached_alias):
                        os.unlink(cached_alias)
                    importlib.invalidate_caches()
                except Exception:
                    logger.exception("Failed to restore the driver alias after custom-code publication failed.")
            if archive_created and not delivery_attempted:
                try:
                    os.unlink(archive_path)
                except OSError:
                    logger.exception("Failed to remove an uncommitted custom-code archive.")
            raise
        # Retained archives also preserve possibly delivered sequences if an acknowledgement is lost.
        _publish_code_file(state_path, json.dumps(publication) + "\n")
        _published_code_aliases.add(file_name)
        _published_code_revisions[file_name] = (digest, root, sequence, archive_path)

    return custom_path


def get_broadcast_data(broadcast_data):
    """Get the broadcast data from the broadcast variable.

    Args:
        broadcast_data: pyspark.broadcast.Broadcast | the broadcast variable.

    Returns:
        The broadcast data.
    """
    if _have_spark and isinstance(broadcast_data, pyspark.broadcast.Broadcast):
        broadcast_data = broadcast_data.value
    return broadcast_data


class PySparkOvertimeMonitor:
    """A context manager class to monitor if the PySpark job is overtime.
    Example:

    ```python
    with PySparkOvertimeMonitor(time_start, time_budget_s, force_cancel, parallel=parallel):
        results = parallel(
            delayed(evaluation_function)(trial_to_run.config)
            for trial_to_run in trials_to_run
        )
    ```

    """

    def __init__(
        self,
        start_time,
        time_budget_s,
        force_cancel=False,
        cancel_func=None,
        parallel=None,
        sc=None,
    ):
        """Constructor.

        Specify the time budget and start time of the PySpark job, and specify how to cancel them.

        Args:
            Args relate to monitoring:
                start_time: float | The start time of the PySpark job.
                time_budget_s: float | The time budget of the PySpark job in seconds.
                force_cancel: boolean, default=False | Whether to forcely cancel the PySpark job if overtime.

            Args relate to how to cancel the PySpark job:
            (Only one of the following args will work. Priorities from top to bottom)
                cancel_func: function | A function to cancel the PySpark job.
                parallel: joblib.parallel.Parallel | Specify this if using joblib_spark as a parallel backend. It will call parallel._backend.terminate() to cancel the jobs.
                sc: pyspark.SparkContext object | You can pass a specific SparkContext.

                If all three args is None, the monitor will call pyspark.SparkContext.getOrCreate().cancelAllJobs() to cancel the jobs.


        """
        self._time_budget_s = time_budget_s
        self._start_time = start_time
        self._force_cancel = force_cancel
        self._cancelled_error = False
        # TODO: add support for non-spark scenario
        if self._force_cancel and _have_spark:
            self._monitor_daemon = None
            self._finished_flag = False
            self._cancel_flag = False
            self.sc = None
            if cancel_func:
                self.__cancel_func = cancel_func
            elif parallel:
                self.__cancel_func = parallel._backend.terminate
            elif sc:
                self.sc = sc
                self.__cancel_func = self.sc.cancelAllJobs
            else:
                self.__cancel_func = pyspark.SparkContext.getOrCreate().cancelAllJobs
            # logger.info(self.__cancel_func)

    def _monitor_overtime(self):
        """The lifecycle function for monitor thread."""
        if self._time_budget_s is None:
            self.__cancel_func()
            self._cancel_flag = True
            return
        while time.time() - self._start_time <= self._time_budget_s:
            time.sleep(0.01)
            if self._finished_flag:
                return
        self.__cancel_func()
        self._cancel_flag = True
        return

    def _setLogLevel(self, level):
        """Set the log level of the spark context.
        Set the level to OFF could block the warning message of Spark."""
        if self.sc:
            self.sc.setLogLevel(level)
        else:
            pyspark.SparkContext.getOrCreate().setLogLevel(level)

    def __enter__(self):
        """Enter the context manager.
        This will start a monitor thread if spark is available and force_cancel is True.
        """
        if self._force_cancel and _have_spark:
            self._monitor_daemon = threading.Thread(target=self._monitor_overtime)
            # logger.setLevel("INFO")
            logger.info("monitor started")
            self._setLogLevel("OFF")
            self._monitor_daemon.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit the context manager.
        This will wait for the monitor thread to nicely exit."""
        logger.debug(f"monitor exited: {exc_type}, {exc_value}, {exc_traceback}")
        if self._force_cancel and _have_spark:
            self._finished_flag = True
            self._monitor_daemon.join()
            if self._cancel_flag:
                print()
                logger.warning("Time exceeded, canceled jobs")
            # self._setLogLevel("WARN")
            if not exc_type:
                return True
            elif self._cancel_flag and exc_type == py4j.protocol.Py4JJavaError:
                self._cancelled_error = True
                logger.debug("Py4JJavaError Exception: %s", exc_value)
                return True
            elif self._cancel_flag and exc_type == TypeError:
                self._cancelled_error = True
                # When force cancel, joblib>1.2.0 will raise joblib.externals.loky.process_executor._ExceptionWithTraceback
                logger.debug("TypeError Exception: %s", exc_value)
                return True
            else:
                return False
