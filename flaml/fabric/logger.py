import logging
import re

from flaml.fabric import is_fabric_runtime

no_synapse = True
if is_fabric_runtime():
    try:
        from synapse.ml.pymds import get_mds_logger
        from synapse.ml.pymds.handler import default_scrubbers
        from synapse.ml.pymds.scrubbers.scrubber import IScrub
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.split(".")[0] != "synapse":
            raise
        logging.getLogger(__name__).debug("Fabric logging is unavailable: %s", exc)
    else:
        no_synapse = False

_KUSTO_TABLE_NAME = "SynapseMLLogs"


class KustoLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


kusto_logger = KustoLogger()


if not no_synapse:

    class ArtifactNameScrubber(IScrub):
        pattern = re.compile("Artifact \\S+", re.IGNORECASE)
        mask = "[artifact name redacted]"

        def scrub(self, msg: str) -> str:
            return re.sub(self.pattern, self.mask, msg)


def init_kusto_logger(logger_name: str = ""):
    if no_synapse:
        return kusto_logger
    if not logger_name:
        logger_name = __name__
    try:
        return get_mds_logger(
            logger_name,
            tableName=_KUSTO_TABLE_NAME,
            scrubbers=[*default_scrubbers, ArtifactNameScrubber()],
        )
    except Exception:
        logging.getLogger(__name__).warning("Fabric diagnostic logger initialization failed.", exc_info=True)
        return kusto_logger
