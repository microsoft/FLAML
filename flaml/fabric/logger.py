import re

from flaml.fabric import is_fabric_runtime

try:
    from synapse.ml.pymds import get_mds_logger
    from synapse.ml.pymds.handler import default_scrubbers
    from synapse.ml.pymds.scrubbers.scrubber import IScrub

    if not is_fabric_runtime():
        raise ImportError("Not running in Fabric runtime")
except ImportError:
    no_synapse = True
else:
    no_synapse = False

_KUSTO_TABLE_NAME = "SynapseMLLogs"

if no_synapse:

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

    def init_kusto_logger(logger_name: str = ""):
        return kusto_logger

else:

    class ArtifactNameScrubber(IScrub):
        pattern = re.compile("Artifact \\S+", re.IGNORECASE)
        mask = "[artifact name redacted]"

        def scrub(self, msg: str) -> str:
            return re.sub(self.pattern, self.mask, msg)

    def init_kusto_logger(logger_name: str = ""):
        if not logger_name:
            logger_name = __name__
        return get_mds_logger(
            logger_name,
            tableName=_KUSTO_TABLE_NAME,
            scrubbers=[*default_scrubbers, ArtifactNameScrubber()],
        )
