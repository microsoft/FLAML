# Fabric runtime behavior

FLAML has a single implementation for public Python installations and Microsoft
Fabric. Installing `flaml` still requires only NumPy; `flaml[automl]` does not
require MLflow, Spark, or Fabric services.

`flaml.fabric.is_fabric_runtime()` detects Fabric's
`MSNOTEBOOKUTILS_RUNTIME_TYPE` notebook marker or Spark runtime context file.
It does not import Spark, start a session, or contact a service. Installing
PySpark or SynapseML alone does not enable Fabric defaults.

| Behavior                            | Outside Fabric                                | In Fabric                                     |
| ----------------------------------- | --------------------------------------------- | --------------------------------------------- |
| `AutoML(model_history=...)` default | `False`                                       | `True`                                        |
| MLflow integration, when installed  | Requires an active run or enabled autologging | Also collects trial history for visualization |
| MLflow tag namespace                | `flaml.*`                                     | `synapseml.flaml.*`                           |
| AutoFE default                      | `off`                                         | `off`, unless configured explicitly           |

Explicit constructor or `fit()` options override these defaults. In particular,
`model_history=False` avoids retaining intermediate models, and
`mlflow_logging=False` disables FLAML's MLflow integration in both `AutoML` and
`tune.run()`. Run logging still follows the user's active-run/autologging setup.
Fabric history collection requires MLflow; it does not make MLflow a mandatory
dependency for ordinary AutoML.

## Explicitly selecting behavior

Set `FLAML_FABRIC_RUNTIME` **before importing FLAML**:

- `auto` (the default): detect the running environment.
- `true` or `1`: enable Fabric defaults explicitly.
- `false` or `0`: preserve public defaults, even inside Fabric.

Values are case-insensitive. Invalid values raise `ValueError`. This switch
selects behavior; it does not emulate Fabric services or configure credentials.

```python
import os

os.environ["FLAML_FABRIC_RUNTIME"] = "false"

from flaml import AutoML
from flaml.fabric import is_fabric_runtime

assert not is_fabric_runtime()
automl = AutoML(model_history=False, mlflow_logging=False)
```

## Optional features

`flaml[fabric_python]` includes plotting and MLflow dependencies for Python
workloads; `flaml[synapse]` additionally includes Spark-related dependencies.
These optional extras are not required for a normal `flaml[automl]` installation.
Internal Fabric delivery infrastructure can consume the same public source
without maintaining a separate library fork.

Enable tunable feature engineering with `AutoML(featurization="auto")` or
`"force"`. `FLAML_FEATURIZATION` supplies the default when no explicit option is
given; `"off"` always opts out. Custom configuration dictionaries are not yet
supported. Feature engineering is fitted only on training data, separately for
each cross-validation fold, never on the held-out labels. Spark learners, sparse
data, NLP tasks, and ensembles retain their existing AutoFE exclusions.

MLflow pipeline registration remains available as
`flaml.automl.register_automl_pipeline`; MLflow is imported when it is called,
not merely when `AutoML` is imported. As with the existing pipeline export,
enable AutoFE (`"auto"` or `"force"`) to obtain an `automl_pipeline`.
Registration logs the current fitted
pipeline and registers those exact artifacts, even when model history is disabled.
AutoML restores both global and flavor-specific autologging settings after fitting.
An autologging `log_models=False` setting suppresses automatic model and pipeline
artifacts; explicitly calling `register_automl_pipeline` still registers the requested pipeline.
