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
dependency for ordinary AutoML. Without an active run or enabled autologging,
history collection stays in memory: it does not select an experiment, contact
Fabric services, or change the caller's autologging configuration.

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
`flaml[autofe]` includes the ordinary AutoML import dependencies and the
scikit-learn version required for feature engineering, without MLflow or Spark.

Enable tunable feature engineering with `AutoML(featurization="auto")` or
`"force"`. `FLAML_FEATURIZATION` supplies the default when no explicit option is
given; `"off"` always opts out. Custom configuration dictionaries are not yet
supported. Feature engineering is fitted only on training data, separately for
each cross-validation fold, never on the held-out labels. Spark learners, sparse
data, NLP tasks, and ensembles retain their existing AutoFE exclusions.
For the opt-in path, feature preprocessing is also fitted on each training
partition: median imputation, categorical handling, and feature statistics do
not use held-out rows. Each fitted estimator retains that preprocessor for
prediction and export. The ordinary `featurization="off"` preprocessing path is
unchanged.
Reported per-row prediction latency includes validation-time preprocessing and
AutoFE transformation, so `pred_time_limit` accounts for the serving pipeline
without charging feature fitting or training-only resampling.
Rare-class augmentation is likewise applied only to the training partition in
the AutoFE path, through a cloneable fit-only sampler. Impossible singleton-class
splits and missing class coverage are rejected instead of copying observations
across the training/validation boundary. Feature-off augmentation is unchanged.
Selecting AutoFE with an ensemble emits a warning and records the effective
`off` setting rather than silently claiming to apply it.
Registered custom learner names are used for AutoFE selection; unknown learners
use the generic feature space instead of inheriting an unrelated built-in name.
Single-trial fits honor the selected AutoFE mode without requiring explicit
starting points, including list-valued and zero-shot starting points. Warm
starts with `featurization="off"` discard previous `fe.*` choices with a warning;
they do not silently re-enable AutoFE. Integer and mixed-type feature labels are mapped consistently
without collisions, preserving their identity when input columns are reordered.
The same name mapping is applied before sklearn's task preprocessing, including
for mixed string/integer feature labels in AutoML and exported refits.

`AutoML.get_estimator_from_log()` preserves logged `fe.*` choices in an unfitted
scikit-learn pipeline. Fit that pipeline on appropriately preprocessed training
data before prediction: logs contain choices, not fitted task-level preprocessing.
Records without AutoFE retain the existing raw-estimator return behavior.
Featurization supports scikit-learn cloning. An unfitted or failed-refit
transformer raises `NotFittedError` instead of silently passing data through;
successful no-op fits remain valid. Fitted time-series transformers retain their
timestamp column when used directly or inside an exported pipeline.
No-op feature transforms preserve categorical values. Ordinal feature encoding
rejects unsupported mixed categorical types rather than merging distinct values
through string conversion.
Unsupervised feature-only recipes can fit without target labels; supervised LDA
still requires them explicitly.
Changing featurizer parameters invalidates its fitted state. Expected
training-partition incompatibilities, such as variance selection removing every
feature, are logged as rejected infinite-loss AutoML trials so another candidate
can be evaluated; unrelated implementation errors still propagate.
Configuration exports use JSON-compatible column lists and identify the actual
features seen by each selection/extraction stage. Forecast configuration
reconstruction uses task-aware preprocessing and embedded dataset targets.

MLflow pipeline registration remains available as
`flaml.automl.register_automl_pipeline`. MLflow is optional, and registration
requires it when invoked. The registration wrapper is lazy; other AutoML/Tune
integration modules may import MLflow when it is installed. As with the existing pipeline export,
enable AutoFE (`"auto"` or `"force"`) to obtain an `automl_pipeline`.
Registration logs the current fitted
pipeline and registers those exact artifacts, even when model history is disabled.
AutoML restores both global and flavor-specific autologging settings after fitting.
The exported pipeline decodes classification predictions to the original labels,
and each retained trial artifact uses that trial's own fitted featurizer. Caller
autologging settings are restored even when fitting raises.
If restoration also fails, that secondary error is logged without replacing the
original training error. A restoration failure after successful training remains
visible to the caller.
Exported pipelines adapt task preprocessing for scikit-learn refitting and
cross-validation. Forecast prediction preserves caller DataFrames and supports
the fitted estimator's integer-horizon interface. Models fitted with
`retrain_from_log()` can also export and register their fitted AutoFE pipeline.
Exports are detached fitted snapshots: refitting one does not change the source
AutoML instance. Copying a large fitted model requires additional memory.
When AutoFE is enabled, `feature_transformer` is also a detached, refittable
feature-only pipeline with the same preprocessing step names. The lower-level
`model` still expects processed features; use `AutoML.predict()` or the full
exported pipeline for raw inputs. A log-derived unfitted recipe still requires
the appropriately preprocessed data described above.
Configured resampling remains a cloneable, fit-only operation after feature
engineering; prediction and validation rows are never resampled. The same
sample-weight and group-alignment restrictions apply to exported refits.
An early failed refit does not replace the previous label decoder. Exported
classifiers reject non-integral, non-finite, or incorrectly shaped class-index
outputs instead of silently truncating them.
An autologging `log_models=False` setting suppresses automatic model and pipeline
artifacts; explicitly calling `register_automl_pipeline` still registers the requested pipeline.

Tune's centralized startup log contains only configuration counts and boolean
flags, not configuration contents, user tags, metric names, or experiment names.
Plot diagnostics likewise avoid serializing caller-provided values, and
signature/artifact failures send exception types rather than data-bearing
exception text to centralized diagnostics. Detailed failures remain in local
debug logs.
Fabric telemetry and its no-op logger do not depend on an MLflow installation.
Optional reporting or diagnostic-logger initialization failures emit warnings
without stopping model work. Invalid runtime configuration remains an explicit
error; it is not silently ignored.
Logged configuration metadata is rebuilt for each fit from its resolved settings.
History-only Spark searches do not probe MLflow latency or log probe artifacts;
explicit model-artifact opt-outs also apply to logging-enabled probes.

Forecasting uses the same `estimated` exponential-smoothing initialization in
all environments; the internal legacy initialization is not a Fabric default.
For regular fitted Statsmodels models, forecast dates follow the fitted model's
frequency, which can differ from the full dataset's inferred frequency during
cross-validation. Existing public prediction-index and date-gap rules remain
unchanged for irregular series.
Training rows are stably ordered by timestamp before temporal holdout/CV
splitting, preserving feature, target, index, and group alignment. Explicit
forecast request order is not rearranged.
Explicit training and validation partitions are reconstructed separately, so
sorting cannot transfer held-out rows into training. Legacy step-based forecasts
are used only for models with a generated integer index and no saved training
boundary; unrelated prediction errors propagate and exogenous values are retained.

Spark string-to-integer conversion preserves signed 64-bit values, including
large scientific-notation integers, using decimal rather than float rounding.
A full-column check can widen an inferred 32-bit type when sampling misses a
large value. Incompatible finite numeric values raise instead of silently becoming
null; `skip_columns` can retain the original strings.

The fANOVA adapter widens replay distributions when recorded trials contain
values outside the declared search space, preserving compatible log scales and
step grids. This does not mutate the original search space or change subsequent
searches.
Replay expansion also supports Optuna 2.x's legacy logarithmic and quantized
numeric distribution classes.

The public `flaml.visualization` facade exports only its documented plotting
and analysis functions, not implementation modules. Plotly is optional and is
included in `flaml[fabric_python]`.
History-based AutoML plots raise a descriptive error when no trial history was
collected; `model_history=True` retains models, not a separate plot history.
AutoML plots consistently treat `1 - validation_loss` as higher-is-better.
Importance analysis uses the tuning backend's domain converter, including
supported Ray domains, log/quantized distributions, and integer-bound rules.
Optuna's NumPy compatibility adaptation is local to each FLAML evaluator and
does not replace methods on Optuna's process-global tree class.
Slice plots display null/absent categorical values explicitly rather than
failing during trial-count lookup.
Cardinality selection uses fitted categorical provenance, not a name prefix that
a user could also use for an ordinary numeric feature.
Importance analysis keeps explicit categorical `None` choices while excluding
absent conditional parameters. Nested parameter names use the tuning backend's
`/` convention, and an auxiliary metric named `score` cannot replace the
selected objective.
Native Ray tuning results retain their original analysis object and search-space
metadata, and are accepted by the same plotting and importance adapters.
Running-best lines ignore invalid trial scores until a valid observation exists,
and objective coloring does not overwrite a hyperparameter named `score`.
Tune timelines preserve recorded trial start times, including overlap and
scheduling gaps. Histories without timing metadata are explicitly labeled as
estimated sequences. Feature-importance labels come from the fitted learner or
retained transformed feature schema.

Compatibility coverage includes offline NLP tokenization, tiny local
classification-head checkpoints, and bounded AutoML NLP fitting/prediction,
collected by a maintained CI step. Spark
model persistence is tested separately from metadata-only visualization loads;
native save/load requires Hadoop filesystem tooling on Windows.

Fabric demo installation cells and the combined-notebook generator use the
public GitHub source, not internal release-wheel feeds. The `synapse` extra
remains available publicly for demos that need Spark dependencies. Stale
installation outputs have been cleared rather than presented as fresh execution.

The obsolete OpenAI code/math demos using the removed `flaml.oai` namespace are
not included in this consolidation. Use the separately maintained
[AutoGen project](https://github.com/microsoft/autogen) for those workflows.
