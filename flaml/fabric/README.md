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
without charging feature fitting, training-only resampling, or transformation
of a forecast dataset's training-history partition. Empty validation partitions
do not contribute a per-row transform latency.
Rare-class augmentation is likewise applied only to the training partition in
the AutoFE path, through a cloneable fit-only sampler. Automatically generated
singleton-class splits and missing class coverage are rejected instead of copying
observations across the training/validation boundary. Caller-provided validation
is used as supplied, without augmentation or class-coverage repair.
Time and group holdouts must also represent every advertised class in training,
so probability columns cannot silently disagree with `classes_`.
AutoFE trial samples must retain that vocabulary too; increase `min_sample_size`
or disable sampling if a time-ordered prefix omits a class.
Coverage is also checked on each actual CV training fold and after fit-only
resampling, including exported refits, before the learner is trained.
Classification resamplers must preserve the training vocabulary even with AutoFE
disabled or in standalone pipelines; regression resampling may change target values.
Generic resamplers are rejected for forecasting and panel tasks, including
exported refits, before they can receive embedded validation data. Their row-based
interface does not define safe time-order or panel-group semantics.
Feature-off augmentation is unchanged.
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
Statsmodels-backed integer forecasts follow the fitted training boundary,
including in exported pipelines. Public naive and seasonal-naive lag semantics are retained.
Panel time indexes use steps for regular non-month-start frequencies and retain
gaps for irregular observations; nullable timestamps and DST folds stay distinct.
No-op feature transforms preserve categorical values. Ordinal feature encoding
rejects unsupported mixed categorical types rather than merging distinct values
through string conversion.
Unsupervised feature-only recipes can fit without target labels; supervised LDA
still requires them explicitly.
Changing featurizer parameters invalidates its fitted state. Expected
training-partition incompatibilities, such as variance selection removing every
feature, are logged as rejected infinite-loss AutoML trials so another candidate
can be evaluated; unrelated implementation errors still propagate.
Configuration exports use JSON-compatible column lists. Categorical and numeric
lists select transformed columns; selection lists record columns dropped by the
original fit, and extraction lists record its input features. Selection and
extraction recipes are refitted on the complete surviving feature matrix, even
when the original selector dropped no columns. Forecast configuration
reconstruction uses task-aware preprocessing and embedded dataset targets.
New forecast training logs retain the evaluation mode and training boundary,
including the single-series end timestamp and its UTC offset when present.
Default sampled replay uses that boundary rather than choosing a new split from
the replay time budget; `train_full=True` fits the supplied training data without
constructing an unused validation split. For legacy sampled logs without boundary
metadata, pass the original `eval_method` explicitly when the split is ambiguous.
An embedded validation partition remains an explicit holdout for legacy replay.
Panel logs record whether automatic holdout preparation applied group/time
ordering. Replay reproduces that choice before tail sampling; explicit-validation
and full-data refits retain their original training-row order.

Forecast AutoFE fits feature statistics and target encoders on each training
partition. Explicit validation rows are never eligible for final refitting;
validation-only target classes, including numeric labels, raise an error instead
of changing training codes. Exported pipelines translate dataset target names to
their fitted internal names before normalizing feature names. Generated extraction
columns cannot collide with timestamps or targets; their fitted names are reused
by transforms and `get_feature_names_out()`.
Named external targets participate in that namespace. Empty partitions inherit
the fitted transformed schema without running transforms on zero rows.
Panel/TFT feature engineering is excluded so group identifiers, time indexes, and
named covariates are not removed by generic feature transforms.
Panel forecasting chooses holdout for automatic evaluation; an explicit panel CV
request is rejected rather than using a non-group-aware row splitter. Declared
group/covariate columns and caller-supplied absolute integer time indexes are
preserved through search, final refitting, and prediction. Panel indexes must be
signed integers without missing values, as required by TFT.
Numeric group keys are not imputed or rounded through a lossy floating-point conversion.
Matching group dtypes and categorical ordering are preserved. Mixed integer keys
use a lossless signed or unsigned dtype where possible, and object storage only
when a common numeric representation would lose identity.
The same structural protection applies when refitting an exported panel pipeline.
An explicit `estimator__group_ids` refit override is propagated to preprocessing
so renamed group columns retain the same protection.
Custom dataset time-index names are retained. TFT prediction supports consecutive
coordinates immediately after each group's training boundary, up to the fitted
horizon; unsupported gaps are rejected rather than relabeled. Predictions are
mapped back to the caller's group/time keys, key dtypes, and row order.
Accepted string timestamp keys are retained in that returned index while
normalized timestamps are used internally.
Shorter or unequal group horizons retain the latest encoder observations and the
fitted lag-feature order. Irregular calendars require explicit consecutive indexes
and strictly increasing future timestamps. An empty validation partition produces
no forecasts; TFT's unsupported in-sample training metric is omitted with a warning
when `log_training_metric=True`, without suppressing validation or weakening the
future-coordinate checks.
Exact integer group values are preserved when request and training integer dtypes
differ. Staggered-origin prediction cohorts partition the panel once rather than
repeatedly scanning every group's history.

MLflow pipeline registration remains available as
`flaml.automl.register_automl_pipeline`. MLflow is optional, and registration
requires it when invoked. The registration wrapper is lazy; other AutoML/Tune
integration modules may import MLflow when it is installed. As with the existing pipeline export,
enable AutoFE (`"auto"` or `"force"`) to obtain an `automl_pipeline`.
Registration logs the current fitted
pipeline and registers those exact artifacts, even when model history is disabled.
Each registration uses its own nested run, so registering another pipeline under
the same caller run cannot overwrite an older registered model version.
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
Unsupervised forecast feature-only refits do not require targets and return
feature DataFrames with caller row order and timestamps retained. LDA still
requires training labels, which may be embedded in a `TimeSeriesDataset`.
Detached feature refits propagate the resolved timestamp name between stages.
Full forecast pipeline refits preserve named targets
and an existing explicit timestamp column even when it is not the first column.
Separate external targets use a collision-free internal name when a regressor
has the old target name; that regressor is not removed.
Refitting a numeric-target forecast classifier with strings also learns the new
prediction vocabulary instead of returning encoded class indexes.
Configured resampling remains a cloneable, fit-only operation after feature
engineering; prediction and validation rows are never resampled. The same
sample-weight and group-alignment restrictions apply to exported refits.
Resampling refits use traditional `step__parameter` fit inputs; routing
row-aligned metadata through the resampling operation is not supported.
With a target encoder or resampler, use `fit()` followed by `predict()` instead
of `fit_predict()`, which would bypass those operations in scikit-learn.
An early failed refit does not replace the previous label decoder. Exported
classifiers reject non-integral, non-finite, or incorrectly shaped class-index
outputs instead of silently truncating them.
An autologging `log_models=False` setting suppresses automatic model and pipeline
artifacts; explicitly calling `register_automl_pipeline` still registers the requested pipeline.
Tune preserves the caller's local autolog configuration and propagates those
options to remote workers without permanently reconfiguring reused workers.
Remote logging uses the driver's tracking URI, which must be reachable from the
workers. Each worker trial uses an isolated fluent run stack; unrelated worker runs
are hidden while the objective executes and restored without being ended. Manual
objectives may create their own fluent runs; FLAML's manual result recording remains
driver-owned. Unclosed trial-created runs are ended before restoring the worker.
Worker restoration includes the raw tracking-URI override, tracking environment,
active experiment and previous run stack, so inherited settings remain inherited.
Parallel trial/run
matching uses trial identities rather than task start order.
Native-Ray MLflow logging in AutoML and Tune requires transactional tracking, such as
`mlflow.set_tracking_uri("sqlite:///tracking.db")` or a tracking server.
Relative SQLite filenames are resolved against the driver's directory before
worker execution; SQLite URI options and the caller's tracking settings are retained.
Use a tracking server for multi-host Ray clusters; in-memory databases cannot
share tracking state across worker processes.
File-based tracking is rejected before worker execution because Ray can stop a
worker during FileStore's non-atomic metadata rewrite. Non-Ray file tracking and
history-only collection remain available. For an explicitly identified or uniquely matched owned
attempt that Ray has completed, the driver reconciles a still-running status;
it retains an observed terminal status and does not finalize unrelated runs.
Tagged trial association does not depend on worker clock agreement. Local, Spark,
and native Ray results use the winning trial ID. Local and Ray report-style
objectives retain their final reported numeric metrics; Spark worker reports are
captured per trial and delivered when that objective returns, rather than streamed
to the driver scheduler. Nested tuning has a separate reporting context.
Spark results and Ray returned metrics, generator yields, and `flaml.tune.report`
also carry the accepted worker attempt's MLflow run ID.
Only that attempt is adopted as the winner; other completed attempts retain their
diagnostics. Direct native Ray reporters bypass FLAML's attempt-ID transport:
native analysis results remain available, and a unique owned run can be associated.
If an attempt ID is absent and several runs match a trial, adoption warns, retains
all attempts, and leaves `best_run_id=None` rather than guessing. Unambiguous sibling
trials are still adopted. Use returned metrics or `flaml.tune.report` when
accepted-attempt association must be guaranteed.
Known accepted run IDs are checked directly against experiment, owner and trial
identity. Fallback uniqueness checks include all owned result pages, not only a
truncated latest-run window.
Framework bookkeeping,
config values, strings, and booleans are not logged as objective metrics.
User reports are captured before optimizer callbacks add constraint annotations;
a user-authored metric with a similar name is not removed by suffix guessing.
Callbacks use separate working results, including nested report containers, so
public trial/analysis metrics remain unchanged. Opaque model/artifact payloads
are not deep-copied; completed AutoML callbacks release cached estimator payloads.
Generated elapsed time is included in analysis and tracking when it is the selected
objective. Spark measures it on the worker at the accepted report or return,
not while the driver waits for sibling trials.
An explicitly selected objective such as `time_total_s` is retained even when its
name is also used for framework bookkeeping. `training_iteration` is reserved for
framework progress and cannot be the primary objective, including when a
preconfigured searcher refuses an analysis-metric override.
Errored trials are not recorded as empty successes. Successful adoption writes each trial's metrics
once, and secondary cleanup errors do not replace an objective failure.
Remote trials retain their driver's lineage without ending an unrelated active
worker run. Objective and restoration `RuntimeError`s are not retried as cluster
failures.
Child-run adoption requires an integration-owned tag; unrelated runs in the same
experiment are not changed. Adoption never deletes runs, including empty runs and
duplicate cross-validation runs; duplicate bookkeeping is skipped without deleting
stored data. This avoids deleting annotations or artifacts written after a metadata
check. Returning `{}` marks trial-owned runs failed,
including unclosed nested runs, and uses Ray's normal trial-failure handling when
running natively. An errored trial's
intermediate score cannot displace a successful winner or become a successful
Optuna observation or CFO starting point. Failed trials remain in analysis history,
including native-Ray-backed analyses. Raised local objectives notify their searcher
of failure before propagating the original error. Clean scheduler stops preserve accepted
reports and finish their owned runs without aborting sibling trials. Native Ray
generator functions retain their yield semantics and trial context.
Provisional local-search updates remain enabled. If their trial fails, its
observations are removed and the search center is recovered from surviving
observations, including concurrent successful trials. Scalar rollback retains only
the best committed observation and unfinished trials. Clean local stops must match
the active local trial; clean Ray exits require an
explicit active Ray context; Spark worker exits are not inferred from module
defaults. A caller's already-handled exception is not an objective failure.
As in native Ray, a regular callable returning a generator is not a generator
trainable: use a generator function instead. Such unsupported returns fail inside
the owned tracking context rather than finishing an empty run successfully.
Ray 1.13's [upstream `with_parameters` wrapper](https://github.com/ray-project/ray/blob/ray-1.13.0/python/ray/tune/utils/trainable.py#L399-L425)
discards function return values.
Use report-style objectives with that wrapper, or a return-preserving closure
for returned metrics and the `{}` failure signal. FLAML does not inspect or
rewrite a third-party callable's closure to recover a discarded return value.
Trial artifacts use that trial's fitted model, and
fit completion waits for dependent artifact-environment writes.
Fluent model writes stay on the run-owning thread; only client-addressed metadata
and requirement updates are asynchronous. The parent artifact is published after
the final refit or ensemble is selected, without ending the caller's run.
A refused final artifact write remains a visible failure rather than being
marked published. Completed model-write results are observed and released once.
Opaque per-estimator fit arguments are reported by argument name only, not by
value; credentials and training payloads are not configuration metadata.
The metric-name tag retains its historical spelling: `flaml.meric` publicly and
`synapseml.flaml.meric` in Fabric. Consumers should use that key, not `*.metric`.

Tune's centralized startup log contains only configuration counts and boolean
flags, not configuration contents, user tags, metric names, or experiment names.
Plot diagnostics likewise avoid serializing caller-provided values, and
signature/artifact failures send exception types rather than data-bearing
exception text to centralized diagnostics. Detailed failures remain in local
debug logs.
Fabric telemetry and its no-op logger do not depend on an MLflow installation.
Optional usage-telemetry or diagnostic-logger initialization failures emit warnings
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
Finite fractions missed by the sample widen to double only when the full column
supports that conversion. Near-integral literals are not rounded into integers,
and decimal round-trip checks preserve strings when either ordinary or large
fractional values would collapse to the same double.

`broadcast_code` accepts a non-reserved Python module name and publishes immutable
code revisions through Spark's file distribution. Reimport the public module on
the driver after publishing changed code; newly imported classes carry the new
revision identity while existing class references retain the old implementation.
Cached worker aliases resolve the newest visible publication, including code
reverts. Publication ordering is independent of the archive order Spark supplies
to late workers; immutable revision modules still back existing pickles.
Package API, metadata, and generated-revision names are reserved. Archive modules
live under a private prefix rather than shadowing top-level imports. Fallible
local alias and bytecode preparation precedes distribution, so preparation
failures do not distribute a new revision.
A distribution failure restores the previous driver alias; if restoring local
state also fails, that secondary failure is logged without masking the original.
Local commitment is recorded atomically in `.flaml-code-state`, with writers
serialized across processes. A prepared alias file alone cannot establish a new
revision, including after a driver restart. Imports execute after releasing the
publication lock, avoiding lock-order inversion with Python's module loader.
The publishing driver's failed initial alias remains unavailable, and rollback
invalidates generated bytecode. Worker archives can supersede an older copied
installation. Distribution is not a cluster-wide transaction: if state publication
or a distribution acknowledgement fails after possible delivery, artifacts are
retained for recovery and the committed local state determines the binding.
A later successful publication, including a revert, advances past every retained
archive sequence even after a driver restart or clock rollback. Retry a failed publication
before submitting work that depends on it; do not assume every worker rolled back.
Publication regressions use test-owned temporary driver namespaces, including
failure paths, so generated code cannot contaminate subsequent package builds.
As before, the driver package directory must be writable. The returned source
path is revision-specific; workers do not need writable package installations.

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
Native Ray tuning returns a Ray-compatible analysis object with search-space
metadata, accepted by the same plotting and importance adapters. It remains an
instance of Ray's analysis type; identity with Ray's original object is not guaranteed.
Zero-shot class counting retains distributed label operations without importing
Spark for ordinary local inputs.
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
