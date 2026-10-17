import logging
import time
from numbers import Number
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import BaseEstimator as SKLearnBaseEstimator
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectorMixin, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

from flaml import tune
from flaml.automl.data import _make_column_name_mapping
from flaml.automl.model import BaseEstimator
from flaml.automl.task.task import Task
from flaml.automl.time_series.ts_data import TimeSeriesDataset
from flaml.fabric.logger import init_kusto_logger

logger = logging.getLogger(__name__)
kusto_logger = init_kusto_logger("flaml.autofe")


class AutoFEIncompatibleDataError(ValueError):
    """A valid AutoFE candidate cannot fit this training partition."""


class _PartitionAwareLDA(LinearDiscriminantAnalysis):
    def fit(self, X, y):
        values, labels = np.asarray(X), np.asarray(y)
        if values.ndim == 2 and labels.ndim == 1 and len(values) == len(labels):
            groups = [values[labels == label] for label in np.unique(labels)]
            if groups and all(len(group) and np.all(group == group[0]) for group in groups):
                raise AutoFEIncompatibleDataError("LDA requires within-class feature variation.")
        return super().fit(X, y)


def get_transformer(stage: str, method: str) -> TransformerMixin:
    """
    Get the transformer object based on the specified stage and method.

    Args:
        stage: The stage of the transformation.
        method: The method of the transformation.

    Returns:
        transformer: The transformer object, should be a sklearn.base.TransformerMixin object.

    Raises:
        ValueError: If the specified stage or method is not available.
    """

    if stage not in available_methods:
        raise ValueError(f"Unknown stage {stage}")
    methods = available_methods[stage]

    if method is None:
        return None
    elif method not in methods:
        raise ValueError(f"Unknown method {method}")

    transformer_class = methods[method]
    if transformer_class is not None:
        return transformer_class["class"](**transformer_class["args"])
    else:
        return None


def _to_search_space(methods: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """
    Convert a dictionary of methods into a FLAML search space .

    Args:
        methods: A dictionary mapping stages to choices.

    Returns:
        search_space: a FLAML search space.

    """
    res = {}
    for stage, choices in methods.items():
        res["fe." + stage] = {"domain": tune.choice(choices), "init_value": choices[0]}
    return res


def parse_autofe_config(
    raw_config: str,
    data: Any,
    task: Task,
    learner_class: BaseEstimator,
    estimator_name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Handle the autofe config.

    Args:
        raw_config: One of "off", "auto", or "force". Custom dictionaries are not supported.
        data: Training data. Could be any type that flaml supports.
        task: The flaml Task object, to determine what methods are available.
        learner_class: The flaml Estimator class, to determine what methods are available.
        estimator_name: The registered learner name, including custom learners.
            If omitted, infer an exact built-in match; unmatched learners use the generic feature space.

    Raises:
        ValueError: Unsupported config.

    Returns:
        search_space: a FLAML search space.
    """
    empty_search_space = _to_search_space({})
    if raw_config == "off":
        return empty_search_space
    if issparse(data):
        logger.warning("Auto featurization is not supported for sparse data. Featurization is turned off.")
        return empty_search_space

    if task.is_nlp():
        logger.warning("Auto featurization is not supported for NLP task. Featurization is turned off.")
        return empty_search_space

    if estimator_name is None:
        estimator_name = next(
            (name for name, estimator_class in task.estimators.items() if estimator_class == learner_class), None
        )

    if estimator_name is not None and estimator_name.endswith("_spark"):
        logger.warning("Auto featurization is not supported for spark data. Featurization is turned off.")
        return empty_search_space

    tree_based_estimators = {
        "xgboost",
        "xgb_limitdepth",
        "rf",
        "lgbm",
        "catboost",
        "extra_tree",
        "lgbm_spark",
        "rf_spark",
        "gbt_spark",
    }

    fe_search_space = {
        "selection": ["null", "cardinality", "variance"],
        "categorical": ["ordinal"],
        "extraction": ["null", "PCA", "LDA"] if task.is_classification() else ["null", "PCA"],
        "numerical": [
            "null",
            # "bucket",  # or "bin"?
            "scaler_standard",
            "scaler_minmax",
            "scaler_maxabs",
            "scaler_robust",
            "normalizer_sparse",
        ],
    }

    if estimator_name in tree_based_estimators:
        del fe_search_space["numerical"]

    if raw_config == "auto":
        return _to_search_space(fe_search_space)
    elif raw_config == "force":
        for stage in fe_search_space:
            if "null" in fe_search_space[stage]:
                fe_search_space[stage].remove("null")
        return _to_search_space(fe_search_space)
    else:
        raise ValueError(f"Unsupported AutoFE config {raw_config}")

    """TODO: will support custom config in the future. The code I wrote before:

    def _handle_argument(values, stage, checkset):

        if values is None:
            logger.warning(f"Missing value 'methods' in stage {stage}, using all method to tune.")
            return None
        if values == "auto":
            return None
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            raise ValueError(f"Expected methods for stage {stage} to be a string or a list of string, got {values}")
        for value in values:
            if value not in checkset:
                raise ValueError(f"Unsupported {value} in stage {stage}")
        return values

    for stage, conf in raw_config.items():
        if not (stage in fe_search_space or stage == "numerical"):
            raise ValueError(f"Unknown stage {stage}, current supported stages are {list(fe_search_space.keys())}")
        if not isinstance(conf, dict):
            raise ValueError(f"Expected config for stage {stage} to be a dict or 'auto', got {conf}")

        methods = _handle_argument(conf.get("methods", None), stage, fe_search_space[stage])
        if methods is None:
            continue
        else:
            fe_search_space[stage] = methods

        if isinstance(data, pd.DataFrame):
            avaliable_cols = data.columns
        columns = _handle_argument(conf.get("columns", None), stage, avaliable_cols)
        if columns is None:
            continue
        else:
            column_space[stage] = columns

    return _to_search_space(fe_search_space, column_space)
    """


class Featurization(TransformerMixin, SKLearnBaseEstimator):
    """A class to implement the featurization pipeline."""

    def __init__(
        self,
        params: Optional[Dict] = None,
        task: Optional[Task] = None,
        config: Optional[List[Dict]] = None,
    ):
        """Init the Featurization class.

        Args:
            params: Init based on a hyperparameter config.
            task: The flaml Task object, to determine implementation detail.
            config: Init based on a config.

        Raises:
            ValueError: If neither params nor config is provided.
        """
        self._is_fitted = False
        self.time_col_ = None
        if params is None and config is None:
            raise ValueError("Either params or config should be provided")
        self._init_params = params
        self._init_config = config
        self.pipeline = None
        self.flaml_transformer = None
        self.task = task
        if config is None:
            self._config = {k.replace("fe.", ""): {"method": v, "columns": "auto"} for k, v in params.items()}
        else:
            if task is None:
                raise ValueError("Task should be provided when reconstruct.")
            self._config = self.standalone_init(config)
        self.detail_config = []
        self.ts_dataset = None
        self._input_columns = None
        self._column_mapping = {}

    def get_params(self, deep=True):
        """Return initialization arguments rather than learned export configuration."""
        return {"params": self._init_params, "task": self.task, "config": self._init_config}

    def set_params(self, **params):
        if not params:
            return self
        current = self.get_params(deep=False)
        unknown = params.keys() - current.keys()
        if unknown:
            raise ValueError(f"Invalid Featurization parameter(s): {sorted(unknown)}")
        current.update(params)
        self.__init__(**current)
        return self

    @property
    def config(self) -> List[Dict]:
        """Get the config of the featurization pipeline. Could be use for reconstruct. Also an alias for self.detail_config"""
        conf = self.detail_config
        if len(conf) == 0:
            logger.warning("You have to fit the model first to get complete config")
        return conf

    @property
    def params(self) -> Dict:
        """Get the hyperparameter config of the featurization pipeline. Could be use for reconstruct."""
        return {f"fe.{k}": v["method"] for k, v in self._config.items()}

    def standalone_init(self, config: List[Dict]) -> Dict:
        """Init function for reconstruct.

        Args:
            config: The config to init the Featurization class.

        Raises:
            ValueError: Unknown stage or method.

        Returns:
            config: The inner config inside the Featurization class.
        """
        from flaml.automl.data import DataTransformer
        from flaml.automl.pipeline import _TaskPreprocessor

        self.flaml_transformer = _TaskPreprocessor(self.task, DataTransformer(), preserve_features=True)
        res_config = {}
        for stage_config in config:
            if stage_config["stage"] not in available_methods:
                raise ValueError(f"Unknown stage {stage_config['stage']}")
            if stage_config["method"] not in available_methods[stage_config["stage"]]:
                raise ValueError(f"Unknown method {stage_config['method']} for stage {stage_config['stage']}")

            res_config[stage_config["stage"]] = {"method": stage_config["method"], "columns": stage_config["columns"]}
        return res_config

    def static_preprocess(self, X, y, *, fit=False):
        """General static preprocess function before all featurization stages.

        Args:
            X: Training data.
            y: Training label.
            fit: Whether to learn a stable input-column mapping.

        Returns:
            X: Processed training data.
            y: Processed training label.
            categorical_features: a list of names of categorical features in the X.
            numerical_features: a list of names of numerical features in the X.
        """
        if X is None:
            return None, None, None, None
        if not isinstance(X, pd.DataFrame):
            columns = None if fit else getattr(self, "_input_columns", None)
            X = pd.DataFrame(X, columns=columns)
            if columns is None:
                X.columns = [str(c) for c in X.columns]
        if fit:
            if not X.columns.is_unique:
                raise ValueError("AutoFE requires unique input column labels.")
            self._input_columns = X.columns.tolist()
            self._column_mapping = _make_column_name_mapping(X.columns)
        input_columns = getattr(self, "_input_columns", None)
        if input_columns is not None:
            if not X.columns.is_unique or set(X.columns) != set(input_columns):
                raise ValueError("AutoFE input columns must match the fitted columns.")
            X = X.loc[:, input_columns]
        X = X.rename(columns=getattr(self, "_column_mapping", {})).copy()
        categorical_features = list(X.select_dtypes(include=["category"]).columns)
        numerical_features = list(X.select_dtypes(exclude=["category"]).columns)
        if self._config.get("categorical", {}).get("method") not in (None, "null"):
            for col in categorical_features:
                kinds = {
                    "string"
                    if isinstance(value, str)
                    else "number"
                    if isinstance(value, (Number, bool, np.bool_))
                    else type(value)
                    for value in X[col].dropna().unique()
                }
                if len(kinds) > 1:
                    raise AutoFEIncompatibleDataError(
                        f"Cannot ordinal-encode mixed categorical types in {col!r} without losing value identity."
                    )
        return X, y, categorical_features, numerical_features

    def build_transformer(self, stage: str, features: List[str]) -> Tuple[TransformerMixin, List[str], Dict]:
        """
        Build a transformer object based on the specified stage and features.

        Args:
            stage: The stage of the transformation.
            features: The list of features to be transformed.

        Returns:
            encoder: The transformer object, should be a sklearn.base.TransformerMixin object.
            features: The list of features to be transformed.
            detail_config: The config of the transformer object.

        """
        if stage not in self._config or features is None:
            return None, None, None
        config = self._config[stage]
        encoder = get_transformer(stage, config["method"])
        columns = config.get("columns", "auto")
        if not isinstance(columns, str) or (columns != "auto"):
            columns = {self._column_mapping.get(column, column) for column in columns}
            features = [feature for feature in features if feature in columns]
        if encoder is not None and len(features) > 0:
            encoder.set_output(transform="pandas")
            detail_config = {
                "stage": stage,
                "method": config["method"],
                "columns": features,
            }
            return encoder, features, detail_config
        else:
            return None, None, None

    def fit(self, X, y=None):
        """Fit the featurization pipeline."""
        _st = time.time()
        kusto_logger.info(f"Start featurization pipeline fitting at timestamp {_st}")
        self.pipeline = None
        self.detail_config = []
        self._is_fitted = False
        self._input_columns = None
        self._column_mapping = {}
        detail_config = []
        transformers = []
        column_transformers = []
        if self.flaml_transformer is not None:
            X = self.flaml_transformer.fit_transform(X, y)
        self.time_col_ = X.time_col if isinstance(X, TimeSeriesDataset) else None

        if isinstance(X, TimeSeriesDataset):
            y = X.train_data[X.target_names]
            X = X.train_data.drop(columns=X.target_names + [X.time_col])

        X, y, categorical_features, numerical_features = self.static_preprocess(X, y, fit=True)
        categorical_inputs = list(categorical_features)

        categorical_encoder, categorical_features, categorical_detail_config = self.build_transformer(
            "categorical", categorical_features
        )
        if categorical_encoder is not None:
            column_transformers.append(("categorical", categorical_encoder, categorical_features))
            detail_config.append(categorical_detail_config)

        numerical_encoder, numerical_features, numerical_detail_config = self.build_transformer(
            "numerical", numerical_features
        )
        if numerical_encoder is not None:
            column_transformers.append(("numerical", numerical_encoder, numerical_features))
            detail_config.append(numerical_detail_config)

        if len(column_transformers) > 0:
            column_transformer = ColumnTransformer(column_transformers, remainder="passthrough")
            column_transformer.set_output(transform="pandas")
            transformers.append(("transformer", column_transformer))

        feature_selector, _, feature_selector_detail_config = self.build_transformer("selection", X.columns)
        if feature_selector is not None:
            if isinstance(feature_selector, CardinalitySelector):
                if column_transformers:
                    encoded = set(categorical_features or [])
                    categorical_outputs = [
                        f"{'categorical' if column in encoded else 'remainder'}__{column}"
                        for column in categorical_inputs
                    ]
                else:
                    categorical_outputs = categorical_inputs
                feature_selector.set_params(categorical_features=categorical_outputs)
            transformers.append(("selection", feature_selector))

        feature_extractor, _, feature_extractor_detail_config = self.build_transformer("extraction", X.columns)
        if feature_extractor is not None:
            transformers.append(("extraction", feature_extractor))

        if len(transformers) == 0:
            self._is_fitted = True
            return self

        pipeline = Pipeline(transformers)
        if isinstance(feature_extractor, LinearDiscriminantAnalysis) and y is None:
            raise ValueError("LDA requires target labels.")
        if isinstance(feature_extractor, LinearDiscriminantAnalysis) and y is not None:
            class_count = len(np.unique(np.asarray(y)))
            if class_count < 2 or len(y) <= class_count:
                raise AutoFEIncompatibleDataError("LDA needs at least two classes and more samples than classes.")
        try:
            pipeline.fit(X, y)
        except ValueError as exc:
            variances = getattr(feature_selector, "variances_", None)
            if (
                isinstance(feature_selector, VarianceThreshold)
                and variances is not None
                and not np.any(variances > feature_selector.threshold)
            ):
                raise AutoFEIncompatibleDataError("No training features survive variance selection.") from exc
            raise

        keep_cols = list(X.columns)
        for stage, transformer in pipeline.steps:
            if stage == "transformer":
                keep_cols = list(transformer.get_feature_names_out())
            elif stage == "selection":
                support = transformer.get_support()
                drop_cols = [column for column, keep in zip(keep_cols, support) if not keep]
                keep_cols = [column for column, keep in zip(keep_cols, support) if keep]
                feature_selector_detail_config["columns"] = self._unprefix_columns(drop_cols, pipeline)
                detail_config.append(feature_selector_detail_config)
            elif stage == "extraction":
                feature_extractor_detail_config["columns"] = self._unprefix_columns(keep_cols, pipeline)
                detail_config.append(feature_extractor_detail_config)
        self.pipeline = pipeline
        self.detail_config = detail_config
        self._is_fitted = True
        kusto_logger.info(f"Featurization pipeline fitting finished in {time.time() - _st} seconds")
        return self

    @staticmethod
    def _unprefix_columns(columns, pipeline):
        column_transformer = pipeline.named_steps.get("transformer")
        prefixes = (
            tuple(f"{name}__" for name, _, _ in column_transformer.transformers_)
            if column_transformer is not None
            else ()
        )
        return [
            column.split("__", 1)[1] if isinstance(column, str) and column.startswith(prefixes) else str(column)
            for column in columns
        ]

    def _transform(self, X):
        """Transform the data based on the featurization pipeline."""
        if self.flaml_transformer is not None:
            X = self.flaml_transformer.transform(X)
        X, y, categorical_features, numerical_features = self.static_preprocess(X, None)
        if self.pipeline is None or X is None:
            return X
        else:
            raw_res = self.pipeline.transform(X)
            raw_res.columns = self._unprefix_columns(raw_res.columns, self.pipeline)
            return raw_res

    def transform(self, X, time_col=None):
        check_is_fitted(self)
        if time_col is None:
            time_col = getattr(self, "time_col_", None)
        if isinstance(X, TimeSeriesDataset):
            keep_cols = X.target_names + [X.time_col]
            if X.test_data is not None and len(X.test_data) > 0:
                keep_test_data = X.test_data[keep_cols]
                transformed_test_data = self._transform(X.test_data.drop(columns=keep_cols))
                test_data = pd.concat([keep_test_data, transformed_test_data], axis=1)
            else:
                test_data = X.test_data
            keep_train_data = X.train_data[keep_cols]
            transformed_train_data = self._transform(X.train_data.drop(columns=keep_cols))
            train_data = pd.concat([keep_train_data, transformed_train_data], axis=1)
            new_ts_dataset = TimeSeriesDataset(
                train_data=train_data,
                time_col=X.time_col,
                target_names=X.target_names,
                time_idx=X.time_idx,
                test_data=test_data,
            )
            return new_ts_dataset
        elif time_col is not None:
            if isinstance(X, int):  # predict single timestamp
                return X
            keep_data = X[[time_col]]
            transformed_data = self._transform(X.drop(columns=[time_col]))
            return pd.concat([keep_data, transformed_data], axis=1)
        else:
            return self._transform(X)

    def __repr__(self):
        return repr(self.pipeline) if self.pipeline is not None else "Featurization(no pipeline)"

    def _repr_mimebundle_(self, **kwargs):
        """Used in Jupyter notebook to display the featurization pipeline."""
        if self.pipeline is None:
            return {"text/plain": repr(self)}
        return self.pipeline._repr_mimebundle_(**kwargs)

    def show_transformations(self):
        """Print the featurization pipeline."""
        pprint(self.detail_config)

    def __sklearn_is_fitted__(self):
        """sklearn 1.8 hook used by `check_is_fitted` (e.g. when this estimator
        is the last step of a Pipeline). Prefer the explicit flag set by
        `fit()` so that no-op fits (no transformers configured) still register
        as fitted. Falls back to `pipeline is not None` for objects produced
        by older code paths."""
        return getattr(self, "_is_fitted", self.pipeline is not None)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features is not None and list(input_features) != self._input_columns:
            raise ValueError("input_features must match the fitted feature columns.")
        if self.pipeline is None:
            return np.asarray([self._column_mapping.get(name, name) for name in self._input_columns], dtype=object)
        return np.asarray(self._unprefix_columns(self.pipeline.get_feature_names_out(), self.pipeline), dtype=object)


class CardinalitySelector(SelectorMixin, SKLearnBaseEstimator):
    """Class to drop columns with high cardinality."""

    def __init__(self, threshold=0.6, categorical_features=None):
        self.threshold = threshold
        self.categorical_features = categorical_features
        self.support_ = None

    def _get_support_mask(self):
        return self.support_

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy()
        threshold = self.threshold * len(X)
        categorical_features = (
            list(X.select_dtypes(include=["category"]).columns)
            if self.categorical_features is None
            else self.categorical_features
        )
        if any(column not in X for column in categorical_features):
            raise ValueError("Categorical feature metadata references missing columns.")
        high_cardinality_cols = [column for column in categorical_features if X[column].nunique() > threshold]
        self.support_ = ~X.columns.isin(high_cardinality_cols)
        if not self.support_.any():
            raise AutoFEIncompatibleDataError("No training features survive cardinality selection.")
        return self


available_methods = {
    "selection": {
        "null": None,
        "cardinality": {"class": CardinalitySelector, "args": {}},
        "variance": {"class": VarianceThreshold, "args": {}},
    },
    "numerical": {
        "null": None,
        "bucket": {"class": KBinsDiscretizer, "args": {"n_bins": 5, "encode": "ordinal"}},
        "scaler_standard": {"class": StandardScaler, "args": {}},
        "scaler_minmax": {"class": MinMaxScaler, "args": {}},
        "scaler_maxabs": {"class": MaxAbsScaler, "args": {}},
        "scaler_robust": {"class": RobustScaler, "args": {}},
        "normalizer_sparse": {"class": Normalizer, "args": {"norm": "l1"}},
    },
    "categorical": {
        "null": None,
        "ordinal": {"class": OrdinalEncoder, "args": {"handle_unknown": "use_encoded_value", "unknown_value": -1}},
    },
    "extraction": {
        "null": None,
        "PCA": {"class": PCA, "args": {}},
        "LDA": {"class": _PartitionAwareLDA, "args": {}},
    },
}

# Retain the original registry name for existing imports.
avaliable_methods = available_methods
