import logging
import time
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

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

from flaml import tune
from flaml.automl.model import BaseEstimator
from flaml.automl.task.task import Task
from flaml.automl.time_series.ts_data import TimeSeriesDataset
from flaml.fabric.logger import init_kusto_logger

logger = logging.getLogger(__name__)
kusto_logger = init_kusto_logger("flaml.autofe")


def get_transformer(stage: str, method: str) -> TransformerMixin:
    """
    Get the transformer object based on the specified stage and method.

    Args:
        stage: The stage of the transformation.
        method: The method of the transformation.

    Returns:
        transformer: The transformer object, should be a sklearn.base.TransformerMixin object.

    Raises:
        ValueError: If the specified stage or method is not avaliable.
    """

    if stage not in avaliable_methods:
        raise ValueError(f"Unknown stage {stage}")
    methods = avaliable_methods[stage]

    if method is None:
        return None
    elif method not in methods:
        raise ValueError(f"Unknown method {method}")

    transformer_class = methods[method]
    if transformer_class is not None:
        return transformer_class["class"](**transformer_class["args"])
    else:
        return None


def _to_search_space(methods: Dict[str, str]) -> Dict[str, Dict[str, tune.sample.Categorical]]:
    """
    Convert a dictionary of methods into a FLAML search space .

    Args:
        methods: A dictionary mapping stages to choices.

    Returns:
        search_space: a FLAML search space.

    """
    res = {}
    for stage, chocices in methods.items():
        res["fe." + stage] = {"domain": tune.choice(chocices)}
    return res


def parse_autofe_config(
    raw_config: Union[str, Dict],
    data: Any,
    task: Task,
    learner_class: BaseEstimator,
) -> Dict[str, Dict[str, tune.sample.Categorical]]:
    """Handle the autofe config.

    Args:
        raw_config: Source config to handle, should be a dict or a string.
        data: Training data. Could be any type that flaml supports.
        task: The flaml Task object, to determine what methods are avaliable.
        learner_class: The flaml Estimator class, to determine what methods are avaliable.

    Raises:
        ValueError: Unsupported config.

    Returns:
        search_space: a FLAML search space.
    """
    empty_search_space = _to_search_space({})
    if issparse(data):
        logger.warning("Auto featurization is not supported for sparse data. Featurization is turned off.")
        return empty_search_space

    if task.is_nlp():
        logger.warning("Auto featurization is not supported for NLP task. Featurization is turned off.")
        return empty_search_space

    if raw_config == "off":
        return empty_search_space

    for estimator_name, estimator_class in task.estimators.items():
        if estimator_class == learner_class:
            break

    if estimator_name.endswith("_spark"):
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


class Featurization(SKLearnBaseEstimator, TransformerMixin):
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
        if params is None and config is None:
            raise ValueError("Either params or config should be provided")
        self.pipeline = None
        self.flaml_transformer = None
        if config is None:
            self._config = {k.replace("fe.", ""): {"method": v, "columns": "auto"} for k, v in params.items()}
        else:
            if task is None:
                raise ValueError("Task should be provided when reconstruct.")
            self._config = self.standalone_init(config)
        self.task = task
        self.detail_config = []
        self.ts_dataset = None

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

        self.flaml_transformer = DataTransformer()
        res_config = {}
        for stage_config in config:
            if stage_config["stage"] not in avaliable_methods:
                raise ValueError(f"Unknown stage {stage_config['stage']}")
            if stage_config["method"] not in avaliable_methods[stage_config["stage"]]:
                raise ValueError(f"Unknown method {stage_config['method']} for stage {stage_config['stage']}")

            res_config[stage_config["stage"]] = {"method": stage_config["method"], "columns": stage_config["columns"]}
        return res_config

    def static_preprocess(self, X, y):
        """General static preprocess function before all featurization stages.

        Args:
            X: Training data.
            y: Training label.

        Returns:
            X: Processed training data.
            y: Processed training label.
            categorical_features: a list of names of categorical features in the X.
            numerical_features: a list of names of numerical features in the X.
        """
        if X is None:
            return None, None, None, None
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [str(c) for c in X.columns]
        categorical_features = list(X.select_dtypes(include=["category"]).columns)
        numerical_features = list(X.select_dtypes(exclude=["category"]).columns)
        for col in categorical_features:
            X[col] = pd.Categorical(X[col].astype(str))
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
            features = list(set(features) & set(columns))
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
        detail_config = []
        transformers = []
        column_transformers = []
        if self.flaml_transformer is not None:
            X, y = self.flaml_transformer.fit_transform(X, y, self.task)

        if isinstance(X, TimeSeriesDataset):
            y = X.all_data[X.target_names]
            X = X.all_data.drop(columns=X.target_names + [X.time_col])

        X, y, categorical_features, numerical_features = self.static_preprocess(X, y)

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
            transformers.append(("selection", feature_selector))

        feature_extractor, _, feature_extractor_detail_config = self.build_transformer("extraction", X.columns)
        if feature_extractor is not None:
            transformers.append(("extraction", feature_extractor))

        if len(transformers) == 0:
            return self

        self.pipeline = Pipeline(transformers)
        self.pipeline.fit(X, y)

        keep_cols = X.columns
        for stage, transformer in self.pipeline.steps:
            if stage == "selection":
                drop_mask = transformer.get_support()
                drop_cols = list(X.columns[~drop_mask])
                keep_cols = list(X.columns[drop_mask])
                feature_selector_detail_config["columns"] = drop_cols
                detail_config.append(feature_selector_detail_config)
            elif stage == "extraction":
                feature_extractor_detail_config["columns"] = keep_cols
                detail_config.append(feature_extractor_detail_config)
        self.detail_config = detail_config
        kusto_logger.info(f"Featurization pipeline fitting finished in {time.time() - _st} seconds")
        return self

    def _transform(self, X):
        """Transform the data based on the featurization pipeline."""
        if self.flaml_transformer is not None:
            X = self.flaml_transformer.transform(X)
        X, y, categorical_features, numerical_features = self.static_preprocess(X, None)
        if self.pipeline is None or X is None:
            return X
        else:
            raw_res = self.pipeline.transform(X)
            raw_name = []
            for col in raw_res.columns:
                if not isinstance(col, str):
                    raw_name.append(str(col))
                    continue
                parts = col.split("__")
                if len(parts) == 1:
                    raw_name.append(col)
                else:
                    raw_name.append("__".join(parts[1:]))
            raw_res.columns = raw_name
            return raw_res

    def transform(self, X, time_col=None):
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
        return self.pipeline.__repr__()

    def _repr_mimebundle_(self, **kwargs):
        """Used in Jupyter notebook to display the featurization pipeline."""
        return self.pipeline._repr_mimebundle_(**kwargs)

    def show_transformations(self):
        """Print the featurization pipeline."""
        pprint(self.detail_config)


class CardinalitySelector(SelectorMixin, SKLearnBaseEstimator):
    """Class to drop columns with high cardinality."""

    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.support_ = None

    def _get_support_mask(self):
        return self.support_

    def fit(self, X, y=None):
        threshold = self.threshold * len(X)
        # Find columns with high cardinality
        high_cardinality_cols = [
            col for col in X.columns if X[col].nunique() > threshold and (col.startswith("categorical__"))
        ]
        self.support_ = ~X.columns.isin(high_cardinality_cols)
        return self


avaliable_methods = {
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
        "LDA": {"class": LinearDiscriminantAnalysis, "args": {}},
    },
}
