from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class _TaskPreprocessor(TransformerMixin, BaseEstimator):
    """Adapt FLAML task preprocessing to sklearn's feature-only interface."""

    def __init__(self, task, transformer=None, *, preserve_features=False):
        self.task = task
        self.transformer = transformer
        self.preserve_features = preserve_features
        self._is_fitted = False

    def __sklearn_is_fitted__(self):
        return self._is_fitted

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        from flaml.automl.data import DataTransformer
        from flaml.automl.task.factory import task_factory
        from flaml.automl.time_series.ts_data import DataTransformerTS, TimeSeriesDataset, normalize_ts_data

        self._is_fitted = False
        self.task_ = task_factory(self.task, X, y) if isinstance(self.task, str) else deepcopy(self.task)
        task = self.task_
        if task.is_ts_forecast():
            time_idx = getattr(self.transformer, "time_idx", "time_idx")
            test_data = None
            if isinstance(X, TimeSeriesDataset):
                task.time_col = X.time_col
                task.target_names = list(X.target_names)
                time_idx = X.time_idx
                test_data = X.test_data
            else:
                if getattr(task, "time_col", None) is None:
                    task.time_col = X.columns[0] if isinstance(X, pd.DataFrame) else "ds"
                if getattr(task, "target_names", None) is None:
                    task.target_names = (
                        list(y.columns) if isinstance(y, pd.DataFrame) else [getattr(y, "name", None) or "y"]
                    )
            frame = normalize_ts_data(
                X.copy() if isinstance(X, (pd.DataFrame, np.ndarray)) else X,
                task.target_names,
                task.time_col,
                None if y is None else np.asarray(y),
            )
            if isinstance(frame, TimeSeriesDataset):
                frame = frame.train_data.copy()
            features = frame.drop(columns=task.target_names)
            labels = frame[task.target_names]
            self.transformer_ = DataTransformerTS(task.time_col, task.target_names, time_idx=time_idx)
            features, labels = self.transformer_.fit_transform(
                features, labels, preserve_features=self.preserve_features
            )
            if test_data is not None and len(test_data):
                test_data = self._transform_forecast_frame(test_data)
            data = TimeSeriesDataset(
                pd.concat([features, labels], axis=1),
                task.time_col,
                task.target_names,
                time_idx=time_idx,
                test_data=test_data,
            )
            self._is_fitted = True
            return data
        if self.transformer is None:
            self.transformer_ = None
            features = task.preprocess(X)
            self._is_fitted = True
            return features
        self.transformer_ = DataTransformer()
        features, _ = self.transformer_.fit_transform(X, y, task, preserve_features=self.preserve_features)
        self._is_fitted = True
        return features

    def transform(self, X):
        check_is_fitted(self)
        from flaml.automl.time_series.ts_data import TimeSeriesDataset

        if isinstance(X, TimeSeriesDataset):
            return TimeSeriesDataset(
                self._transform_forecast_frame(X.train_data),
                X.time_col,
                X.target_names,
                time_idx=X.time_idx,
                test_data=self._transform_forecast_frame(X.test_data) if len(X.test_data) else X.test_data.copy(),
            )
        return self.task_.preprocess(X, self.transformer_)

    def _transform_forecast_frame(self, frame):
        targets = self.task_.target_names
        features = frame.drop(columns=targets, errors="ignore").copy()
        if not all(name in frame for name in targets):
            return self.transformer_.transform(features)
        features, labels = self.transformer_.transform(features, frame[targets].copy())
        return pd.concat([features, labels], axis=1)


class AutoMLPipeline(Pipeline):
    """A feature pipeline that preserves AutoML's original target labels."""

    def __init__(self, steps, label_transformer=None, *, resampler=None, memory=None, verbose=False):
        self.label_transformer = label_transformer
        self.resampler = resampler
        super().__init__(steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **params):
        label_transformer = clone(self.label_transformer) if self.label_transformer is not None else None
        if label_transformer is not None:
            y = label_transformer.fit_transform(y)
        if self.resampler is None:
            result = super().fit(X, y, **params)
            self.label_transformer_ = label_transformer
            return result

        from flaml.automl.ml import _resample_training_data, _validate_resampler

        self._validate_steps()
        step_params = {name: {} for name, _ in self.steps}
        for name, value in params.items():
            if "__" not in name:
                raise ValueError("Pipeline fit parameters must use step__parameter names.")
            step, parameter = name.split("__", 1)
            if step not in step_params:
                raise ValueError(f"Unknown pipeline step: {step}")
            step_params[step][parameter] = value
        _, step_params = _validate_resampler(self.resampler, None, False, {}, step_params)
        estimator_name, estimator = self.steps[-1]
        if len(self.steps) > 1:
            preprocessing = Pipeline(self.steps[:-1], memory=self.memory, verbose=self.verbose)
            preprocessing_params = {
                f"{step}__{parameter}": value
                for step, values in step_params.items()
                if step != estimator_name
                for parameter, value in values.items()
            }
            X = preprocessing.fit_transform(X, y, **preprocessing_params)
            self.steps = preprocessing.steps + [(estimator_name, estimator)]
        X, y = _resample_training_data(X, y, resampler=self.resampler)
        estimator.fit(X, y, **step_params[estimator_name])
        self.label_transformer_ = label_transformer
        return self

    def predict(self, X, **params):
        prediction = super().predict(X, **params)
        transformer = getattr(self, "label_transformer_", self.label_transformer)
        if transformer is not None:
            codes = np.asarray(prediction)
            if codes.ndim == 2 and codes.shape[1] == 1:
                codes = codes[:, 0]
            if isinstance(X, (int, np.integer)):
                expected_rows = int(X)
            elif hasattr(X, "shape"):
                expected_rows = X.shape[0]
            elif isinstance(X, list) and X and not isinstance(X[0], list):
                expected_rows = 1
            else:
                expected_rows = len(X)
            if codes.ndim != 1 or len(codes) != expected_rows:
                raise ValueError("Classifier predictions must contain one class index per input row.")
            try:
                with np.errstate(invalid="ignore", over="ignore"):
                    indices = codes.astype(np.int64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Classifier predictions must be finite integral class indices.") from exc
            if not np.array_equal(codes, indices):
                raise ValueError("Classifier predictions must be finite integral class indices.")
            prediction = transformer.inverse_transform(indices)
        return prediction

    def score(self, X, y=None, sample_weight=None, **params):
        transformer = getattr(self, "label_transformer_", self.label_transformer)
        if transformer is not None:
            y = transformer.transform(y)
        return super().score(X, y, sample_weight=sample_weight, **params)

    @property
    def classes_(self):
        transformer = getattr(self, "label_transformer_", self.label_transformer)
        return transformer.classes_ if transformer is not None else super().classes_


def _preprocessing_steps(automl, estimator):
    preprocessor = getattr(estimator, "_flaml_preprocessor", None)
    if preprocessor is None:
        transformer = getattr(automl, "_transformer", None)
        if transformer is False:
            transformer = None
        preprocessor = _TaskPreprocessor(automl._state.task, transformer)
        preprocessor.task_ = automl._state.task
        preprocessor.transformer_ = transformer
        preprocessor._is_fitted = True
    steps = [("feature_transformer", preprocessor)]
    autofe = getattr(estimator, "autofe", None)
    if autofe is not None:
        steps.append(("autofe", autofe))
    return steps


def make_automl_feature_transformer(automl, estimator):
    """Snapshot only preprocessing without copying the fitted learner."""
    return deepcopy(Pipeline(_preprocessing_steps(automl, estimator)))


def make_automl_pipeline(automl, estimator):
    """Snapshot fitted preprocessing and the supplied model without sharing mutable state."""
    steps = _preprocessing_steps(automl, estimator)
    steps.append(("estimator", estimator))
    label_transformer = getattr(automl, "_label_transformer", None)
    if label_transformer is False:
        label_transformer = None
    return deepcopy(
        AutoMLPipeline(
            steps, label_transformer=label_transformer, resampler=getattr(automl._state.task, "_resampler", None)
        )
    )
