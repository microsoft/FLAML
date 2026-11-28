from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted


def _forecast_time_column(task, X):
    current = getattr(task, "time_col", None)
    if not isinstance(X, pd.DataFrame):
        return current or "ds"
    if current in X.columns:
        return current
    datetimes = [name for name in X if pd.api.types.is_datetime64_any_dtype(X[name])]
    return datetimes[0] if len(datetimes) == 1 else X.columns[0]


class _TaskPreprocessor(TransformerMixin, BaseEstimator):
    """Adapt FLAML task preprocessing to sklearn's feature-only interface."""

    def __init__(self, task, transformer=None, *, preserve_features=False, feature_only=False):
        self.task = task
        self.transformer = transformer
        self.preserve_features = preserve_features
        self.feature_only = feature_only
        self._is_fitted = False
        self._target_classes = None

    def __sklearn_is_fitted__(self):
        return self._is_fitted

    def fit(self, X, y=None, *, group_ids=None):
        self.fit_transform(X, y, group_ids=group_ids)
        return self

    def fit_transform(self, X, y=None, *, group_ids=None):
        from flaml.automl.data import DataTransformer, _make_column_name_mapping
        from flaml.automl.task.factory import task_factory
        from flaml.automl.time_series.ts_data import (
            DataTransformerTS,
            TimeSeriesDataset,
            _empty_forecast_frame,
            normalize_ts_data,
        )

        self._is_fitted = False
        self._target_classes = None
        self.task_ = task_factory(self.task, X, y) if isinstance(self.task, str) else deepcopy(self.task)
        task = self.task_
        if group_ids is not None:
            if not isinstance(group_ids, list) or not group_ids:
                raise ValueError("Panel forecasting requires a nonempty list of group_ids.")
            task.group_ids = group_ids.copy()
        if task.is_ts_forecast():
            time_idx = getattr(self.transformer, "time_idx", "time_idx")
            test_data = None
            if isinstance(X, TimeSeriesDataset):
                task.time_col = X.time_col
                task.target_names = list(X.target_names)
                time_idx = X.time_idx
                test_data = X.test_data
            else:
                task.time_col = _forecast_time_column(task, X)
                if isinstance(y, pd.DataFrame) or getattr(y, "name", None) is not None:
                    task.target_names = (
                        list(y.columns) if isinstance(y, pd.DataFrame) else [getattr(y, "name", None) or "y"]
                    )
                elif getattr(task, "target_names", None) is None:
                    task.target_names = ["y"]
            if getattr(self, "feature_only", False):
                features = (
                    X.X_train.copy()
                    if isinstance(X, TimeSeriesDataset)
                    else normalize_ts_data(
                        X.copy() if isinstance(X, (pd.DataFrame, np.ndarray)) else X,
                        task.target_names,
                        task.time_col,
                    )
                )
                labels = X.y_train if isinstance(X, TimeSeriesDataset) and y is None else y
                if labels is not None:
                    labels = pd.DataFrame(
                        np.asarray(labels).reshape(len(features), -1),
                        index=features.index,
                        columns=task.target_names,
                    )
            elif y is not None and not isinstance(X, TimeSeriesDataset):
                features = normalize_ts_data(
                    X.copy() if isinstance(X, (pd.DataFrame, np.ndarray)) else X,
                    task.target_names,
                    task.time_col,
                )
                labels = pd.DataFrame(
                    np.asarray(y).reshape(len(features), -1), index=features.index, columns=task.target_names
                )
            else:
                frame = normalize_ts_data(
                    X.copy() if isinstance(X, (pd.DataFrame, np.ndarray)) else X,
                    task.target_names,
                    task.time_col,
                    None if y is None else np.asarray(y),
                )
                if isinstance(frame, TimeSeriesDataset):
                    frame = frame.train_data.copy()
                if not all(name in frame for name in task.target_names):
                    raise ValueError("Forecast model refits require target labels.")
                features = frame.drop(columns=task.target_names)
                labels = frame[task.target_names]
            input_targets = list(task.target_names)
            if labels is not None:
                targets, used = [], set(features.columns)
                if self.preserve_features:
                    mapping = _make_column_name_mapping(features.columns)
                    mapping.pop(task.time_col, None)
                    mapping.pop(time_idx, None)
                    used.update(mapping.values())
                for name in input_targets:
                    candidate = name
                    while candidate in used:
                        candidate = f"{candidate}_target"
                    targets.append(candidate)
                    used.add(candidate)
                task.target_names = targets
                labels = labels.copy()
                labels.columns = targets
                if self.preserve_features and task.is_classification():
                    self._target_classes = set(np.unique(labels.iloc[:, 0]))
            features[task.time_col] = pd.to_datetime(features[task.time_col])
            self.transformer_ = DataTransformerTS(task.time_col, task.target_names, time_idx=time_idx)
            transformed = self.transformer_.fit_transform(
                features,
                labels,
                preserve_features=self.preserve_features,
                protected_columns=features.columns if task.is_ts_forecastpanel() else (),
                group_columns=getattr(task, "group_ids", ()) if task.is_ts_forecastpanel() else (),
            )
            if labels is None:
                features = transformed
            else:
                features, labels = transformed
            if getattr(self, "feature_only", False):
                self._is_fitted = True
                return features
            train_data = pd.concat([features, labels], axis=1)
            if test_data is not None and len(test_data):
                test_data = self._transform_forecast_frame(test_data, input_targets)
            else:
                test_data = _empty_forecast_frame(train_data, test_data, input_targets, task.target_names)
            data = TimeSeriesDataset(
                train_data,
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
            if getattr(self, "feature_only", False):
                features = X.X_train.copy()
                features[self.task_.time_col] = pd.to_datetime(features[self.task_.time_col])
                return self.transformer_.transform(features)
            return self._transform_validation_data(X)[0]
        if self.task_.is_ts_forecast() and isinstance(X, pd.DataFrame):
            X = X.copy()
            if self.task_.time_col in X:
                X[self.task_.time_col] = pd.to_datetime(X[self.task_.time_col])
        return self.task_.preprocess(X, self.transformer_)

    def _transform_validation_data(self, X):
        from flaml.automl.time_series.ts_data import _transform_ts_dataset

        check_is_fitted(self)
        return _transform_ts_dataset(
            X,
            lambda frame: self._transform_forecast_frame(frame, X.target_names),
            target_names=self.task_.target_names,
        )

    def _transform_forecast_frame(self, frame, input_targets=None):
        targets = self.task_.target_names
        input_targets = targets if input_targets is None else input_targets
        features = frame.drop(columns=input_targets, errors="ignore").copy()
        features[self.task_.time_col] = pd.to_datetime(features[self.task_.time_col])
        if not all(name in frame for name in input_targets):
            return self.transformer_.transform(features)
        labels = frame[input_targets].copy()
        if len(input_targets) != len(targets):
            raise ValueError("Forecast target columns must match the fitted target count.")
        classes = getattr(self, "_target_classes", None)
        if classes is not None and not set(np.unique(labels.iloc[:, 0])).issubset(classes):
            raise ValueError("AutoFE forecast validation contains classes unseen in the training data.")
        labels.columns = targets
        features, labels = self.transformer_.transform(features, labels)
        return pd.concat([features, labels], axis=1)


class AutoMLPipeline(Pipeline):
    """A feature pipeline that preserves AutoML's original target labels."""

    def __init__(self, steps, label_transformer=None, *, resampler=None, memory=None, verbose=False):
        self.label_transformer = label_transformer
        self.resampler = resampler
        super().__init__(steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **params):
        from flaml.automl.ml import _validate_resampler_task
        from flaml.automl.time_series.ts_data import TimeSeriesDataset

        if self.resampler is not None:
            _validate_resampler_task(self.resampler, getattr(self.steps[-1][1], "_task", None), X)
            for _, step in self.steps[:-1]:
                if isinstance(step, _TaskPreprocessor):
                    _validate_resampler_task(self.resampler, step.task, X)
        estimator_groups = params.get(f"{self.steps[-1][0]}__group_ids")
        if estimator_groups is not None:
            for name, step in self.steps[:-1]:
                if isinstance(step, _TaskPreprocessor):
                    key = f"{name}__group_ids"
                    if key in params and params[key] != estimator_groups:
                        raise ValueError("Preprocessing and estimator group_ids must agree.")
                    params = {**params, key: estimator_groups}
        label_transformer = clone(self.label_transformer) if self.label_transformer is not None else None
        if label_transformer is not None:
            if isinstance(X, TimeSeriesDataset):
                X = deepcopy(X)
                target = X.target_names[0]
                X.train_data[target] = label_transformer.fit_transform(X.y_train)
                if target in X.test_data and len(X.test_data):
                    X.test_data[target] = label_transformer.transform(X.test_data[target])
            else:
                encoded = label_transformer.fit_transform(np.asarray(y).reshape(-1))
                if isinstance(y, pd.Series):
                    y = pd.Series(encoded, index=y.index, name=y.name)
                elif isinstance(y, pd.DataFrame):
                    y = pd.DataFrame(encoded, index=y.index, columns=y.columns)
                else:
                    y = encoded
        if self.resampler is None:
            result = super().fit(X, y, **params)
            self._set_label_transformer(label_transformer)
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
        expected_classes = (
            np.unique(y)
            if label_transformer is not None
            or is_classifier(estimator)
            or any(
                isinstance(step, _TaskPreprocessor) and step.task_.is_classification() for _, step in self.steps[:-1]
            )
            else None
        )
        X, y = _resample_training_data(X, y, resampler=self.resampler, expected_classes=expected_classes)
        estimator.fit(X, y, **step_params[estimator_name])
        self._set_label_transformer(label_transformer)
        return self

    def _set_label_transformer(self, transformer):
        for _, step in self.steps[:-1]:
            if isinstance(step, _TaskPreprocessor) and step.task_.is_ts_forecast():
                if transformer is None:
                    transformer = step.transformer_.label_transformer
                elif transformer is not None:
                    step.transformer_.label_transformer = transformer
                if step.preserve_features and step.task_.is_classification() and transformer is not None:
                    step._target_classes = set(transformer.classes_)
        self.label_transformer_ = transformer

    @available_if(lambda self: hasattr(self._final_estimator, "fit_predict"))
    def fit_predict(self, X, y=None, **params):
        if self.label_transformer is not None or self.resampler is not None:
            raise NotImplementedError(
                "fit_predict cannot apply target transforms or fit-only resampling; use fit() followed by predict()."
            )
        return super().fit_predict(X, y, **params)

    def predict(self, X, **params):
        prediction = super().predict(X, **params)
        restore_index = getattr(type(self._final_estimator), "_restore_prediction_index", None)
        if callable(restore_index):
            prediction = restore_index(self._final_estimator, X, prediction)
        transformer = getattr(self, "label_transformer_", self.label_transformer)
        if transformer is not None:
            codes = np.asarray(prediction)
            if codes.ndim == 2 and codes.shape[1] == 1:
                codes = codes[:, 0]
            expected_rows = None
            if isinstance(X, (int, np.integer)):
                expected_rows = int(X)
            elif hasattr(X, "shape") and len(X.shape) > 1:
                expected_rows = X.shape[0]
            elif isinstance(X, (list, tuple)) and all(isinstance(row, (list, tuple)) for row in X):
                expected_rows = len(X)
            # A 1D input may be one feature row or a univariate batch; its
            # estimator/transformer owns that convention, not the label decoder.
            if codes.ndim != 1 or (expected_rows is not None and len(codes) != expected_rows):
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
        from flaml.automl.time_series.ts_data import TimeSeriesDataset

        transformer = getattr(self, "label_transformer_", self.label_transformer)
        if transformer is not None and not (isinstance(X, TimeSeriesDataset) and y is None):
            y = transformer.transform(y)
        return super().score(X, y, sample_weight=sample_weight, **params)

    @property
    def classes_(self):
        transformer = getattr(self, "label_transformer_", self.label_transformer)
        return transformer.classes_ if transformer is not None else super().classes_


class _ForecastFeaturePipeline(Pipeline):
    def _prepare_fit(self, X, y):
        from flaml.automl.task.factory import task_factory
        from flaml.automl.time_series.ts_data import TimeSeriesDataset

        preprocessor = self.named_steps["feature_transformer"]
        task = task_factory(preprocessor.task) if isinstance(preprocessor.task, str) else deepcopy(preprocessor.task)
        if isinstance(X, TimeSeriesDataset):
            task.time_col = X.time_col
            task.target_names = X.target_names
            y = X.y_train if y is None else y
        else:
            task.time_col = _forecast_time_column(task, X)
        preprocessor.task = task
        if "autofe" in self.named_steps:
            self.named_steps["autofe"].task = task
        return y

    def fit(self, X, y=None, **params):
        return super().fit(X, self._prepare_fit(X, y), **params)

    def fit_transform(self, X, y=None, **params):
        return super().fit_transform(X, self._prepare_fit(X, y), **params)


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
    pipeline_class = _ForecastFeaturePipeline if automl._state.task.is_ts_forecast() else Pipeline
    pipeline = deepcopy(pipeline_class(_preprocessing_steps(automl, estimator)))
    pipeline.named_steps["feature_transformer"].feature_only = True
    return pipeline


def make_automl_pipeline(automl, estimator):
    """Snapshot fitted preprocessing and the supplied model without sharing mutable state."""
    steps = _preprocessing_steps(automl, estimator)
    steps.append(("estimator", estimator))
    label_transformer = getattr(automl, "label_transformer", getattr(automl, "_label_transformer", None))
    preprocessor = getattr(estimator, "_flaml_preprocessor", None)
    if preprocessor is not None and preprocessor.task_.is_ts_forecast():
        label_transformer = getattr(preprocessor.transformer_, "label_transformer", None)
    if label_transformer is False:
        label_transformer = None
    return deepcopy(
        AutoMLPipeline(
            steps, label_transformer=label_transformer, resampler=getattr(automl._state.task, "_resampler", None)
        )
    )
