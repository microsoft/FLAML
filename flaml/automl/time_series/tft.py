import inspect
import time

import numpy as np

try:
    import pandas as pd
    from pandas import DataFrame, Series, to_datetime
except ImportError:

    class PD:
        pass

    pd = PD()
    pd.DataFrame = None
    pd.Series = None
    DataFrame = Series = None

from flaml import tune
from flaml.automl.time_series.ts_data import TimeSeriesDataset, _concat_panel_frames
from flaml.automl.time_series.ts_model import TimeSeriesEstimator


class TemporalFusionTransformerEstimator(TimeSeriesEstimator):
    """The class for tuning Temporal Fusion Transformer"""

    _supports_in_sample_prediction = False

    @classmethod
    def search_space(cls, data, task, pred_horizon, **params):
        space = {
            "gradient_clip_val": {
                "domain": tune.loguniform(lower=0.01, upper=100.0),
                "init_value": 0.01,
            },
            "hidden_size": {
                "domain": tune.lograndint(lower=8, upper=512),
                "init_value": 16,
            },
            "hidden_continuous_size": {
                "domain": tune.randint(lower=1, upper=65),
                "init_value": 8,
            },
            "attention_head_size": {
                "domain": tune.randint(lower=1, upper=5),
                "init_value": 4,
            },
            "dropout": {
                "domain": tune.uniform(lower=0.1, upper=0.3),
                "init_value": 0.1,
            },
            "learning_rate": {
                "domain": tune.loguniform(lower=0.00001, upper=1.0),
                "init_value": 0.001,
            },
        }
        return space

    def transform_ds(self, X_train: TimeSeriesDataset, y_train, **kwargs):
        self.data = X_train.train_data.copy()
        self.time_idx = X_train.time_idx

        max_prediction_length = kwargs["period"]
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = kwargs["max_encoder_length"]
        training_cutoff = self.data[self.time_idx].max() - max_prediction_length

        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        self.group_ids = kwargs["group_ids"].copy()
        training = TimeSeriesDataSet(
            self.data[lambda x: x[self.time_idx] <= training_cutoff],
            time_idx=self.time_idx,
            target=X_train.target_names[0],
            group_ids=self.group_ids,
            min_encoder_length=kwargs.get(
                "min_encoder_length", self.max_encoder_length // 2
            ),  # keep encoder length long (as it is in the validation set)
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=kwargs.get("static_categoricals", []),
            static_reals=kwargs.get("static_reals", []),
            time_varying_known_categoricals=kwargs.get("time_varying_known_categoricals", []),
            time_varying_known_reals=kwargs.get("time_varying_known_reals", []),
            time_varying_unknown_categoricals=kwargs.get("time_varying_unknown_categoricals", []),
            time_varying_unknown_reals=kwargs.get("time_varying_unknown_reals", []),
            variable_groups=kwargs.get(
                "variable_groups", {}
            ),  # group of categorical variables can be treated as one variable
            lags=kwargs.get("lags", {}),
            target_normalizer=GroupNormalizer(
                groups=kwargs["group_ids"], transformation="softplus"
            ),  # use softplus and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        validation = TimeSeriesDataSet.from_dataset(training, self.data, predict=True, stop_randomization=True)

        # create dataloaders for model
        batch_size = kwargs.get("batch_size", 64)
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

        return training, train_dataloader, val_dataloader

    def fit(self, X_train, y_train, budget=None, **kwargs):
        import warnings

        try:
            import lightning.pytorch as pl
            from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
            from lightning.pytorch.loggers import TensorBoardLogger
        except ImportError:
            import pytorch_lightning as pl
            from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
            from pytorch_lightning.loggers import TensorBoardLogger
        import torch
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss

        # a bit of monkey patching to fix the MacOS test
        # all the log_prediction method appears to do is plot stuff, which ?breaks github tests
        def log_prediction(*args, **kwargs):
            pass

        TemporalFusionTransformer.log_prediction = log_prediction

        warnings.filterwarnings("ignore")
        current_time = time.time()
        super().fit(X_train, **kwargs)
        training, train_dataloader, val_dataloader = self.transform_ds(X_train, y_train, **kwargs)
        params = self.params.copy()
        gradient_clip_val = params.pop("gradient_clip_val", None)
        params.pop("n_jobs", None)
        max_epochs = kwargs.get("max_epochs", 20)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger(kwargs.get("log_dir", "lightning_logs"))  # logging results to a tensorboard
        default_trainer_kwargs = dict(
            max_epochs=max_epochs,
            gradient_clip_val=gradient_clip_val,
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )

        # PyTorch Lightning >=2.0 replaced `gpus` with `accelerator`/`devices`.
        # Also, passing `gpus=None` is not accepted on newer versions.
        trainer_sig_params = inspect.signature(pl.Trainer.__init__).parameters
        if torch.cuda.is_available() and "gpus" in trainer_sig_params:
            gpus = self._kwargs.get("gpu_per_trial", None)
            if gpus is not None:
                default_trainer_kwargs["gpus"] = gpus
        elif torch.cuda.is_available() and "devices" in trainer_sig_params:
            devices = self._kwargs.get("gpu_per_trial", None)
            if devices == -1:
                devices = "auto"
            if devices is not None:
                default_trainer_kwargs["accelerator"] = "gpu"
                default_trainer_kwargs["devices"] = devices
        trainer = pl.Trainer(
            **default_trainer_kwargs,
        )
        tft = TemporalFusionTransformer.from_dataset(
            training,
            **params,
            lstm_layers=2,  # 2 is mostly optimal according to documentation
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            reduce_on_plateau_patience=4,
        )
        # fit network
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        best_model_path = trainer.checkpoint_callback.best_model_path
        # PyTorch 2.6 changed `torch.load` default `weights_only` from False -> True.
        # Some Lightning checkpoints (including those produced here) can require full unpickling.
        # This path is generated locally during training, so it's trusted.
        load_sig_params = inspect.signature(TemporalFusionTransformer.load_from_checkpoint).parameters
        if "weights_only" in load_sig_params:
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path, weights_only=False)
        else:
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        train_time = time.time() - current_time
        self._model = best_tft
        return train_time

    def _prediction_indexes(self, history, future, time_idx, supplied_index):
        maximum = getattr(self, "max_prediction_length", None)
        if maximum is not None and len(future) > maximum:
            raise ValueError("TFT prediction exceeds the fitted forecast horizon.")
        end = int(history[time_idx].iloc[-1])
        expected = list(range(end + 1, end + 1 + len(future)))
        if expected[-1] > np.iinfo(np.int64).max:
            raise ValueError("TFT prediction indexes exceed the signed integer range.")
        if supplied_index and future[time_idx].tolist() != expected:
            raise ValueError("TFT requires consecutive prediction indexes immediately after training.")
        timestamps = future[self.time_col]
        if timestamps.isna().any() or not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise ValueError("TFT prediction timestamps must be nonmissing and strictly increasing.")
        try:
            starts_after_training = timestamps.iloc[0] > history[self.time_col].iloc[-1]
        except TypeError as exc:
            raise ValueError("TFT prediction timestamps must use a timezone compatible with training.") from exc
        if not starts_after_training:
            raise ValueError("TFT prediction timestamps must be after the group's training history.")
        history_dates = history[self.time_col].drop_duplicates()
        frequency = pd.infer_freq(history_dates) if len(history_dates) >= 3 else None
        if frequency is not None:
            dates = pd.date_range(history[self.time_col].iloc[-1], periods=len(future) + 1, freq=frequency)[1:]
            if list(future[self.time_col]) != list(dates):
                raise ValueError("TFT requires consecutive prediction timestamps immediately after training.")
        elif not supplied_index:
            raise ValueError("Supply explicit TFT time indexes when the training calendar is irregular.")
        return expected

    def _prediction_frame(self, decoder, time_idx):
        decoder[self.time_col] = pd.to_datetime(decoder[self.time_col])
        supplied_index = time_idx in decoder
        if supplied_index and (decoder[time_idx].dtype.kind != "i" or decoder[time_idx].isna().any()):
            raise ValueError("TFT prediction time indexes must be nonmissing signed integers.")
        if not supplied_index:
            decoder[time_idx] = pd.Series(pd.NA, index=decoder.index, dtype="Int64")
        by = self.group_ids[0] if len(self.group_ids) == 1 else self.group_ids
        training = self.data.groupby(by, observed=True, sort=False)
        parameters = getattr(self._model, "dataset_parameters", None) or {}
        max_lag = max((lag for lags in (parameters.get("lags") or {}).values() for lag in lags), default=0)
        encoders = []
        for key, positions in decoder.groupby(by, observed=True, sort=False).indices.items():
            try:
                history = training.get_group(key).sort_values(time_idx)
            except KeyError as exc:
                raise ValueError(f"TFT prediction contains an unknown group: {key!r}") from exc
            history[self.time_col] = pd.to_datetime(history[self.time_col])
            future = decoder.iloc[positions].sort_values(time_idx if supplied_index else self.time_col)
            decoder.loc[future.index, time_idx] = self._prediction_indexes(history, future, time_idx, supplied_index)
            decoder.loc[future.index, self.target_names[0]] = history[self.target_names[0]].iloc[-1]
            encoders.append(history.tail(self.max_encoder_length + max_lag))
        decoder[time_idx] = decoder[time_idx].astype("int64")
        return _concat_panel_frames(
            [*encoders, decoder.sort_values(self.group_ids + [time_idx])], self.group_ids, ignore_index=True
        )

    def _prediction_dataset(self, data, horizon):
        from pytorch_forecasting import TimeSeriesDataSet

        # Changing the fitted maximum can reorder known/unknown lagged features.
        return TimeSeriesDataSet.from_parameters(
            self._model.dataset_parameters,
            data,
            predict=True,
            stop_randomization=True,
            min_prediction_length=horizon,
            min_prediction_idx=int(data[getattr(self, "time_idx", "time_idx")].max()) - horizon + 1,
        )

    def _restore_prediction_index(self, X, prediction):
        if isinstance(X, TimeSeriesDataset):
            request = X.X_val if X.test_data is not None else X.X_train.iloc[:0]
        elif isinstance(X, pd.DataFrame):
            request = X
        else:
            return prediction
        ids = self.group_ids + [self.time_col]
        original = request[ids]
        original_index = pd.MultiIndex.from_frame(original)
        normalized = original.copy()
        normalized[self.time_col] = pd.to_datetime(normalized[self.time_col])
        expected = pd.MultiIndex.from_frame(normalized)
        index = getattr(prediction, "index", None)
        if (
            not isinstance(index, pd.MultiIndex)
            or index.names != expected.names
            or len(index) != len(expected)
            or (
                any(actual != requested for actual, requested in zip(index, expected))
                and any(actual != requested for actual, requested in zip(index, original_index))
            )
        ):
            raise ValueError("TFT preprocessing changed prediction coordinates or order.")
        return prediction.set_axis(original_index)

    def predict(self, X):
        time_idx = getattr(self, "time_idx", "time_idx")
        ids = self.group_ids + [self.time_col]
        if isinstance(X, TimeSeriesDataset):
            request = X.X_val if X.test_data is not None else X.X_train.iloc[:0]
        else:
            request = X
        decoder = request.drop(columns=self.target_names, errors="ignore").copy().reset_index(drop=True)
        if not set(ids).issubset(decoder) or decoder[ids].isna().any().any():
            raise ValueError("TFT prediction requires nonmissing group keys and timestamps.")
        request_index = pd.MultiIndex.from_frame(decoder[ids])
        if decoder.empty:
            return pd.Series(dtype=float, index=request_index)
        data = self._prediction_frame(decoder, time_idx)
        by = self.group_ids[0] if len(self.group_ids) == 1 else self.group_ids
        data_groups = data.groupby(by, observed=True, sort=False).indices
        cohorts = {}
        for group, positions in decoder.groupby(by, observed=True, sort=False).indices.items():
            key = (len(positions), int(decoder.iloc[positions][time_idx].min()))
            cohorts.setdefault(key, []).append((positions, data_groups[group]))
        forecasts = {}
        # A fitted maximum decoder length must not consume recent encoder rows in shorter requests.
        for (horizon, origin), groups in cohorts.items():
            requested = decoder.iloc[np.concatenate([positions for positions, _ in groups])]
            subset = data.iloc[np.concatenate([positions for _, positions in groups])]
            prediction_data = self._prediction_dataset(subset, int(horizon))
            result = self._model.predict(prediction_data, return_index=True)
            values, index = (result.output, result.index) if hasattr(result, "output") else result[:2]
            values = values.detach().cpu().numpy() if hasattr(values, "detach") else values
            if len(values) != len(index):
                raise ValueError("TFT predictions do not match their returned group indexes.")
            origins = {
                row[:-1]: int(row[-1])
                for row in requested.groupby(self.group_ids, observed=True, sort=False)[time_idx]
                .min()
                .reset_index()
                .itertuples(index=False, name=None)
            }
            seen = set()
            for coordinates, predictions in zip(
                index[self.group_ids + [time_idx]].itertuples(index=False, name=None), values
            ):
                group, start = coordinates[:-1], int(coordinates[-1])
                if group in seen or origins.get(group) != start:
                    raise ValueError("TFT decoder must start immediately after each group's training history.")
                seen.add(group)
                predictions = predictions.detach().cpu().numpy() if hasattr(predictions, "detach") else predictions
                predictions = np.asarray(predictions).reshape(-1)
                if len(predictions) != horizon:
                    raise ValueError("TFT decoder length must match the requested forecast horizon.")
                forecasts.update({group + (start + offset,): value for offset, value in enumerate(predictions)})
            if seen != set(origins):
                raise ValueError("TFT did not produce every requested panel coordinate.")
        keys = decoder[self.group_ids + [time_idx]].itertuples(index=False, name=None)
        try:
            predictions = [forecasts[key] for key in keys]
        except KeyError as exc:
            raise ValueError("TFT did not produce every requested panel coordinate.") from exc
        return pd.Series(predictions, index=request_index)
