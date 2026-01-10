import inspect
import time

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
from flaml.automl.data import add_time_idx_col
from flaml.automl.time_series.ts_data import TimeSeriesDataset
from flaml.automl.time_series.ts_model import TimeSeriesEstimator


class TemporalFusionTransformerEstimator(TimeSeriesEstimator):
    """The class for tuning Temporal Fusion Transformer"""

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
        self.data = X_train.train_data

        max_prediction_length = kwargs["period"]
        self.max_encoder_length = kwargs["max_encoder_length"]
        training_cutoff = self.data["time_idx"].max() - max_prediction_length

        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        self.group_ids = kwargs["group_ids"].copy()
        training = TimeSeriesDataSet(
            self.data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
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

    def predict(self, X):
        ids = self.group_ids.copy()
        ids.append(self.time_col)
        encoder_data = self.data[lambda x: x.time_idx > x.time_idx.max() - self.max_encoder_length]
        # following pytorchforecasting example, make all target values equal to the last data
        last_data_cols = self.group_ids.copy()
        last_data_cols.append(self.target_names[0])
        last_data = self.data[lambda x: x.time_idx == x.time_idx.max()][last_data_cols]
        # Use X_train if test_data is empty (e.g., when computing training metrics)
        if isinstance(X, TimeSeriesDataset):
            decoder_data = X.X_val if len(X.test_data) > 0 else X.X_train
        else:
            decoder_data = X
        if "time_idx" not in decoder_data:
            decoder_data = add_time_idx_col(decoder_data)
        decoder_data["time_idx"] += encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()
        decoder_data = decoder_data.merge(last_data, how="inner", on=self.group_ids)
        decoder_data = decoder_data.sort_values(ids)
        new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
        new_prediction_data["time_idx"] = new_prediction_data["time_idx"].astype("int")
        new_raw_predictions = self._model.predict(new_prediction_data)
        index = [decoder_data[idx].to_numpy() for idx in ids]
        predictions = pd.Series(new_raw_predictions.numpy().ravel(), index=index)
        return predictions
