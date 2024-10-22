# This file is adapted from
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# https://github.com/locuslab/TCN/blob/master/TCN/adding_problem/add_test.py

import datetime
import logging
import time

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset

from flaml import tune
from flaml.automl.data import add_time_idx_col
from flaml.automl.logger import logger, logger_formatter
from flaml.automl.time_series.ts_data import TimeSeriesDataset
from flaml.automl.time_series.ts_model import TimeSeriesEstimator


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    def __init__(
        self,
        input_feature_num,
        num_outputs,
        num_channels,
        kernel_size=2,
        dropout=0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_feature_num if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_outputs)

    def forward(self, x):
        y1 = self.network(x)
        return self.linear(y1[:, :, -1])


class TCNForecasterLightningModule(pl.LightningModule):
    def __init__(self, model: TCNForecaster, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class DataframeDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, target_column, features_columns, sequence_length, train=True):
        self.data = torch.tensor(dataframe[features_columns].to_numpy(), dtype=torch.float)
        self.sequence_length = sequence_length
        if train:
            self.labels = torch.tensor(dataframe[target_column].to_numpy(), dtype=torch.float)
        self.is_train = train

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        data = self.data[idx : idx + self.sequence_length]
        data = data.permute(1, 0)
        if self.is_train:
            label = self.labels[idx : idx + self.sequence_length]
            return data, label
        else:
            return data


class TCNEstimator(TimeSeriesEstimator):
    """The class for tuning TCN Forecaster"""

    @classmethod
    def search_space(cls, data, task, pred_horizon, **params):
        space = {
            "num_levels": {
                "domain": tune.randint(lower=4, upper=20),  # hidden = 2^num_hidden
                "init_value": 4,
            },
            "num_hidden": {
                "domain": tune.randint(lower=4, upper=8),  # hidden = 2^num_hidden
                "init_value": 5,
            },
            "kernel_size": {
                "domain": tune.choice([2, 3, 5, 7]),  # common choices for kernel size
                "init_value": 3,
            },
            "dropout": {
                "domain": tune.uniform(lower=0.0, upper=0.5),  # standard range for dropout
                "init_value": 0.1,
            },
            "learning_rate": {
                "domain": tune.loguniform(lower=1e-4, upper=1e-1),  # typical range for learning rate
                "init_value": 1e-3,
            },
        }
        return space

    def __init__(self, task="ts_forecast", n_jobs=1, **params):
        super().__init__(task, **params)
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    def fit(self, X_train: TimeSeriesDataset, y_train=None, budget=None, **kwargs):
        start_time = time.time()
        if budget is not None:
            deltabudget = datetime.timedelta(seconds=budget)
        else:
            deltabudget = None
        X_train = self.enrich(X_train)
        super().fit(X_train, y_train, budget, **kwargs)

        self.batch_size = kwargs.get("batch_size", 64)
        self.horizon = kwargs.get("period", 1)
        self.feature_cols = X_train.time_varying_known_reals
        self.target_col = X_train.target_names[0]

        train_dataset = DataframeDataset(
            X_train.train_data,
            self.target_col,
            self.feature_cols,
            self.horizon,
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        if not X_train.test_data.empty:
            val_dataset = DataframeDataset(
                X_train.test_data,
                self.target_col,
                self.feature_cols,
                self.horizon,
            )
        else:
            val_dataset = DataframeDataset(
                X_train.train_data.sample(frac=0.2, random_state=kwargs.get("random_state", 0)),
                self.target_col,
                self.feature_cols,
                self.horizon,
            )

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        model = TCNForecaster(
            len(self.feature_cols),
            self.horizon,
            [2 ** self.params["num_hidden"]] * self.params["num_levels"],
            self.params["kernel_size"],
            self.params["dropout"],
        )

        pl_module = TCNForecasterLightningModule(model, self.params["learning_rate"])

        # Training loop
        # gpus is deprecated in v1.7 and removed in v2.0
        # accelerator="auto" can cast all condition.
        trainer = pl.Trainer(
            max_epochs=kwargs.get("max_epochs", 10),
            accelerator="auto",
            callbacks=[
                EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"),
                LearningRateMonitor(),
            ],
            logger=TensorBoardLogger(kwargs.get("log_dir", "logs/lightning_logs")),  # logging results to a tensorboard
            max_time=deltabudget,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        trainer.fit(
            pl_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        best_model = trainer.model
        self._model = best_model
        train_time = time.time() - start_time
        return train_time

    def predict(self, X):
        X = self.enrich(X)
        if isinstance(X, TimeSeriesDataset):
            df = X.X_val
        else:
            df = X
        dataset = DataframeDataset(
            df,
            self.target_col,
            self.feature_cols,
            self.horizon,
            train=False,
        )
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self._model.eval()
        raw_preds = []
        for batch_x in data_loader:
            raw_pred = self._model(batch_x)
            raw_preds.append(raw_pred)
        raw_preds = torch.cat(raw_preds, dim=0)
        preds = pd.Series(raw_preds.detach().numpy().ravel())
        return preds
