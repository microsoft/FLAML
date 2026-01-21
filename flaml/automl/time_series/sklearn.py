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

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def make_lag_features(X: pd.DataFrame, y: pd.Series, lags: int):
    """Transform input data X, y into autoregressive form by creating `lags` columns.

    This function is called automatically by FLAML during the training process
    to convert time series data into a format suitable for sklearn-based regression
    models (e.g., lgbm, rf, xgboost). Users do NOT need to manually call this function
    or create lagged features themselves.

    Parameters
    ----------
    X : pandas.DataFrame
        Input feature DataFrame, which may contain temporal features and/or exogenous variables.

    y : array_like, (1d)
        Target vector (time series values to forecast).

    lags : int
        Number of lagged time steps to use as features.

    Returns
    -------
    pandas.DataFrame
        Shifted dataframe with `lags` columns for each original feature.
        The target variable y is also lagged to prevent data leakage
        (i.e., we use y(t-1), y(t-2), ..., y(t-lags) to predict y(t)).
    """
    lag_features = []

    # make sure we show y's _previous_ value to exclude data leaks
    X = X.reset_index(drop=True)
    X["lag_" + y.name] = y.shift(1).values

    X_lag = X.copy()
    for i in range(0, lags):
        X_lag.columns = [f"{c}_lag_{i}" for c in X.columns]
        lag_features.append(X_lag)
        X_lag = X_lag.shift(1)

    X_lags = pd.concat(lag_features, axis=1)
    X_out = X_lags.dropna().reset_index(drop=True)
    assert len(X_out) + lags == len(X)
    return X_out


class SklearnWrapper:
    """Wrapper class for using sklearn-based models for time series forecasting.

    This wrapper automatically handles the transformation of time series data into
    a supervised learning format by creating lagged features. It trains separate
    models for each step in the forecast horizon.

    Users typically don't interact with this class directly - it's used internally
    by FLAML when sklearn-based estimators (lgbm, rf, xgboost, etc.) are selected
    for time series forecasting tasks.
    """

    def __init__(
        self,
        model_class: type,
        horizon: int,
        lags: int,
        init_params: dict = None,
        fit_params: dict = None,
        pca_features: bool = False,
    ):
        init_params = init_params if init_params else {}
        self.fit_params = fit_params if fit_params else {}
        self.lags = lags
        self.horizon = horizon
        # TODO: use multiregression where available
        self.models = [model_class(**init_params) for _ in range(horizon)]
        self.pca_features = pca_features
        if self.pca_features:
            self.norm = StandardScaler()
            self.pca = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        if "is_retrain" in kwargs:
            kwargs.pop("is_retrain")
        self._X = X
        self._y = y

        fit_params = {**self.fit_params, **kwargs}
        X_feat = make_lag_features(X, y, self.lags)
        if self.pca_features:
            X_trans = self.norm.fit_transform(X_feat)

            cum_expl_var = np.cumsum(PCA(svd_solver="full").fit(X_trans).explained_variance_ratio_)
            self.pca = PCA(svd_solver="full", n_components=np.argmax(1 - cum_expl_var < 1e-6))
            X_trans = self.pca.fit_transform(X_trans)
        else:
            X_trans = X_feat

        for i, model in enumerate(self.models):
            offset = i + self.lags
            if len(X) - offset > 2:
                # series with length 2 will meet All features are either constant or ignored.
                # TODO: see why the non-constant features are ignored. Selector?
                model.fit(X_trans[: len(X) - offset], y[offset:], **fit_params)
            elif len(X) > offset and "catboost" not in str(model).lower():
                model.fit(X_trans[: len(X) - offset], y[offset:], **fit_params)
            else:
                print("[INFO]: Length of data should longer than period + lags.")
        return self

    def predict(self, X, X_train=None, y_train=None):
        if X_train is None:
            X_train = self._X
        if y_train is None:
            y_train = self._y

        X_train = X_train.reset_index(drop=True)
        X_train[self._y.name] = y_train.values
        Xall = pd.concat([X_train, X], axis=0).reset_index(drop=True)
        y = Xall.pop(self._y.name)

        X_feat = make_lag_features(Xall[: len(X_train) + 1], y[: len(X_train) + 1], self.lags)
        if self.pca_features:
            X_trans = self.pca.transform(self.norm.transform(X_feat))
        else:
            X_trans = X_feat
        # predict all horizons from the latest features vector
        preds = pd.Series([m.predict(X_trans[-1:])[0] for m in self.models])
        if len(preds) < len(X):
            # recursive call if len(X) > trained horizon
            y_train = pd.concat([y_train, preds], axis=0, ignore_index=True)
            preds = pd.concat(
                [
                    preds,
                    self.predict(
                        X=Xall[len(y_train) :],
                        X_train=Xall[: len(y_train)],
                        y_train=y_train,
                    ),
                ],
                axis=0,
                ignore_index=True,
            )
        if len(preds) > len(X):
            preds = preds[: len(X)]

        preds.index = X.index
        # TODO: do we want auto-clipping?
        # return self._clip_predictions(preds)
        return preds

    # TODO: fix
    # @staticmethod
    # def _adjust_holidays(X):
    #     """Transform 'holiday' columns to binary feature.
    #
    #     Parameters
    #     ----------
    #     X : pandas.DataFrame
    #         Input features with 'holiday' column.
    #
    #     Returns
    #     -------
    #     pandas.DataFrame
    #         Holiday feature in numeric form
    #     """
    #     return X.assign(
    #         **{col: X[col] != "" for col in X.filter(like="_holiday_").columns}
    #     )
