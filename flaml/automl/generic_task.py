import logging
import time

import pandas as pd
import numpy as np
from scipy.sparse import issparse
from sklearn.utils import shuffle
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    RepeatedKFold,
    GroupKFold,
    TimeSeriesSplit,
    GroupShuffleSplit,
)

from flaml.automl.data import concat
from flaml.automl.ml import default_cv_score_agg_func, get_val_loss
from flaml.automl.model import (
    XGBoostSklearnEstimator,
    XGBoostLimitDepthEstimator,
    RandomForestEstimator,
    LGBMEstimator,
    LRL1Classifier,
    LRL2Classifier,
    CatBoostEstimator,
    ExtraTreesEstimator,
    KNeighborsEstimator,
    TransformersEstimator,
    TransformersEstimatorModelSelection,
)
from flaml.automl.task import (
    CLASSIFICATION,
    REGRESSION,
    TOKENCLASSIFICATION,
    Task,
    get_classification_objective,
)
from flaml.config import RANDOM_SEED

logger = logging.getLogger(__name__)


class GenericTask(Task):
    estimators = {
        "xgboost": XGBoostSklearnEstimator,
        "xgb_limitdepth": XGBoostLimitDepthEstimator,
        "rf": RandomForestEstimator,
        "lgbm": LGBMEstimator,
        "lrl1": LRL1Classifier,
        "lrl2": LRL2Classifier,
        "catboost": CatBoostEstimator,
        "extra_tree": ExtraTreesEstimator,
        "kneighbor": KNeighborsEstimator,
        "transformer": TransformersEstimator,
        "transformer_ms": TransformersEstimatorModelSelection,
    }

    @staticmethod
    def _validate_data(
        automl,
        X_train_all,
        y_train_all,
        dataframe,
        label,
        eval_method,
        time_col=None,
        X_val=None,
        y_val=None,
        groups_val=None,
        groups=None,
    ):

        if X_train_all is not None and y_train_all is not None:
            assert (
                isinstance(X_train_all, np.ndarray)
                or issparse(X_train_all)
                or isinstance(X_train_all, pd.DataFrame)
            ), (
                "X_train_all must be a numpy array, a pandas dataframe, "
                "or Scipy sparse matrix."
            )
            assert isinstance(y_train_all, np.ndarray) or isinstance(
                y_train_all, pd.Series
            ), "y_train_all must be a numpy array or a pandas series."
            assert (
                X_train_all.size != 0 and y_train_all.size != 0
            ), "Input data must not be empty."
            if isinstance(X_train_all, np.ndarray) and len(X_train_all.shape) == 1:
                X_train_all = np.reshape(X_train_all, (X_train_all.size, 1))
            if isinstance(y_train_all, np.ndarray):
                y_train_all = y_train_all.flatten()
            assert (
                X_train_all.shape[0] == y_train_all.shape[0]
            ), "# rows in X_train must match length of y_train."
            automl._df = isinstance(X_train_all, pd.DataFrame)
            automl._nrow, automl._ndim = X_train_all.shape
            X, y = X_train_all, y_train_all
        elif dataframe is not None and label is not None:
            assert isinstance(
                dataframe, pd.DataFrame
            ), "dataframe must be a pandas DataFrame"
            assert label in dataframe.columns, "label must a column name in dataframe"
            automl._df = True
            X = dataframe.drop(columns=label)
            automl._nrow, automl._ndim = X.shape
            y = dataframe[label]
        else:
            raise ValueError("either X_train+y_train or dataframe+label are required")

        # check the validity of input dimensions for NLP tasks, so need to check _is_nlp_task not estimator
        if automl._state.task.is_nlp():
            from flaml.automl.nlp.utils import is_a_list_of_str

            is_all_str = True
            is_all_list = True
            for column in X.columns:
                assert X[column].dtype.name in (
                    "object",
                    "string",
                ), "If the task is an NLP task, X can only contain text columns"
                for each_cell in X[column]:
                    if each_cell is not None:
                        is_str = isinstance(each_cell, str)
                        is_list_of_int = isinstance(each_cell, list) and all(
                            isinstance(x, int) for x in each_cell
                        )
                        is_list_of_str = is_a_list_of_str(each_cell)
                        if automl._state.task == TOKENCLASSIFICATION:
                            assert is_list_of_str, (
                                "For the token-classification task, the input column needs to be a list of string,"
                                "instead of string, e.g., ['EU', 'rejects','German', 'call','to','boycott','British','lamb','.',].",
                                "For more examples, please refer to test/nlp/test_autohf_tokenclassification.py",
                            )
                        else:
                            assert is_str or is_list_of_int, (
                                "Each column of the input must either be str (untokenized) "
                                "or a list of integers (tokenized)"
                            )
                        is_all_str &= is_str
                        is_all_list &= is_list_of_int or is_list_of_str
            assert is_all_str or is_all_list, (
                "Currently FLAML only supports two modes for NLP: either all columns of X are string (non-tokenized), "
                "or all columns of X are integer ids (tokenized)"
            )

        if issparse(X_train_all):
            automl._transformer = automl._label_transformer = False
            automl._X_train_all, automl._y_train_all = X, y
        else:
            from flaml.automl.data import DataTransformer

            automl._transformer = DataTransformer()

            (
                automl._X_train_all,
                automl._y_train_all,
            ) = automl._transformer.fit_transform(X, y, automl._state.task)
            automl._label_transformer = automl._transformer.label_transformer
            if automl._state.task == TOKENCLASSIFICATION:
                if hasattr(automl._label_transformer, "label_list"):
                    automl._state.fit_kwargs.update(
                        {"label_list": automl._label_transformer.label_list}
                    )
                elif "label_list" not in automl._state.fit_kwargs:
                    for (
                        each_fit_kwargs
                    ) in automl._state.fit_kwargs_by_estimator.values():
                        assert (
                            "label_list" in each_fit_kwargs
                        ), "For the token-classification task, you must either (1) pass token labels; or (2) pass id labels and the label list. "
                        "Please refer to the documentation for more details: https://microsoft.github.io/FLAML/docs/Examples/AutoML-NLP#a-simple-token-classification-example"
            automl._feature_names_in_ = (
                automl._X_train_all.columns.to_list()
                if hasattr(automl._X_train_all, "columns")
                else None
            )

        automl._sample_weight_full = automl._state.fit_kwargs.get(
            "sample_weight"
        )  # NOTE: _validate_data is before kwargs is updated to fit_kwargs_by_estimator
        if X_val is not None and y_val is not None:
            assert (
                isinstance(X_val, np.ndarray)
                or issparse(X_val)
                or isinstance(X_val, pd.DataFrame)
            ), (
                "X_val must be None, a numpy array, a pandas dataframe, "
                "or Scipy sparse matrix."
            )
            assert isinstance(y_val, np.ndarray) or isinstance(
                y_val, pd.Series
            ), "y_val must be None, a numpy array or a pandas series."
            assert X_val.size != 0 and y_val.size != 0, (
                "Validation data are expected to be nonempty. "
                "Use None for X_val and y_val if no validation data."
            )
            if isinstance(y_val, np.ndarray):
                y_val = y_val.flatten()
            assert (
                X_val.shape[0] == y_val.shape[0]
            ), "# rows in X_val must match length of y_val."
            if automl._transformer:
                automl._state.X_val = automl._transformer.transform(X_val)
            else:
                automl._state.X_val = X_val
            # If it's NLG_TASKS, y_val is a pandas series containing the output sequence tokens,
            # so we cannot use label_transformer.transform to process it
            if automl._label_transformer:
                automl._state.y_val = automl._label_transformer.transform(y_val)
            else:
                automl._state.y_val = y_val
        else:
            automl._state.X_val = automl._state.y_val = None
        if groups is not None and len(groups) != automl._nrow:
            # groups is given as group counts
            automl._state.groups = np.concatenate(
                [[i] * c for i, c in enumerate(groups)]
            )
            assert (
                len(automl._state.groups) == automl._nrow
            ), "the sum of group counts must match the number of examples"
            automl._state.groups_val = (
                np.concatenate([[i] * c for i, c in enumerate(groups_val)])
                if groups_val is not None
                else None
            )
        else:
            automl._state.groups_val = groups_val
            automl._state.groups = groups

    @staticmethod
    def _prepare_data(automl, eval_method, split_ratio, n_splits, time_col=None):
        assert time_col is None, "Time col is only relevant for time series tasks"
        X_val, y_val = automl._state.X_val, automl._state.y_val
        if issparse(X_val):
            X_val = X_val.tocsr()
        X_train_all, y_train_all = automl._X_train_all, automl._y_train_all
        if issparse(X_train_all):
            X_train_all = X_train_all.tocsr()
        if (
            automl._state.task in CLASSIFICATION
            and automl._auto_augment
            and automl._state.fit_kwargs.get("sample_weight")
            is None  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
            and automl._split_type in ["stratified", "uniform"]
            and automl._state.task != TOKENCLASSIFICATION
        ):
            # logger.info(f"label {pd.unique(y_train_all)}")
            label_set, counts = np.unique(y_train_all, return_counts=True)
            # augment rare classes
            rare_threshld = 20
            rare = counts < rare_threshld
            rare_label, rare_counts = label_set[rare], counts[rare]
            for i, label in enumerate(rare_label):
                count = rare_count = rare_counts[i]
                rare_index = y_train_all == label
                n = len(y_train_all)
                while count < rare_threshld:
                    if automl._df:
                        X_train_all = concat(
                            X_train_all, X_train_all.iloc[:n].loc[rare_index]
                        )
                    else:
                        X_train_all = concat(
                            X_train_all, X_train_all[:n][rare_index, :]
                        )
                    if isinstance(y_train_all, pd.Series):
                        y_train_all = concat(
                            y_train_all, y_train_all.iloc[:n].loc[rare_index]
                        )
                    else:
                        y_train_all = np.concatenate(
                            [y_train_all, y_train_all[:n][rare_index]]
                        )
                    count += rare_count
                logger.info(f"class {label} augmented from {rare_count} to {count}")
        SHUFFLE_SPLIT_TYPES = ["uniform", "stratified"]
        if automl._split_type in SHUFFLE_SPLIT_TYPES:
            if automl._sample_weight_full is not None:
                X_train_all, y_train_all, automl._state.sample_weight_all = shuffle(
                    X_train_all,
                    y_train_all,
                    automl._sample_weight_full,
                    random_state=RANDOM_SEED,
                )
                automl._state.fit_kwargs[
                    "sample_weight"
                ] = (
                    automl._state.sample_weight_all
                )  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
            else:
                X_train_all, y_train_all = shuffle(
                    X_train_all, y_train_all, random_state=RANDOM_SEED
                )
            if automl._df:
                X_train_all.reset_index(drop=True, inplace=True)
                if isinstance(y_train_all, pd.Series):
                    y_train_all.reset_index(drop=True, inplace=True)

        X_train, y_train = X_train_all, y_train_all
        automl._state.groups_all = automl._state.groups
        if X_val is None and eval_method == "holdout":
            # if eval_method = holdout, make holdout data
            if automl._split_type == "time":
                if (
                    "sample_weight" in automl._state.fit_kwargs
                ):  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
                    (
                        X_train,
                        X_val,
                        y_train,
                        y_val,
                        automl._state.fit_kwargs[
                            "sample_weight"
                        ],  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
                        automl._state.weight_val,
                    ) = train_test_split(
                        X_train_all,
                        y_train_all,
                        automl._state.fit_kwargs[
                            "sample_weight"
                        ],  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
                        test_size=split_ratio,
                        shuffle=False,
                    )
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_all,
                        y_train_all,
                        test_size=split_ratio,
                        shuffle=False,
                    )
            elif automl._split_type == "group":
                gss = GroupShuffleSplit(
                    n_splits=1, test_size=split_ratio, random_state=RANDOM_SEED
                )
                for train_idx, val_idx in gss.split(
                    X_train_all, y_train_all, automl._state.groups_all
                ):
                    if automl._df:
                        X_train = X_train_all.iloc[train_idx]
                        X_val = X_train_all.iloc[val_idx]
                    else:
                        X_train, X_val = X_train_all[train_idx], X_train_all[val_idx]
                    y_train, y_val = y_train_all[train_idx], y_train_all[val_idx]
                    automl._state.groups = automl._state.groups_all[train_idx]
                    automl._state.groups_val = automl._state.groups_all[val_idx]
            elif automl._state.task in CLASSIFICATION:
                # for classification, make sure the labels are complete in both
                # training and validation data
                label_set, first = np.unique(y_train_all, return_index=True)
                rest = []
                last = 0
                first.sort()
                for i in range(len(first)):
                    rest.extend(range(last, first[i]))
                    last = first[i] + 1
                rest.extend(range(last, len(y_train_all)))
                X_first = X_train_all.iloc[first] if automl._df else X_train_all[first]
                X_rest = X_train_all.iloc[rest] if automl._df else X_train_all[rest]
                y_rest = y_train_all[rest]
                stratify = y_rest if automl._split_type == "stratified" else None
                if (
                    "sample_weight" in automl._state.fit_kwargs
                ):  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
                    (
                        X_train,
                        X_val,
                        y_train,
                        y_val,
                        weight_train,
                        weight_val,
                    ) = train_test_split(
                        X_rest,
                        y_rest,
                        automl._state.fit_kwargs["sample_weight"][
                            rest
                        ],  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
                        test_size=split_ratio,
                        random_state=RANDOM_SEED,
                    )
                    weight1 = automl._state.fit_kwargs["sample_weight"][
                        first
                    ]  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
                    automl._state.weight_val = concat(weight1, weight_val)
                    automl._state.fit_kwargs[
                        "sample_weight"
                    ] = concat(  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
                        weight1, weight_train
                    )
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_rest,
                        y_rest,
                        test_size=split_ratio,
                        stratify=stratify,
                        random_state=RANDOM_SEED,
                    )
                X_train = concat(X_first, X_train)
                y_train = (
                    concat(label_set, y_train)
                    if automl._df
                    else np.concatenate([label_set, y_train])
                )
                X_val = concat(X_first, X_val)
                y_val = (
                    concat(label_set, y_val)
                    if automl._df
                    else np.concatenate([label_set, y_val])
                )
            elif automl._state.task in REGRESSION:
                if (
                    "sample_weight" in automl._state.fit_kwargs
                ):  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
                    (
                        X_train,
                        X_val,
                        y_train,
                        y_val,
                        automl._state.fit_kwargs[
                            "sample_weight"
                        ],  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
                        automl._state.weight_val,
                    ) = train_test_split(
                        X_train_all,
                        y_train_all,
                        automl._state.fit_kwargs[
                            "sample_weight"
                        ],  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
                        test_size=split_ratio,
                        random_state=RANDOM_SEED,
                    )
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_all,
                        y_train_all,
                        test_size=split_ratio,
                        random_state=RANDOM_SEED,
                    )
        automl._state.data_size = X_train.shape
        automl.data_size_full = len(y_train_all)
        automl._state.X_train, automl._state.y_train = X_train, y_train
        automl._state.X_val, automl._state.y_val = X_val, y_val
        automl._state.X_train_all = X_train_all
        automl._state.y_train_all = y_train_all
        if eval_method == "holdout":
            automl._state.kf = None
            return
        if automl._split_type == "group":
            # logger.info("Using GroupKFold")
            assert (
                len(automl._state.groups_all) == y_train_all.size
            ), "the length of groups must match the number of examples"
            assert (
                len(np.unique(automl._state.groups_all)) >= n_splits
            ), "the number of groups must be equal or larger than n_splits"
            automl._state.kf = GroupKFold(n_splits)
        elif automl._split_type == "stratified":
            # logger.info("Using StratifiedKFold")
            assert y_train_all.size >= n_splits, (
                f"{n_splits}-fold cross validation"
                f" requires input data with at least {n_splits} examples."
            )
            assert y_train_all.size >= 2 * n_splits, (
                f"{n_splits}-fold cross validation with metric=r2 "
                f"requires input data with at least {n_splits*2} examples."
            )
            automl._state.kf = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=1, random_state=RANDOM_SEED
            )
        elif automl._split_type == "time":
            automl._state.kf = TimeSeriesSplit(n_splits=n_splits)
        elif isinstance(automl._split_type, str):
            # logger.info("Using RepeatedKFold")
            automl._state.kf = RepeatedKFold(
                n_splits=n_splits, n_repeats=1, random_state=RANDOM_SEED
            )
        else:
            # logger.info("Using splitter object")
            automl._state.kf = automl._split_type
        if isinstance(automl._state.kf, GroupKFold):
            # automl._split_type is either "group" or a GroupKFold object
            automl._state.kf.groups = automl._state.groups_all

    @staticmethod
    def _decide_split_type(automl, split_type):
        if automl._state.task == "classification":
            # TODO: another usecase for deciding task based on y
            # Should instead be done at task creation time
            task_name = get_classification_objective(
                len(np.unique(automl._y_train_all))
            )
            automl._state.task.name = task_name
        if not isinstance(split_type, str):
            assert hasattr(split_type, "split") and hasattr(
                split_type, "get_n_splits"
            ), "split_type must be a string or a splitter object with split and get_n_splits methods."
            assert (
                not isinstance(split_type, GroupKFold)
                or automl._state.groups is not None
            ), "GroupKFold requires groups to be provided."
            automl._split_type = split_type
        elif automl._state.task in CLASSIFICATION:
            assert split_type in ["auto", "stratified", "uniform", "time", "group"]
            automl._split_type = (
                split_type
                if split_type != "auto"
                else automl._state.groups is None and "stratified" or "group"
            )
        elif automl._state.task in REGRESSION:
            assert split_type in ["auto", "uniform", "time", "group"]
            automl._split_type = split_type if split_type != "auto" else "uniform"
        elif automl._state.task == "rank":
            assert (
                automl._state.groups is not None
            ), "groups must be specified for ranking task."
            assert split_type in ["auto", "group"]
            automl._split_type = "group"
        elif automl.task.is_nlg():
            assert split_type in ["auto", "uniform", "time", "group"]
            automl._split_type = split_type if split_type != "auto" else "uniform"

    @staticmethod
    def _prepare_sample_train_data(automlstate, sample_size):
        return automlstate._prepare_sample_train_data(sample_size)

    @staticmethod
    def _preprocess(automl, X):
        return automl._preprocess(X)

    def default_estimator_list(self):
        estimator_list = super().default_estimator_list()
        if self.name not in ("regression", "rank"):
            estimator_list += ["lrl1"]
        return estimator_list

    def evaluate_model_CV(
        self,
        config,
        estimator,
        X_train_all,
        y_train_all,
        budget,
        kf,
        eval_metric,
        best_val_loss,
        cv_score_agg_func=None,
        log_training_metric=False,
        fit_kwargs={},
    ):
        if cv_score_agg_func is None:
            cv_score_agg_func = default_cv_score_agg_func
        start_time = time.time()
        val_loss_folds = []
        log_metric_folds = []
        metric = None
        train_time = pred_time = 0
        total_fold_num = 0
        n = kf.get_n_splits()
        X_train_split, y_train_split = X_train_all, y_train_all
        if self.name in CLASSIFICATION:
            labels = np.unique(y_train_all)
        else:
            labels = fit_kwargs.get(
                "label_list"
            )  # pass the label list on to compute the evaluation metric
        groups = None
        shuffle = getattr(kf, "shuffle", False)
        if isinstance(kf, RepeatedStratifiedKFold):
            kf = kf.split(X_train_split, y_train_split)
        elif isinstance(kf, GroupKFold):
            groups = kf.groups
            kf = kf.split(X_train_split, y_train_split, groups)
            shuffle = False
        else:
            kf = kf.split(X_train_split)
        rng = np.random.RandomState(2020)
        budget_per_train = budget / n
        if "sample_weight" in fit_kwargs:
            weight = fit_kwargs["sample_weight"]
            weight_val = None
        else:
            weight = weight_val = None
        for train_index, val_index in kf:
            if shuffle:
                train_index = rng.permutation(train_index)
            if isinstance(X_train_all, pd.DataFrame):
                X_train = X_train_split.iloc[train_index]
                X_val = X_train_split.iloc[val_index]
            else:
                X_train, X_val = X_train_split[train_index], X_train_split[val_index]
            y_train, y_val = y_train_split[train_index], y_train_split[val_index]
            estimator.cleanup()
            if weight is not None:
                fit_kwargs["sample_weight"], weight_val = (
                    weight[train_index],
                    weight[val_index],
                )
            if groups is not None:
                fit_kwargs["groups"] = groups[train_index]
                groups_val = groups[val_index]
            else:
                groups_val = None
            val_loss_i, metric_i, train_time_i, pred_time_i = get_val_loss(
                config,
                estimator,
                X_train,
                y_train,
                X_val,
                y_val,
                weight_val,
                groups_val,
                eval_metric,
                self,
                labels,
                budget_per_train,
                log_training_metric=log_training_metric,
                fit_kwargs=fit_kwargs,
            )
            if isinstance(metric_i, dict) and "intermediate_results" in metric_i:
                del metric_i["intermediate_results"]
            if weight is not None:
                fit_kwargs["sample_weight"] = weight
            total_fold_num += 1
            val_loss_folds.append(val_loss_i)
            log_metric_folds.append(metric_i)
            train_time += train_time_i
            pred_time += pred_time_i
            if time.time() - start_time >= budget:
                break
        val_loss, metric = cv_score_agg_func(val_loss_folds, log_metric_folds)
        n = total_fold_num
        pred_time /= n
        return val_loss, metric, train_time, pred_time
