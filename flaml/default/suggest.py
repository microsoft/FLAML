import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging
import pathlib
import json
from flaml.data import CLASSIFICATION, DataTransformer
from flaml.ml import get_estimator_class, get_classification_objective

LOCATION = pathlib.Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)
CONFIG_PREDICTORS = {}


def meta_feature(task, X_train, y_train):
    is_classification = task in CLASSIFICATION
    n_row = X_train.shape[0]
    n_feat = X_train.shape[1]
    n_class = len(np.unique(y_train)) if is_classification else 0
    percent_num = X_train.select_dtypes(include=np.number).shape[1] / n_feat
    return (n_row, n_feat, n_class, percent_num)


def load_config_predictor(estimator_name, task, location=None):
    key = f"{estimator_name}_{task}"
    predictor = CONFIG_PREDICTORS.get(key)
    if predictor:
        return predictor
    task = "multiclass" if task == "multi" else task
    try:
        location = location or LOCATION
        with open(f"{location}/{estimator_name}/{task}.json", "r") as f:
            CONFIG_PREDICTORS[key] = predictor = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Portfolio has not been built for {estimator_name} on {task} task."
        )
    return predictor


def suggest_config(task, X, y, estimator_or_predictor, location=None, k=None):
    """Suggest a list of configs for the given task and training data.

    The returned configs can be used as starting points for AutoML.fit().
    `FLAML_sample_size` is removed from the configs.
    """
    task = (
        get_classification_objective(len(np.unique(y)))
        if task == "classification"
        else task
    )
    predictor = (
        load_config_predictor(estimator_or_predictor, task, location)
        if isinstance(estimator_or_predictor, str)
        else estimator_or_predictor
    )
    assert predictor["version"] == "default"
    prep = predictor["preprocessing"]
    feature = meta_feature(task, X, y)
    feature = (np.array(feature) - np.array(prep["center"])) / np.array(prep["scale"])
    neighbors = predictor["neighbors"]
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit([x["features"] for x in neighbors])
    dist, ind = nn.kneighbors(feature.reshape(1, -1), return_distance=True)
    logger.info(f"metafeature distance: {dist.item()}")
    ind = int(ind.item())
    choice = neighbors[ind]["choice"] if k is None else neighbors[ind]["choice"][:k]
    configs = [predictor["portfolio"][x] for x in choice]
    for config in configs:
        hyperparams = config["hyperparameters"]
        if hyperparams and "FLAML_sample_size" in hyperparams:
            hyperparams.pop("FLAML_sample_size")
    return configs


def suggest_learner(
    task, X, y, estimator_or_predictor="all", estimator_list=None, location=None
):
    """Suggest best learner within estimator_list."""
    configs = suggest_config(task, X, y, estimator_or_predictor, location)
    if not estimator_list:
        return configs[0]["class"]
    for c in configs:
        if c["class"] in estimator_list:
            return c["class"]
    return estimator_list[0]


def suggest_hyperparams(task, X, y, estimator_or_predictor, location=None):
    """Suggest hyperparameter configurations and an estimator class.

    The configurations can be used to initialize the estimator class like lightgbm.LGBMRegressor.

    Example:

    ```python
    hyperparams, estimator_class = suggest_hyperparams("regression", X_train, y_train, "lgbm")
    model = estimator_class(**hyperparams)  # estimator_class is LGBMRegressor
    model.fit(X_train, y_train)
    ```

    Args:
        task: A string of the task type, e.g.,
            'classification', 'regression', 'ts_forecast', 'rank',
            'seq-classification', 'seq-regression'.
        X: A dataframe of training data in shape n*m.
            For 'ts_forecast' task, the first column of X_train
            must be the timestamp column (datetime type). Other
            columns in the dataframe are assumed to be exogenous
            variables (categorical or numeric).
        y: A series of labels in shape n*1.
        estimator_or_predictor: A str of the learner name or a dict of the learned config predictor.
            If a dict, it contains:
            - "version": a str of the version number.
            - "preprocessing": a dictionary containing:
                * "center": a list of meta feature value offsets for normalization.
                * "scale": a list of meta feature scales to normalize each dimension.
            - "neighbors": a list of dictionaries. Each dictionary contains:
                * "features": a list of the normalized meta features for a neighbor.
                * "choice": an integer of the configuration id in the portfolio.
            - "portfolio": a list of dictionaries, each corresponding to a configuration:
                * "class": a str of the learner name.
                * "hyperparameters": a dict of the config. The key "FLAML_sample_size" will be ignored.
        location: (Optional) A str of the location containing mined portfolio file.
            Only valid when the portfolio is a str, by default the location is flaml/default.

    Returns:
        hyperparams: A dict of the hyperparameter configurations.
        estiamtor_class: A class of the underlying estimator, e.g., lightgbm.LGBMClassifier.
    """
    config = suggest_config(task, X, y, estimator_or_predictor, location=location, k=1)[
        0
    ]
    estimator = config["class"]
    model_class = get_estimator_class(task, estimator)
    hyperparams = config["hyperparameters"]
    model = model_class(task=task, **hyperparams)
    estimator_class = model.estimator_class
    hyperparams = hyperparams and model.params
    return hyperparams, estimator_class


def preprocess_and_suggest_hyperparams(
    task,
    X,
    y,
    estimator_or_predictor,
    location=None,
):
    """Preprocess the data and suggest hyperparameters.

    Example:

    ```python
    hyperparams, estimator_class, X, y, feature_transformer, label_transformer = \
        preprocess_and_suggest_hyperparams("classification", X_train, y_train, "xgb_limitdepth")
    model = estimator_class(**hyperparams)  # estimator_class is XGBClassifier
    model.fit(X, y)
    X_test = feature_transformer.transform(X_test)
    y_pred = label_transformer.inverse_transform(pd.Series(model.predict(X_test).astype(int)))
    ```

    Args:
        task: A string of the task type, e.g.,
            'classification', 'regression', 'ts_forecast', 'rank',
            'seq-classification', 'seq-regression'.
        X: A dataframe of training data in shape n*m.
            For 'ts_forecast' task, the first column of X_train
            must be the timestamp column (datetime type). Other
            columns in the dataframe are assumed to be exogenous
            variables (categorical or numeric).
        y: A series of labels in shape n*1.
        estimator_or_predictor: A str of the learner name or a dict of the learned config predictor.
            "choose_xgb" means choosing between xgb_limitdepth and xgboost.
            If a dict, it contains:
            - "version": a str of the version number.
            - "preprocessing": a dictionary containing:
                * "center": a list of meta feature value offsets for normalization.
                * "scale": a list of meta feature scales to normalize each dimension.
            - "neighbors": a list of dictionaries. Each dictionary contains:
                * "features": a list of the normalized meta features for a neighbor.
                * "choice": a integer of the configuration id in the portfolio.
            - "portfolio": a list of dictionaries, each corresponding to a configuration:
                * "class": a str of the learner name.
                * "hyperparameters": a dict of the config. They key "FLAML_sample_size" will be ignored.
        location: (Optional) A str of the location containing mined portfolio file.
            Only valid when the portfolio is a str, by default the location is flaml/default.

    Returns:
        hyperparams: A dict of the hyperparameter configurations.
        estiamtor_class: A class of the underlying estimator, e.g., lightgbm.LGBMClassifier.
        X: the preprocessed X.
        y: the preprocessed y.
        feature_transformer: a data transformer that can be applied to X_test.
        label_transformer: a label transformer that can be applied to y_test.
    """
    dt = DataTransformer()
    X, y = dt.fit_transform(X, y, task)
    if "choose_xgb" == estimator_or_predictor:
        # choose between xgb_limitdepth and xgboost
        estimator_or_predictor = suggest_learner(
            task,
            X,
            y,
            estimator_list=["xgb_limitdepth", "xgboost"],
            location=location,
        )
    config = suggest_config(task, X, y, estimator_or_predictor, location=location, k=1)[
        0
    ]
    estimator = config["class"]
    model_class = get_estimator_class(task, estimator)
    hyperparams = config["hyperparameters"]
    model = model_class(task=task, **hyperparams)
    estimator_class = model.estimator_class
    X = model._preprocess(X)
    hyperparams = hyperparams and model.params

    class AutoMLTransformer:
        def transform(self, X):
            return model._preprocess(dt.transform(X))

    transformer = AutoMLTransformer()
    return hyperparams, estimator_class, X, y, transformer, dt.label_transformer
