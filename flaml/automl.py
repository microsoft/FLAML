'''!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under MICROSOFT RESEARCH LICENSE TERMS. See LICENSE file in the
 * project root for license information.
'''
import time
import warnings
from functools import partial
import ast
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, \
    RepeatedKFold
from sklearn.utils import shuffle
import pandas as pd

from .model_compute_helper import compute_estimator, train_estimator, \
    get_estimator_name_from_log
from .config import MIN_SAMPLE_TRAIN, MEM_THRES, ETI_INI, \
    SMALL_LARGE_THRES, CV_HOLDOUT_THRESHOLD, SPLIT_RATIO, N_SPLITS
from .util import save_info_helper, get_classification_objective, concat
from .search import ParamSearch

import logging
logger = logging.getLogger(__name__)


def prepare_sample_train_data(X_train, y_train, X_val, y_val, sample_size):
    full_size = len(y_train)
    if sample_size <= full_size:
        if isinstance(X_train, pd.DataFrame):
            sampled_X_train = X_train.iloc[:sample_size]
        else:
            sampled_X_train = X_train[:sample_size]
        sampled_y_train = y_train[:sample_size]
        sampled_X_val, sampled_y_val = X_val, y_val
    else:
        sampled_X_train, sampled_y_train = concat(X_train, X_val), \
            np.concatenate([y_train, y_val])
        sampled_X_val, sampled_y_val = [], []
    return sampled_X_train, sampled_y_train, sampled_X_val, sampled_y_val


class AutoML:
    '''The AutoML class

    Attributes:
        best_estimator: A string indicating the best estimator found.
        selected: An object from which we can extract the best configuration
            as selected.best_config.
        model: An object with predict() and predict_proba() method (for
            classification), storing the best trained model.
        model_history: A dictionary of time->model, storing the time when
            the best model is updated and the best model at that time
        config_history: A dictionary of time->(estimator, config), storing the
            time when the best model is updated and the best (estimator,
            config)
        classes_: A list of n_classes elements for class labels
        best_iteration: An integer of the iteration number where the best
            config is found

    Typical usage example:

        automl_experiment = AutoML()
        automl_settings = {
            "time_budget": 60,
            "metric": 'accuracy',
            "objective_name": 'classification',
            "log_file_name": 'test/mylog.log',
        }
        automl_experiment.fit(X_train_all = X_train, y_train_all = y_train,
            **automl_settings)
    '''


    def __init__(self):
        self._eti_ini = ETI_INI
        self._custom_learners = {}
        self._config_space_info = {}
        self._custom_size_estimate = {}
        self._track_iter = 0

    @property
    def model_history(self):
        return self._model_history

    @property
    def config_history(self):
        return self._config_history

    @property
    def model(self):
        if self._model: return self._model.model
        else: return None

    @property
    def best_estimator(self):
        return self._best_estimator

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def selected(self):
        return self._selected

    @property
    def classes_(self):
        if self.label_transformer: 
            return self.label_transformer.classes_.tolist()
        if self._model:
            return self._model.model.classes_.tolist()
        return None

    def predict(self, X_test):
        '''Predict label from features.

        Args:
            X_test: A numpy array of featurized instances, shape n*m.

        Returns:
            A numpy array of shape n*1 -- each element is a predicted class
            label for an instance.
        '''
        X_test = self.preprocess(X_test)
        y_pred = self._model.predict(X_test)
        if y_pred.ndim > 1: y_pred = y_pred.flatten()
        if self.label_transformer:
            return self.label_transformer.inverse_transform(pd.Series(
                y_pred))
        else: return y_pred

    def predict_proba(self, X_test):
        '''Predict the probability of each class from features, only works for
        classification problems.

        Args:
            X_test: A numpy array of featurized instances, shape n*m.

        Returns:
            A numpy array of shape n*c. c is the # classes. Each element at
            (i,j) is the probability for instance i to be in class j.
        '''
        X_test = self.preprocess(X_test)
        proba = self._model.predict_proba(X_test)
        return proba

    def preprocess(self, X):
        if scipy.sparse.issparse(X): 
            X = X.tocsr()
        if self.transformer:
            X = self.transformer.transform(X)
        return X

    def _validate_data(self, X_train_all, y_train_all, dataframe, label):
        if X_train_all is not None and y_train_all is not None:
            if not (isinstance(X_train_all, np.ndarray) or
                    scipy.sparse.issparse(X_train_all)):
                raise ValueError(
                    "X_train_all must be a Numpy array or Scipy sparse matrix.")
            if not isinstance(y_train_all, np.ndarray):
                raise ValueError("y_train_all must be a Numpy array.")
            if X_train_all.size == 0 or y_train_all.size == 0:
                raise ValueError("Input data must not be empty.")
            y_train_all = y_train_all.flatten()
            if X_train_all.shape[0] != y_train_all.shape[0]:
                raise ValueError(
            "# rows in X_train must match length of y_train.")
            self.df = False
            self.nrow, self.ndim = X_train_all.shape
            if scipy.sparse.issparse(X_train_all): 
                self.transformer = self.label_transformer = False
                self.X_train_all, self.y_train_all = X_train_all, y_train_all
                return
            X, y = X_train_all, y_train_all
        elif dataframe is not None and label is not None:
            if not isinstance(dataframe, pd.DataFrame):
                raise ValueError("dataframe must be a pandas DataFrame")
            if not label in dataframe.columns:
                raise ValueError("label must a column name in dataframe")
            self.df = True
            self.dataframe, self.label = dataframe, label
            X = dataframe.drop(columns=label)
            self.nrow, self.ndim = X.shape
            y = dataframe[label]
        else:
            raise ValueError(
        "either X_train_all+y_train_all or dataframe+label need to be provided.")
        from .util import DataTransformer
        self.transformer = DataTransformer()
        self.X_train_all, self.y_train_all = self.transformer.fit_transform(
            X, y, self.objective_name)
        self.label_transformer = self.transformer.label_transformer
        
    def _prepare_data(self,
                      eval_method,
                      split_ratio,
                      n_splits):
        X_val = y_val = None
        X_train_all, y_train_all = self.X_train_all, self.y_train_all
        if scipy.sparse.issparse(X_train_all): 
            X_train_all = X_train_all.tocsr()
        if self.objective_name != 'regression':
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
                    if self.df:
                        X_train_all = concat(X_train_all,
                        X_train_all.iloc[:n].loc[rare_index])
                    else:
                        X_train_all = concat(X_train_all,
                        X_train_all[:n][rare_index,:])
                    if isinstance(y_train_all, pd.Series):
                        y_train_all = concat(y_train_all,
                        y_train_all.iloc[:n].loc[rare_index])
                    else:
                        y_train_all = np.concatenate([y_train_all,
                        y_train_all[:n][rare_index]])
                    count += rare_count
                logger.debug(
                    f"class {label} augmented from {rare_count} to {count}")
        X_train_all, y_train_all = shuffle(
            X_train_all, y_train_all, random_state=202020)
        if self.df:
            X_train_all.reset_index(drop=True, inplace=True)
            if isinstance(y_train_all, pd.Series):
                y_train_all.reset_index(drop=True, inplace=True)

        # logger.info(y_train_all)
        if self.objective_name != 'regression' and eval_method == 'holdout':
            label_set, first = np.unique(y_train_all, return_index=True)
            rest = []
            last = 0
            first.sort()
            for i in range(len(first)):
                rest.extend(range(last, first[i]))
                last = first[i] + 1
            rest.extend(range(last, len(y_train_all)))
            X_first = X_train_all.iloc[first] if self.df else X_train_all[
                first]
            X_rest = X_train_all.iloc[rest] if self.df else X_train_all[rest]
            y_rest = y_train_all.iloc[rest] if isinstance(
                y_train_all, pd.Series) else y_train_all[rest]
        if eval_method == 'holdout':
            if self.objective_name == 'regression':
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_all,
                    y_train_all,
                    test_size=split_ratio,
                    random_state=1)
            else:
                stratify = y_rest if self.split_type=='stratified' else None
                X_train, X_val, y_train, y_val = train_test_split(
                    X_rest,
                    y_rest,
                    test_size=split_ratio,
                    stratify=stratify, 
                    random_state=1)                                                                        
                X_train = concat(X_first, X_train)
                y_train = concat(label_set,
                 y_train) if self.df else np.concatenate([label_set, y_train])
                X_val = concat(X_first, X_val)
                y_val = concat(label_set,
                 y_val) if self.df else np.concatenate([label_set, y_val])
                _, y_train_counts_elements = np.unique(y_train,
                    return_counts=True)
                _, y_val_counts_elements = np.unique(y_val,
                    return_counts=True)
                logger.debug(
                    f"""{self.split_type} split for y_train \
                        {y_train_counts_elements}, \
                        y_val {y_val_counts_elements}""")
        else:
            X_train, y_train = X_train_all, y_train_all
        self.data_size = X_train.shape[0]
        self.X_train, self.y_train, self.X_val, self.y_val = (
            X_train, y_train, X_val, y_val)
        self._sample = partial(prepare_sample_train_data,
                            self.X_train,
                            self.y_train,
                            self.X_val,
                            self.y_val)
        if self.split_type == "stratified":
            logger.info("Using StratifiedKFold")
            self.kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1,
            random_state=202020)
        else:
            logger.info("Using RepeatedKFold")
            self.kf = RepeatedKFold(n_splits=n_splits, n_repeats=1,
            random_state=202020)

    def _compute_with_config_base(self,
                                  X_test,
                                  y_test,
                                  objective_name,
                                  eval_method,
                                  metric,
                                  compute_train_loss,
                                  n_jobs,
                                  estimator,
                                  config,
                                  sample_size):
        sampled_X_train, sampled_y_train, sampled_X_val, sampled_y_val = \
            self._sample(sample_size)
        time_left = self.time_budget - self.time_from_start
        budget = time_left if sample_size == self.data_size else \
            time_left/2*sample_size/self.data_size
        return compute_estimator(sampled_X_train,
                                 sampled_y_train,
                                 sampled_X_val,
                                 sampled_y_val,
                                 budget,
                                 self.kf,
                                 config,
                                 objective_name,
                                 estimator,
                                 eval_method,
                                 metric,
                                 self.best_loss,
                                 n_jobs,
                                 self._custom_learners.get(estimator),
                                 compute_train_loss)

    def _train_with_config_base(self,
                                objective_name,
                                n_jobs,
                                estimator,
                                config,
                                sample_size):
        sampled_X_train, sampled_y_train, __, _ = self._sample(sample_size)
        budget = None if self.time_budget is None else (self.time_budget -
         self.time_from_start)
        MODEL, train_time = train_estimator(
            sampled_X_train,
            sampled_y_train,
            config,
            objective_name,
            estimator,
            n_jobs,
            self._custom_learners.get(estimator),
            budget)
        return MODEL, train_time

    def add_learner(self,
                    learner_name,
                    learner,
                    size_estimate=lambda config: 'unknown',
                    cost_relative2lgbm=1):
        '''Add a customized learner

        Args:
            learner_name: A string of the learner's name
            learner: A subclass of BaseEstimator
            size_estimate: A function from a config to its memory size in float
            cost_relative2lgbm: A float number for the training cost ratio with
                respect to lightgbm (when both use the initial config)
        '''
        self._custom_learners[learner_name] = learner
        self._eti_ini[learner_name] = cost_relative2lgbm
        self._config_space_info[learner_name] = \
            learner().params_configsearch_info  # config_search_list
        self._custom_size_estimate[learner_name] = size_estimate

    def get_estimator_from_log(self, log_file_name, line_number, objective):
        '''Get the estimator from log file

        Args:
            log_file_name: A string of the log file name
            line_number: An integer of the line number in the file, 
                1 corresponds to the first trial
            objective: A string of the objective name, 
                'binary', 'multi', or 'regression'

        Returns:
            An estimator object for the given configuration
        '''

        with open(log_file_name) as file_:
            line = file_.readlines()[line_number]
            data = line.split('\t')
            estimator = get_estimator_name_from_log(data[8])
            config = ast.literal_eval(data[5])

        estimator, _ = train_estimator(
            None, None, config, objective, estimator,
            estimator_class=self._custom_learners.get(estimator)
            )
        return estimator

    def retrain_from_log(self,
                         log_file_name,
                         X_train_all=None,
                         y_train_all=None,
                         dataframe=None,
                         label=None,
                         time_budget=0,
                         objective_name='classification',
                         eval_method='auto',
                         split_ratio=SPLIT_RATIO,
                         n_splits=N_SPLITS,
                         split_type="stratified",
                         n_jobs=1,
                         train_best=True,
                         train_full=False,
                         line_number=0):
        '''Retrain from log file to adjust the time effect of last iteration

        Args:
            time_budget: A float number of the time budget in seconds 
            log_file_name: A string of the log file name
            X_train_all: A numpy array of training data in shape n*m
            y_train_all: A numpy array of labels in shape n*1
            objective_name: A string of the task type, e.g.,
                'classification', 'regression'
            eval_method: A string of resampling strategy, one of
                ['auto', 'cv', 'holdout']
            split_ratio: A float of the validation data percentage for holdout
            n_splits: An integer of the number of folds for cross-validation
            n_jobs: An integer of the number of threads for training
            train_best: A boolean of whether to train the best config in the 
                time budget; if false, train the last config in the budget
            train_full: A boolean of whether to train on the full data. If true,
                eval_method and sample_size in the log file will be ignored
            line_number: An integer of the line number in the file, 
                1 corresponds to the first trial; 0 would be ignored
                when line_number>0, time_budget will be ignored
        '''
        self.objective_name = objective_name
        self._validate_data(X_train_all, y_train_all, dataframe, label)

        logger.info('log file name {}'.format(log_file_name))

        best_config = None
        best_val_loss = float('+inf')
        best_estimator = None
        sample_size = None
        time_used = 0.0
        training_duration = 0
        
        with open(log_file_name) as file_:
            if line_number:
                best = file_.readlines()[line_number].split('\t')
            else:
                for line in file_:
                    data = line.split('\t')
                    try:
                        _ = int(data[0])
                    except:
                        continue
                    time_used = float(data[4])
                    if time_used <= time_budget:
                        training_duration = time_used
                        val_loss = float(data[3])
                        if val_loss <= best_val_loss or not train_best:
                            if val_loss == best_val_loss and train_best:
                                size = int(data[9])
                                if size > sample_size:
                                    best = data
                                    best_val_loss = val_loss
                                    sample_size = size
                            else:
                                best = data
                                size = int(data[9])
                                best_val_loss = val_loss
                                sample_size = size
                if not training_duration: 
                    from .model_helper import BaseEstimator
                    self._model = BaseEstimator()
                    self._model.model = None
                    return training_duration
        best_estimator = get_estimator_name_from_log(best[8])
        best_config = ast.literal_eval(best[5])
        sample_size = len(self.y_train_all) if train_full else int(best[9])

        logger.info(
            'estimator = {}, config = {}, #training instances = {}'.format(
                best_estimator, best_config, sample_size))
        if not hasattr(self, '_train_with_config') or True:
            # Partially copied from fit() function
            # Initilize some attributes required for retrain_from_log
            np.random.seed(0)
            self.objective_name = objective_name
            if self.objective_name == 'classification':
                objective_name = get_classification_objective(
                    len(np.unique(self.y_train_all)))
                assert split_type in ["stratified", "uniform"]
                self.split_type = split_type
            else:
                self.split_type = "uniform"
            if line_number:
                eval_method = 'cv'
            elif eval_method == 'auto':
                eval_method = self._decide_eval_method(time_budget)
            self.modelcount = 0
            self._prepare_data(eval_method, split_ratio, n_splits)
            self._train_with_config = partial(AutoML._train_with_config_base,
                                            self,
                                            objective_name,
                                            n_jobs)
        self.time_budget = None
        self._model = self._train_with_config(
            best_estimator, best_config, sample_size)[0]
        return training_duration            

    def _decide_eval_method(self, time_budget):
        nrow, dim = self.nrow, self.ndim
        if nrow * dim / 0.9 < SMALL_LARGE_THRES * (
            time_budget / 3600) and nrow < CV_HOLDOUT_THRESHOLD:
            # time allows or sampling can be used and cv is necessary
            eval_method = 'cv'
        else:
            eval_method = 'holdout'
        return eval_method

    def fit(self,
            X_train_all=None,
            y_train_all=None,
            dataframe=None,
            label=None,
            metric='auto',
            objective_name='classification',
            n_jobs=-1,
            log_file_name='default.log',
            estimator_list='auto',
            time_budget=60,
            max_iter=1000000,
            sample=True,
            ensemble=False,
            eval_method='auto',
            log_type='better',
            model_history=False,
            split_ratio=SPLIT_RATIO,
            n_splits=N_SPLITS,
            log_training_metric=False,
            mem_thres=MEM_THRES,
            fix_range=False,
            X_test=None,
            y_test=None,
            retrain_full=True,
            reset_type='init_gaussian',
            split_type="stratified",
            base_change='sqrtK',
            learner_selector='sample',
            use_dual_dir=True,
            move_type='geo',
            ):
        '''Find a model for a given task

        Args:
            X_train_all: A numpy array of training data in shape n*m
            y_train_all: A numpy array of labels in shape n*1
            dataframe: A dataframe of training data including label column
            label: A str of the label column name
                Note: If X_train_all and y_train_all are provided, 
                dataframe and label are ignored;
                If not, dataframe and label must be provided.
            metric: A string of the metric name or a function,
                e.g., 'accuracy','roc_auc','f1','log_loss','mae','mse','r2'
                if passing a customized metric function, the function needs to
                have the follwing signature
                
                def metric(X_test, y_test, estimator, labels, X_train, y_train):
                    return metric_to_minimize, metrics_to_log
                    
                which returns a float number as the minimization objective, 
                and a tuple of floats as the metrics to log
            objective_name: A string of the task type, e.g.,
                'classification', 'regression'
            n_jobs: An integer of the number of threads for training
            log_file_name: A string of the log file name
            estimator_list: A list of strings for estimator names, or 'auto'
                e.g., ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree']
            time_budget: A float number of the time budget in seconds
            max_iter: An integer of the maximal number of iterations
            sample: A boolean of whether to sample the training data during
                search
            eval_method: A string of resampling strategy, one of
                ['auto', 'cv', 'holdout']
            split_ratio: A float of the valiation data percentage for holdout
            n_splits: An integer of the number of folds for cross-validation
            mem_thres: A float of the memory size constraint in bytes
            log_type: A string of the log type, one of ['better', 'all', 'new']
                'better' only logs configs with better loss than previos iters
                'all' logs all the tried configs
                'new' only logs non-redundant configs
            model_history: A boolean of whether to keep the history of best
                models in the history property. Make sure memory is large
                enough if setting to True.
            log_training_metric: A boolean of whether to log the training 
                metric for each model. 
        '''
        self.objective_name = objective_name
        self._validate_data(X_train_all, y_train_all, dataframe, label)
        self.start_time_flag = time.time()
        np.random.seed(0)
        self.learner_selector = learner_selector

        if self.objective_name == 'classification':
            objective_name = get_classification_objective(
                len(np.unique(self.y_train_all)))
            assert split_type in ["stratified", "uniform"]
            self.split_type = split_type
        else:
            objective_name = self.objective_name
            self.split_type = "uniform"

        if 'auto' == estimator_list:
            estimator_list = ['lgbm', 'rf', 'catboost', 'xgboost', 'extra_tree']
            if 'regression' != self.objective_name:
                estimator_list += ['lrl1',]
        logger.info("List of ML learners in AutoML Run: {}".format(estimator_list))

        if eval_method == 'auto':
            eval_method = self._decide_eval_method(time_budget)
        self.eval_method = eval_method
        logger.info("Evaluation method: {}".format(eval_method))
        
        retrain_full = retrain_full and eval_method == 'holdout'
        sample &= (eval_method != 'cv')
        if 'auto' == metric:
            if 'binary' in objective_name:
                metric = 'roc_auc'
            elif 'multi' in objective_name:
                metric = 'log_loss'
            else:
                metric = 'r2'
        if metric in ['r2', 'accuracy', 'roc_auc', 'f1', 'ap']:
            error_metric = f"1-{metric}"
        elif isinstance(metric, str):
            error_metric = metric
        else:
            error_metric = 'customized metric'
        logger.info(f'Minimizing error metric: {error_metric}')
    
        self.save_helper = save_info_helper('FLAML', log_file_name)
        self._prepare_data(eval_method, split_ratio, n_splits)
        self._compute_with_config = partial(AutoML._compute_with_config_base,
                                            self,
                                            X_test,
                                            y_test,
                                            objective_name,
                                            eval_method,
                                            metric,
                                            log_training_metric,
                                            n_jobs)
        self._train_with_config = partial(AutoML._train_with_config_base,
                                          self,
                                          objective_name,
                                          n_jobs)
        self.searchers = {}
        # initialize the searchers
        self.eti = []
        self.best_loss = float('+inf')
        self.time_budget = time_budget
        self.best_train_time = 0
        self.time_from_start = 0
        self.estimator_index = -1
        self._best_iteration = 0
        self._model_history = {}
        self._config_history = {}
        self.max_iter_per_learner = 10000 # TODO
        self.iter_per_learner = dict([(e,0) for e in estimator_list])
        self.fullsize = False
        self._model = None
        self.ensemble = ensemble
        if ensemble: self.best_model = {}
        for self._track_iter in range(max_iter):
            if self.estimator_index == -1:
                estimator = estimator_list[0]
            else:
                estimator = self._select_estimator(estimator_list)
                if not estimator: break
            logger.info(f"iteration {self._track_iter}"
              f"  current learner {estimator}")
            if estimator in self.searchers:
                model = self.searchers[estimator].model
                improved = self.searchers[estimator].search1step(
                    global_best_loss=self.best_loss,
                    retrain_full=retrain_full,
                    mem_thres=mem_thres,
                    reset_type=reset_type)
            else:
                model = None
                self.searchers[estimator] = ParamSearch(
                    estimator,
                    self.data_size,
                    self._compute_with_config,
                    self._train_with_config,
                    self.save_helper,
                    MIN_SAMPLE_TRAIN if sample else self.data_size,
                    objective_name,
                    log_type,
                    base_change,
                    use_dual_dir,
                    move_type,
                    self._config_space_info.get(estimator),
                    self._custom_size_estimate.get(estimator),
                    split_ratio)
                self.searchers[estimator].search_begin(time_budget,
                                                       self.start_time_flag)
                if self.estimator_index == -1:
                    eti_base = self._eti_ini[estimator]
                    self.eti.append(
                        self.searchers[estimator]
                            .expected_time_improvement_search())
                    for e in estimator_list[1:]:
                        self.eti.append(
                            self._eti_ini[e]/eti_base*self.eti[0])
                    self.estimator_index = 0
            self.time_from_start = time.time() - self.start_time_flag
            # logger.info(f"{self.searchers[estimator].sample_size}, {data_size}")
            if self.searchers[estimator].sample_size == self.data_size:
                self.iter_per_learner[estimator] += 1
                if not self.fullsize:
                    self.fullsize = True
            if self.searchers[estimator].best_loss < self.best_loss:
                self.best_loss = self.searchers[estimator].best_loss
                self._best_estimator = estimator
                self.best_train_time = self.searchers[estimator].train_time
                self._config_history[self.time_from_start] = (
                    estimator,
                    self.searchers[estimator].best_config)
                if model_history:
                    self._model_history[self.time_from_start] = self.searchers[
                        estimator].model.model
                elif self._model:
                    del self._model
                    self._model = None
                self._model = self.searchers[estimator].model
                self._best_iteration = self._track_iter
            if model and improved and not model_history:
                model.cleanup()

            logger.info(
    " at {:.1f}s,\tbest {}'s error={:.4f}\tbest {}'s error={:.4f}".format(
                          self.time_from_start,
                          estimator,
                          self.searchers[estimator].best_loss,
                          self._best_estimator,
                          self.best_loss))
                  
            if self.time_from_start >= time_budget:
                break
            if ensemble:
                time_left = self.time_from_start-time_budget
                time_ensemble = self.searchers[self._best_estimator].train_time
                if time_left < time_ensemble < 2*time_left:
                    break
            if self.searchers[
                estimator].train_time>time_budget-self.time_from_start:
                self.iter_per_learner[estimator] = self.max_iter_per_learner
        self.best_loss_info = self.save_helper.update_best()
        if self.searchers:
            self._selected = self.searchers[self._best_estimator]
            self._model = self._selected.model
            self.modelcount = sum(self.searchers[estimator].model_count
                                for estimator in self.searchers)
            logger.info(self._model.model)
            if ensemble:
                searchers = list(self.searchers.items())
                searchers.sort(key=lambda x:x[1].best_loss)
                estimators = [(x[0],x[1].model) for x in searchers[:2]]
                estimators += [(x[0],x[1].model) for x in searchers[2:]
                if x[1].best_loss<4*self._selected.best_loss]            
                logger.info(estimators)
                if objective_name != "regression":
                    from sklearn.ensemble import StackingClassifier as Stacker
                    for e in estimators:
                        e[1]._estimator_type = 'classifier'
                else:
                    from sklearn.ensemble import StackingRegressor as Stacker
                if self.best_estimator == 'nn':
                    # find the best performing non-NN model
                    best_m, best_loss = None, np.Inf
                    for e, s in self.searchers.items():
                        # break
                        if e!='nn' and s.best_loss < best_loss:
                            best_m, best_loss = s.model, s.best_loss
                else: 
                    best_m = self._model
                stacker = Stacker(estimators, best_m, n_jobs=n_jobs, 
                    passthrough = True)
                stacker.fit(self.X_train_all, self.y_train_all)
                self._model = stacker
                self._model.model = stacker
        else:
            self._selected = self._model = None
            self.modelcount = 0

        logger.info("fit succeeded")

    def __del__(self):
        if hasattr(self, '_model') and self._model and hasattr(
            self._model, 'cleanup'):
            self._model.cleanup()
            del self._model

    def _select_estimator(self, estimator_list):
        time_left = self.time_budget - self.time_from_start
        if  self.best_train_time < time_left < 2*self.best_train_time:
            best_searcher = self.searchers[self._best_estimator]
            config_sig = best_searcher.get_hist_config_sig(
                best_searcher.sample_size_full,
                best_searcher.best_config[0])
            if config_sig not in best_searcher.config_tried:
                # trainAll
                return self._best_estimator
        if self.learner_selector == 'roundrobin':
            self.estimator_index += 1
            if self.estimator_index == len(estimator_list):
                self.estimator_index = 0
            return estimator_list[self.estimator_index]
        min_expected_time, selected = np.Inf, None
        inv = []
        for i, estimator in enumerate(estimator_list):
            if estimator in self.searchers:
                searcher = self.searchers[estimator]
                if self.iter_per_learner[estimator]>=self.max_iter_per_learner:
                    inv.append(0)
                    continue
                eti_searcher = min(2*searcher.train_time,
                                   searcher.expected_time_improvement_search())
                gap = searcher.best_loss - self.best_loss
                if gap > 0 and not self.ensemble:
                    delta_loss = searcher.old_loss - searcher.new_loss
                    delta_time = searcher.old_loss_time + \
                        searcher.new_loss_time - searcher.old_train_time
                    speed = delta_loss/float(delta_time)
                    try:
                        expected_time = max(gap/speed, searcher.train_time)
                    except ZeroDivisionError:
                        warnings.warn("ZeroDivisionError: need to debug ",
                                      "speed: {0}, "
                                      "old_loss: {1}, "
                                      "new_loss: {2}"
                                      .format(speed,
                                              searcher.old_loss,
                                              searcher.new_loss))
                        expected_time = 0.0
                    expected_time = 2 * max(expected_time, eti_searcher)
                else:
                    expected_time = eti_searcher
                if expected_time == 0:
                    expected_time = 1e-10
                inv.append(1/expected_time)
            else:
                expected_time = self.eti[i]
                inv.append(0)
            if expected_time < min_expected_time:
                min_expected_time = expected_time
                selected = estimator
        if len(self.searchers) < len(estimator_list) or not selected:
            if selected not in self.searchers:
                # print('select',selected,'eti',min_expected_time)
                return selected
        s = sum(inv)
        p = np.random.random()
        q = 0
        for i in range(len(inv)):
            if inv[i]:
                q += inv[i]/s
                if p < q:
                    return estimator_list[i]
