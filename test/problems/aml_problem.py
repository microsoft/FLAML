import time
import numpy as np
import json
import os
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
try:
    import ray
except:
    print("pip install flaml[blendsearch,ray]")
   
try: 
    from flaml import tune
except:
    from ray import tune

import logging
logger = logging.getLogger(__name__)

N_SPLITS = 5
RANDOM_SEED = 1
SPLIT_RATIO = 0.1 #0.33
HISTORY_SIZE = 10000000
MEM_THRES = 4*(1024**3)
SMALL_LARGE_THRES =  10000000
MIN_SAMPLE_TRAIN = 10000
MIN_SAMPLE_VAL = 10000
CV_HOLDOUT_THRESHOLD = 100000

def add_res(log_file_name, *params):
    # params =[time_used, eval_count, best_obj, best_config, choice, obj, eval_time, i_config]
    file_save = open(log_file_name, 'a+')
    line_info = '\t'.join(str(x) for x in params)
    file_save.write(line_info)
    file_save.write('\n')    

class Problem:


    def __init__(self, **kwargs):
        self._setup_search()

    def _setup_search(self):
        self._search_space = {}
        self._init_config = {}
        self._prune_attribute = None 
        self._resource_default, self._resource_min, self._resource_max = None, None, None 
        self._cat_hp_cost = {}

    @property
    def init_config(self):
        return self._init_config

    @property
    def search_space(self):
        return self._search_space

    @property
    def cat_hp_cost(self):
        return self._cat_hp_cost

    @property
    def prune_attribute(self):
        return self._prune_attribute 

    @property
    def prune_attribute_default_min_max(self):
        return self._resource_default, self._resource_min, self._resource_max  

    def trainable_func(self, config, **kwargs):
        obj = 0
        return obj


class Toy(Problem):

    def __init__(self, **kwargs):
        self.name = 'toy'
        self._setup_search()

    def _setup_search(self):
        super()._setup_search()
        self._search_space = {}
        self._search_space['x'] = tune.qloguniform(1,1000000,1) 
        self._search_space['y'] = tune.qloguniform(1,1000000,1) 

    def trainable_func(self, config, **kwargs):
        _, metric2minimize, time2eval = self.compute_with_config(config)
        return metric2minimize

    def compute_with_config(self, config: dict, budget_left = None, state = None):
        curent_time = time.time()
        state = None
        # should be a function of config
        metric2minimize = (round(config['x'])-95000)**2 
        time2eval = time.time() - curent_time
        return state, metric2minimize, time2eval

    
class AutoML(Problem):
    from .openml_info import oml_tasks
    task = oml_tasks
    metric = {
        'binary': 'roc_auc',
        'multi': 'log_loss',
        'regression': 'r2',
    }
    data_dir = 'test/automl/'

    class BaseEstimator:
        '''The abstract class for all learners
    
        '''


        MEMORY_BUDGET = 80*1024**3
        
        def __init__(self, objective_name = 'binary:logistic', n_jobs = 1, 
            memory_budget = MEMORY_BUDGET, **params):
            '''Constructor
            
            Args:
                objective_name: A string of the objective name, one of
                    'binary:logistic', 'multi:softmax', 'regression'
                n_jobs: An integer of the number of parallel threads
                params: A dictionary of the hyperparameter names and values
            '''
            self.params = params
            self.estimator = DummyClassifier
            self.objective_name = objective_name
            self.n_jobs = n_jobs
            self.memory_budget = memory_budget
            self.model = None
            self._dummy_model = None

        def _size(self):
            '''the memory consumption of the model
            '''
            try:
                max_leaves = int(round(self.params['max_leaves']))
                n_estimators = int(round(self.params['n_estimators']))
            except:
                return 0        
            model_size = float((max_leaves*3 + (max_leaves-1)*4 + 1)*
                n_estimators*8) 
            return model_size
            
        @property
        def classes_(self):
            return self.model.classes_

        def preprocess(self, X):
            # print('base preprocess')
            return X

        def cleanup(self): pass            

        def __del__(self):
            self.cleanup()
        
        def dummy_model(self, X_train, y_train):
            if self._dummy_model is None:
                if self.objective_name == 'regression':
                    self._dummy_model = DummyRegressor()
                else:
                    self._dummy_model = DummyClassifier()
                self._dummy_model.fit(X_train, y_train)
            return self._dummy_model
            

        def fit(self, X_train, y_train, budget = None, train_full = None):
            '''Train the model from given training data
            
            Args:
                X_train: A numpy array of training data in shape n*m
                y_train: A numpy array of labels in shape n*1

            Returns:
                model: An object of the trained model, with method predict(), 
                    and predict_proba() if it supports classification
                traing_time: A float of the training time in seconds
            '''
            curent_time = time.time()
            X_train = self.preprocess(X_train)
            if self._size() > self.memory_budget: 
                return None, time.time() - curent_time
            model = self.estimator(**self.params)
            model.fit(X_train, y_train)
            train_time =  time.time() - curent_time
            self.model=model
            return (model, train_time)

        def predict(self, X_test):
            '''Predict label from features
            
            Args:
                model: An object of trained model with method predict()
                X_test: A numpy array of featurized instances, shape n*m

            Returns:
                A numpy array of shape n*1. 
                Each element is the label for a instance
            '''        
            X_test = self.preprocess(X_test)
            return self.model.predict(X_test)

        def predict_proba(self, X_test):
            '''Predict the probability of each class from features

            Only works for classification problems

            Args:
                model: An object of trained model with method predict_proba()
                X_test: A numpy array of featurized instances, shape n*m

            Returns:
                A numpy array of shape n*c. c is the # classes
                Each element at (i,j) is the probability for instance i to be in
                    class j
            '''
            if 'regression' in self.objective_name:
                print('Regression tasks do not support predict_prob')
                raise ValueError
            else:
                X_test = self.preprocess(X_test)
                return self.model.predict_proba(X_test)

    
    class SKLearnEstimator(BaseEstimator):


        def preprocess(self, X):
            if isinstance(X, pd.DataFrame):
                X = X.copy()
                cat_columns = X.select_dtypes(include=['category']).columns
                X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)            
                X = X.fillna(0)
            return X


    class LGBM(BaseEstimator):


        def __init__(self, objective_name = 'binary', n_jobs=1, 
            n_estimators = 2, max_leaves = 2, min_child_weight = 1e-3, 
            learning_rate = 0.1, subsample = 1.0,  reg_lambda = 1.0, 
            reg_alpha = 0.0,  colsample_bylevel = 1.0, colsample_bytree = 1.0,
            log_max_bin=8, **params):
            super().__init__(objective_name, n_jobs)
            #Default: ‘regression’ for LGBMRegressor, ‘binary’ or ‘multiclass’ for LGBMClassifier
            if 'regression' in objective_name:
                final_objective_name = 'regression'
            elif 'binary' in objective_name:
                final_objective_name = 'binary'
            elif 'multi' in objective_name:
                final_objective_name = 'multiclass'
            # print(n_estimators)
            self.params = {
            "n_estimators": int(round(n_estimators)),
            "num_leaves": int(round(max_leaves)),
            'objective': final_objective_name,
            'n_jobs': n_jobs,
            'learning_rate': float(learning_rate),
            'reg_alpha': float(reg_alpha),
            'reg_lambda': float(reg_lambda),
            'min_child_weight': float(min_child_weight),
            # 'colsample_bylevel': float(colsample_bylevel),
            'colsample_bytree':float(colsample_bytree),
            'subsample': float(subsample),
            'max_bin': 1<<int(round(log_max_bin))-1,
            'verbose': -1,
            }
            import lightgbm
            if 'regression' in objective_name:
                self.estimator = lightgbm.LGBMRegressor
            else:
                self.estimator = lightgbm.LGBMClassifier    
            self.time_per_iter = None
            self.train_size = 0

        def _size(self):
            try:
                max_leaves = self.params['num_leaves']
                n_estimators = int(round(self.params['n_estimators']))
            except:
                return 0        
            model_size = float((max_leaves*3 + (max_leaves-1)*4 + 1)*n_estimators*8)
            return model_size

        def preprocess(self, X):
            import scipy.sparse
            if not isinstance(X, pd.DataFrame) and scipy.sparse.issparse(
                X) and np.issubdtype(X.dtype, np.integer):
                X = X.astype(float)
            return X

        def fit(self, X_train, y_train, budget=None, train_full=False):
            # print('restrict time', budget, self.params["n_estimators"])
            start_time = time.time()
            n_iter = self.params["n_estimators"]
            processed = False
            if (not self.time_per_iter or abs(
                self.train_size - X_train.shape[0])>N_SPLITS) and budget:
                X_train, processed = self.preprocess(X_train), True
                self.params["n_estimators"] = 1
                self.model, self.t1 = super().fit(X_train, y_train)
                # print('t1', self.t1)
                if self.t1 >= budget: 
                    self.params["n_estimators"] = n_iter
                    return self.model, self.t1
                self.params["n_estimators"] = 4
                self.model, self.t2 = super().fit(X_train, y_train)
                # print('fit model', self.model, n_iter)
                self.time_per_iter = (self.t2-self.t1)/(
                    self.params["n_estimators"]-1
                    ) if self.t2 > self.t1 else self.t1
                self.train_size = X_train.shape[0]
                if self.t1 + self.t2 >= budget or \
                    n_iter == self.params["n_estimators"]: 
                    self.params["n_estimators"] = n_iter
                    return self.model, time.time() - start_time      
            self.params["n_estimators"] = n_iter
            if budget and budget != np.inf:  #TODO why budget !=np.inf?
                train_times = 1 #+ int(train_full)
                # self.params["n_estimators"] = min(n_iter, 
                # int((budget-time.time()+start_time-self.t1)/train_times/
                # self.time_per_iter+1))
                est_cost = (n_iter - 1)*self.time_per_iter*train_times
                # print('***est cost***:', est_cost) 
                # print('***budget left***:', (budget-time.time()+start_time-self.t1))
                # print('model', self.model)
                if est_cost > (budget-time.time()+start_time-self.t1):
                    # print('return mode', self.model)
                    return self.dummy_model(X_train, y_train), self.t1 + self.t2
            if self.params["n_estimators"] > 0:
                if not processed: X_train = self.preprocess(X_train)
                self.model, _ = super().fit(X_train, y_train)
                # print('model', self.model)
            # self.params["n_estimators"] = n_iter
            train_time = time.time() - start_time
            # print('train_time', train_time, self.params["n_estimators"])
            return self.model, train_time


    class XGB_cat(SKLearnEstimator, LGBM):


        def __init__(self, objective_name = 'binary', n_jobs=1, n_estimators = 4, 
         max_leaves = 4, subsample = 1.0, 
         min_child_weight = 1, learning_rate = 0.1, reg_lambda = 1.0, 
         reg_alpha = 0.0,  colsample_bylevel = 1.0, colsample_bytree = 1.0, 
         tree_method = 'hist', booster = 'gbtree', **params):
            super().__init__(objective_name, n_jobs)
            self.params['max_depth'] = 0
            self.params = {
            "n_estimators": int(round(n_estimators)),
            # 'max_depth': int(round(math.log(self.num_leaf))),
            'max_leaves': int(round(max_leaves)),
            # 'max_depth': 0,
            'max_depth': 1000,
            'grow_policy': 'lossguide',
            'tree_method':tree_method,
            'verbosity': 0,
            'nthread':n_jobs,
            'learning_rate': float(learning_rate),
            'subsample': float(subsample),
            'reg_alpha': float(reg_alpha),
            'reg_lambda': float(reg_lambda),
            'min_child_weight': float(min_child_weight),
            'booster':booster,
            'colsample_bylevel': float(colsample_bylevel),
            'colsample_bytree':float(colsample_bytree),
            'seed': 9999999,
            }
            import xgboost
            if 'regression' in objective_name:
                self.estimator = xgboost.XGBRegressor
            else:
                self.estimator = xgboost.XGBClassifier
     
        def _size(self):
            max_leaves = self.params['max_leaves']
            n_estimators = self.params['n_estimators']
            return float((max_leaves*3 + (max_leaves-1)*4 + 1)*n_estimators*8)
    
     
    class DeepTables(BaseEstimator):


        def __init__(self, objective_name='binary', n_jobs=1, **params):
            super().__init__(objective_name, n_jobs)
            self.params = params
            # assert 'epochs' in params
            # assert 'rounds' in params
            # assert 'net' in params
            self.home_dir = None

        def preprocess(self, X):
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=[str(x) for x in list(range(
                    X.shape[1]))])
            return X

        def fit(self, X_train, y_train, budget=None, train_full=False):
            try:
                from deeptables.models.deeptable import DeepTable, ModelConfig
                from deeptables.models.deepnets import DCN, WideDeep, DeepFM
            except ImportError:
                print("pip install tensorflow==2.2.0 deeptables[gpu]")
            dropout = self.params.get('dropout', 0)
            learning_rate = self.params.get('learning_rate', 0.001)
            batch_norm = self.params.get('batch_norm', True)
            auto_discrete = self.params.get('auto_discrete', False)
            apply_gbm_features = self.params.get('apply_gbm_features', False)
            fixed_embedding_dim = self.params.get('fixed_embedding_dim', True)
            if not fixed_embedding_dim: embeddings_output_dim = 0
            else: embeddings_output_dim = 4
            stacking_op = self.params.get('stacking_op', 'add')
            if 'binary' in self.objective_name:
                # nets = DCN
                metrics, monitor = ['AUC'], 'val_auc'
            elif 'multi' in self.objective_name:
                # nets = WideDeep  
                metrics, monitor = [
                    'categorical_crossentropy'], 'val_categorical_crossentropy'
            else:
                metrics, monitor = ['r2'], 'val_r2'
            l1, l2 = 256, 128 #128, 64
            max_width = 2096
            if 'regression' != self.objective_name:
                n_classes = len(np.unique(y_train))
                base_size = max(1, min(n_classes, 100)/50)
                l1 = min(l1*base_size, max_width)
                l2 = min(l2*base_size, max_width)
            dnn_params = {'hidden_units': ((l1, dropout, batch_norm), 
            (l2, dropout, batch_norm)), 'dnn_activation': 'relu'}
            net = self.params.get('net', 'DCN')
            if net == 'DCN':
                nets = DCN
            elif net == 'WideDeep':
                nets = WideDeep
            elif net == 'DeepFM':
                nets = DeepFM
            elif net == 'dnn_nets':
                nets = [net]
            from tensorflow.keras.optimizers import Adam
            time_stamp = time.time()
            self.home_dir = f'dt_output/{time_stamp}'
            while os.path.exists(self.home_dir):
                self.home_dir += str(np.random.randint(10000000))
            conf = ModelConfig(nets=nets, earlystopping_patience=self.params['rounds'], 
                dense_dropout=self.params["dense_dropout"], 
                auto_discrete=auto_discrete, stacking_op=stacking_op,
                apply_gbm_features=apply_gbm_features,
                fixed_embedding_dim=fixed_embedding_dim,
                embeddings_output_dim=embeddings_output_dim,
                dnn_params=dnn_params,
                optimizer=Adam(learning_rate=learning_rate, clipvalue=100),
                metrics=metrics, monitor_metric=monitor, home_dir=self.home_dir)
            self.model = DeepTable(config=conf)
            log_batchsize = self.params.get('log_batchsize', 8)
            assert 'log_batchsize' in self.params.keys()
            assert 'epochs' in self.params.keys()
            self.model.fit(self.preprocess(X_train), y_train, verbose=0,
             epochs=int(round(self.params['epochs'])), batch_size=1<<log_batchsize)

        def cleanup(self):
            if self.home_dir:
                import shutil
                shutil.rmtree(self.home_dir, ignore_errors=True)


    @staticmethod
    def get_estimator_from_name(name):
        if 'lgbm' in name:
            estimator = AutoML.LGBM
        elif name in ('xgboost', 'xgb_cat', 'xgb'):
            estimator = AutoML.XGB_cat
        elif 'dt' in name or 'deeptable' in name:
            estimator = AutoML.DeepTables
        else: estimator = None
        return estimator

    @staticmethod
    def sklearn_metric_loss_score(metric_name, y_predict, y_true, labels = None):
        '''Loss using the specified metric

        Args:
            metric_name: A string of the mtric name, one of 
                'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'log_loss', 
                'f1', 'ap'
            y_predict: A 1d or 2d numpy array of the predictions which can be
                used to calculate the metric. E.g., 2d for log_loss and 1d
                for others. 
            y_true: A 1d numpy array of the true labels
            labels: A 1d numpy array of the unique labels
        
        Returns:
            score: A float number of the loss, the lower the better
        '''
        from sklearn.metrics import mean_squared_error, r2_score, \
            roc_auc_score, accuracy_score, mean_absolute_error, log_loss

        metric_name = metric_name.lower()
        try:
            if 'r2' in metric_name:
                score = 1.0-r2_score(y_true, y_predict)
            elif metric_name == 'rmse':
                score = np.sqrt(mean_squared_error(y_true, y_predict))
            elif metric_name == 'mae':
                score = mean_absolute_error(y_true, y_predict)
            elif metric_name == 'mse':
                score = mean_squared_error(y_true, y_predict)
            elif metric_name == 'accuracy':
                score = 1.0 - accuracy_score(y_true, y_predict)
            elif 'roc_auc' in metric_name:
                score = 1.0 - roc_auc_score(y_true, y_predict)
            elif 'log_loss' in metric_name:
                score = log_loss(y_true, y_predict, labels=labels)
            # elif 'f1' in metric_name:
            #     score = 1 - f1_score(y_true, y_predict)
            # elif 'ap' in metric_name:
            #     score = 1 - average_precision_score(y_true, y_predict)
            else:
                print('Does not support the specified metric')
                score = None
        except:
            print('score exception', metric_name)
            return np.Inf
        return score

    @staticmethod
    def generate_resource_schedule(reduction_factor, lower, upper, log_max_min_ratio = 5):
        resource_schedule = []
        if log_max_min_ratio: 
            r = max(int(upper/(reduction_factor**log_max_min_ratio)), lower)
        else: r = lower
        while r <= upper:
            resource_schedule.append(r)
            r *= reduction_factor
        if not resource_schedule:
            resource_schedule.append(upper)
        else:
            resource_schedule[-1] = upper
        print('resource_schedule', resource_schedule)
        return resource_schedule
    
     
    def _setup_search(self):
        super()._setup_search()
        # self._search_sapce = {}
        # self._init_config = {} 
        # self._prune_attribute = None
        # self._resource_default, self._resource_min, self._resource_max = None, None, None

        n_estimators_upper = min(32768, self.data_size)
        max_leaves_upper = min(32768, self.data_size)
        early_stopping_rounds = max(min(round(1500000/self.data_size),150), 10)
        if AutoML.DeepTables == self.estimator:
            logger.info('setting up deeptables hpo')
            self._search_space = {
                'rounds': tune.qloguniform(10,int(early_stopping_rounds), 1),
                'net': tune.choice(['DCN', 'dnn_nets']),
                "learning_rate": tune.loguniform(1e-4, 3e-2),
                'auto_discrete': tune.choice([False, True]),
                'apply_gbm_features': tune.choice([False, True]),
                'fixed_embedding_dim': tune.choice([False, True]),
                'dropout': tune.uniform(0,0.5),
                'dense_dropout': tune.uniform(0,0.5),
                "log_batchsize": 8,     
                } 
            self._init_config =  {
            'rounds': 10,
                }
            self._prune_attribute = 'epochs'
            #TODO: _resource_default is not necessary?
            self._resource_default, self._resource_min, self._resource_max = 2**10, 2**1, 2**10 
            self._cat_hp_cost={
                "net": [2,1],
                }

        # TODO: specify the search space for other learners
        elif AutoML.LGBM == self.estimator:
            logger.info('setting up lgbm hpo')
            self._search_space = {
                'n_estimators': tune.qloguniform(4, n_estimators_upper, 1),
                'max_leaves': tune.qloguniform(4, max_leaves_upper, 1),
                "min_child_weight": tune.loguniform(1e-3, 20),
                "learning_rate": tune.loguniform(1e-2, 1),
                "reg_alpha": tune.loguniform(1e-10, 1),
                "reg_lambda": tune.loguniform(1e-10, 1),
                'log_max_bin': tune.randint(3, 10),
                'subsample': tune.uniform(0.6, 1),
                'colsample_bytree': tune.uniform(0.7, 1),
                } 
            self._init_config =  {
                'n_estimators': 4,
                'max_leaves': 4,
                'min_child_weight': 20,
                }
        elif self.estimator == AutoML.XGB_cat:  
            logger.info('setting up xgb_cat hpo')
            self._search_space = {
                'n_estimators': tune.qloguniform(4, n_estimators_upper, 1),
                'max_leaves': tune.qloguniform(4, max_leaves_upper, 1),
                "min_child_weight": tune.loguniform(1e-3, 20),
                "learning_rate": tune.loguniform(1e-2, 1),
                "reg_alpha": tune.loguniform(1e-10, 1),
                "reg_lambda": tune.loguniform(1e-10, 1),
                'subsample': tune.uniform(0.6, 1),
                'colsample_bylevel': tune.uniform(0.7, 1),
                'colsample_bytree': tune.uniform(0.7, 1),
                'booster': tune.choice(['gbtree', 'gblinear']),
                'tree_method': tune.choice(['auto', 'approx', 'hist']),
                } 
            self._init_config =  {
                'n_estimators': 4,
                'max_leaves': 4,
                'min_child_weight': 20,
                }
            self._cat_hp_cost={
                "booster": [2, 1],
                }
        else: 
            NotImplementedError
        # set the configuration (to be always the largest, assuming best at max) for hp which is prune_attribute
        if self._prune_attribute is not None:
            assert self._resource_max is not None
            self._search_space[self._prune_attribute] = self._resource_max
        
    def _get_test_loss(self, estimator = None, X_test = None, y_test = None, 
                            metric = 'r2', labels = None):
        if not estimator.model:
            loss = np.Inf
        else:
            if 'roc_auc' == metric:
                y_pred = estimator.predict_proba(X_test = X_test)
                if y_pred.ndim>1 and y_pred.shape[1]>1:
                    y_pred = y_pred[:,1]
            elif 'log_loss' == metric:
                y_pred = estimator.predict_proba(X_test = X_test)
                # print('estimator', estimator)
            elif 'r2' == metric:
                y_pred = estimator.predict(X_test = X_test)
            loss = AutoML.sklearn_metric_loss_score(metric, y_pred, y_test,
             labels)
            estimator.cleanup()
        return loss

    
    #TODO: can ray tune serise this function?
    def trainable_func(self, config, start_time, log_file_name, resource_schedule,):
        # print('config in trainable_func', config)
        
        for epo in resource_schedule:
            loss, time2eval = self.compute_with_config(config)
            # write result
            time_used = time.time() - start_time
            # NOTE: these fields are missing
            eval_count, best_obj, best_config, choice = None, None, None, None  # missing fields
            obj = loss 
            i_config = config
            if self.prune_attribute:
                i_config[self._prune_attribute] = epo
            log_param = [time_used, eval_count, best_obj, best_config, choice, obj, time2eval, i_config]
            add_res(log_file_name, log_param)
            # TODO: how to specify the name in tune.report properly
            tune.report(epochs=epo, loss=loss)

    def compute_with_config(self, config: dict, budget_left = np.inf, state = None):
        curent_time = time.time()
        objective_name = self.objective
        metric = self.metric[objective_name]
        # print('config', config)
        estimator = self.estimator(**config, objective_name = objective_name,
         n_jobs=self.n_jobs)
        if self.resampling_strategy == 'cv':
            total_val_loss, valid_folder_num = 0, 0 
            n = self.kf.get_n_splits()
            # print('self.y_all',self.X_all[0:5], self.y_all[0:5])
            if budget_left is not None: budget_per_train = budget_left / n
            else: budget_per_train = np.inf
            if objective_name=='regression' or True:
                labels = None
                X_train_split, y_train_split = self.X_all, self.y_all
            else:
                labels = np.unique(self.y_all) 
                l = len(labels)
                X_train_split, y_train_split = self.X_all[l:], self.y_all[l:]
            if isinstance(self.kf, RepeatedStratifiedKFold):
                kf = self.kf.split(X_train_split, y_train_split)
            else:
                kf = self.kf.split(X_train_split)
            rng = np.random.RandomState(2020)
            val_loss_list = []
            for train_index, val_index in kf:
                train_index = rng.permutation(train_index)
                if isinstance(X_train_split, pd.DataFrame):
                    X_train, X_val = X_train_split.iloc[
                        train_index], X_train_split.iloc[val_index]
                else:
                    X_train, X_val = X_train_split[train_index], X_train_split[
                        val_index]
                if isinstance(y_train_split, pd.Series):
                    y_train, y_val = y_train_split.iloc[
                        train_index], y_train_split.iloc[val_index]
                else:
                    y_train, y_val = y_train_split[
                        train_index], y_train_split[val_index] 
                # print( 'X_iclo', X_train.iloc[0:5])               
                if labels is not None:
                    X_train = AutoML.concat(self.X_all[:l], X_train)
                    y_train = np.concatenate([self.y_all[:l], y_train])
                estimator.fit(X_train, y_train, budget_per_train)
                val_loss_i = self._get_test_loss(estimator, X_val, y_val,
                 metric, self.labels)
                # train_loss = self._get_test_loss(estimator, X_train,
                #     y_train, metric, self.labels)
                # val_loss_i = 2*val_loss_i - train_loss
                try:
                    val_loss_i = float(val_loss_i)
                    valid_folder_num += 1
                    total_val_loss += val_loss_i
                    if valid_folder_num == n:
                        val_loss_list.append(total_val_loss/valid_folder_num)
                        total_val_loss = valid_folder_num = 0
                except:
                    print ('Evaluation folder failed !!!')
                    pass
            loss = np.max(val_loss_list)
        else:
            estimator.fit(self.X_train, self.y_train, budget_left)
            loss = self._get_test_loss(estimator, X_test = self.X_val,
             y_test = self.y_val, metric = metric, labels = self.labels)
            # train_loss = self._get_test_loss(estimator, X_test = self.X_train,
            #     y_test = self.y_train, metric = metric, labels = self.labels)
            # loss = 2*loss - train_loss
            # if state: state.model = estimator.model
            # print('hold out val loss', loss)
        time2eval = time.time() - curent_time
        # return state, loss, time2eval
        return loss, time2eval
    
    def get_cat_choice_org_name(self, cat_hp, choice):
        if cat_hp in self.config_search_info_cat.keys():
            choice_index = int(choice)
            choice_name = self.config_search_info[cat_hp].choices[choice_index] 
        # TODO handle not in dic error
        return choice_name

    def _decide_eval_method(self, data_shape, time_budget):
        nrow, dim = int(data_shape[0]), int(data_shape[1])
        print(nrow, dim, nrow * dim- SMALL_LARGE_THRES)
        if nrow * dim < SMALL_LARGE_THRES and nrow < CV_HOLDOUT_THRESHOLD:
            # time allows or sampling can be used and cv is necessary
            eval_method = 'cv'
        else:
            eval_method = 'holdout'
        ## always use hold
        eval_method = 'holdout'
        # print('eval method', eval_method)
        return eval_method

    def __init__(self, dataset, estimator, fold, n_jobs, time_budget = None,
     resampling_strategy = None, **args): 
        
        self.name = f'{dataset}-{estimator}'
        self.time_budget = time_budget
        self.transform = True
        self.n_jobs = n_jobs
        self.split_type =  "stratified"
        self.split_ratio = SPLIT_RATIO
        self.n_splits = N_SPLITS
        task = self.task[dataset]
        self.task_id, self.objective = task['task_id'], task['task_type']
        self.task_type = 'regression' if self.objective == 'regression' else \
            'classification'
        self.fold = fold
        X_all, y_all, _, _ = AutoML.load_openml_task(self.task_id, fold,
         self.task_type, self.transform)
        # X_all, y_all, self.X_test, self.y_test = AutoML.load_openml_task(task_id, fold)
        if resampling_strategy is not None: 
            self.resampling_strategy = resampling_strategy
        else: self.resampling_strategy = self._decide_eval_method(
            X_all.shape, time_budget)
        print('resampling strategy')
        self.test_loss = []
        self.X_all, self.y_all, self.X_train, self.y_train, self.X_val, \
            self.y_val, self.kf, self.labels = AutoML.split_data(
                self.task_type, self.split_type,
                self.split_ratio, self.n_splits, self.resampling_strategy, 
                X_all, y_all)
        self._X_train, self._y_train = self.X_train, self.y_train
        self.estimator = AutoML.get_estimator_from_name(estimator)
        self.data_size = len(self.y_train) if (self.y_train is not None) else int(
            len(self.y_all) * (self.n_splits-1) / self.n_splits)
        # self._configure_search_setting
        # super().__init__(**args)
        print('estimator', self.estimator)
        self._setup_search()

        print('setup search space', self._search_space)

    def get_test_data(self):
        _, _, X_test, y_test = AutoML.load_openml_task(
            self.task_id, self.fold, self.task_type, self.transform)
        return self.X_all, self.y_all, X_test, y_test
        # if self.resampling_strategy == 'cv':
        #     return self.X_all, self.y_all, X_test, y_test
        # return self._X_train, self._y_train, X_test, y_test
        
    @staticmethod
    def load_openml_task(task_id, fold, task_type, transform):
        import os, openml, pickle
        customized_load = False
        if customized_load:
            import arff
            oml_task = openml.tasks.get_task(task_id)
            oml_dataset = oml_task.get_dataset()
            with open(oml_dataset.data_file) as f:
                ds = arff.load(f)
            train_ind, test_ind = oml_task.get_train_test_split_indices(fold)
            split_data_train = np.asarray(ds['data'], dtype=object)[train_ind, :]
            split_data_test = np.asarray(ds['data'], dtype=object)[test_ind, :]
            predictors = [f for f in oml_dataset.features.values()
             if f.name!=oml_dataset.default_target_attribute]
            target = [f for f in oml_dataset.features.values()
             if f.name==oml_dataset.default_target_attribute]
            predictors_ind, target_ind = [p.index for p in predictors], [
                p.index for p in target]
            X_all, y_all = split_data_train[:, predictors_ind], split_data_train[:,
             target_ind]
            X_test, y_test = split_data_test[:, predictors_ind],  split_data_test[:,
             target_ind]
            print('X_all,', X_all.shape, X_all[0:5])
            from sklearn import preprocessing
            import sklearn
            le_label = preprocessing.LabelEncoder()
            le = preprocessing.OrdinalEncoder()
            # le = preprocessing.OneHotEncoder()
            # le =sklearn.pipeline.Pipeline(sklearn.preprocessing._encoders.OneHotEncoder)
            le.fit(X_all)
            X_all = le.transform(X_all)
            X_test = le.transform(X_test)
            
            le_label.fit(y_all)
            y_all = le_label.transform(y_all)
            y_test = le_label.transform(y_test)

            print('X_all_trans,', X_all.shape, X_all[0:5], y_all[0:5])
            # Encoder('label' if self.values is not None else 'no-op',
            #            target=self.is_target,
            #            encoded_type=int if self.is_target and not self.is_numerical() else float,
            #            missing_policy='mask' if self.has_missing_values else 'ignore'
            #            ).fit(self.values)
        else:
            task = openml.tasks.get_task(task_id)
            filename = 'openml_task' + str(task_id) + '.pkl'
            os.makedirs(AutoML.data_dir, exist_ok = True)
            filepath = os.path.join(AutoML.data_dir, filename)
            if os.path.isfile(filepath):
                print('load dataset from', filepath)
                with open(filepath, 'rb') as f:
                    dataset = pickle.load(f)
            else:
                print('download dataset from openml')
                dataset = task.get_dataset()
                with open(filepath, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            # X, y, cat_, _ = dataset.get_data(task.target_name, 
            #     dataset_format='array', include_ignore_attributes = True)
            X, y, cat, _ = dataset.get_data(task.target_name, 
                include_ignore_attributes = True)
            train_indices, test_indices = task.get_train_test_split_indices(
                    repeat=0,
                    fold=fold,
                    sample=0,
                )
            if isinstance(X, pd.DataFrame):
                X_all = X.iloc[train_indices]
                y_all = y.iloc[train_indices]
                X_test = X.iloc[test_indices]
                y_test = y.iloc[test_indices]
            else:
                X_all = X[train_indices]
                y_all = y[train_indices]
                X_test = X[test_indices]
                y_test = y[test_indices]
        # print('X_all,', X_all.shape, X_all[0:5], cat_)
        if transform:
            X_all, y_all, X_test, y_test = AutoML.transform_data(task_type,
             X_all, y_all, X_test, y_test, cat)
        # print('X_all,', type(X_all), X_all.shape, X_all[0:5])
        return X_all, y_all, X_test, y_test

    @staticmethod
    def transform_data(task_type, X, y, X_test=None, y_test=None, cat=[]):
        # from azureml.automl.runtime.featurization import data_transformer
        # transformer = data_transformer.DataTransformer(task=task_type)
        # from deeptables.models.preprocessor import DefaultPreprocessor
        # from deeptables.models.deeptable import ModelConfig
        # conf = ModelConfig(auto_encode_label=False)
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        cat_columns, num_columns = [], []
        n = X.shape[0]
        for i, column in enumerate(X.columns):
            if cat[i]:
                if X[column].nunique()==1 or X[column].nunique(
                    dropna=True)==n-X[column].isnull().sum():
                    X.drop(columns=column, inplace=True)
                    if X_test is not None:
                        X_test.drop(columns=column, inplace=True)
                    continue
                elif X[column].dtype.name == 'object':
                    X.loc[:,column].fillna('__NAN__', inplace=True)
                    if X_test is not None: 
                        X_test.loc[:,column].fillna('__NAN__', inplace=True)
                elif X[column].dtype.name == 'category':
                    current_categories = X[column].cat.categories
                    if '__NAN__' not in current_categories:
                        X.loc[:,column] = X[column].cat.add_categories(
                            '__NAN__').fillna('__NAN__')
                        if X_test is not None: 
                            X_test.loc[:,column] = X_test[
                                column].cat.add_categories('__NAN__').fillna(
                                    '__NAN__')
                cat_columns.append(column)
            else:
                # print(X[column].dtype.name)
                if X[column].nunique(dropna=True)<2:
                    X.drop(columns=column, inplace=True)
                    if X_test is not None:
                        X_test.drop(columns=column, inplace=True)
                else:
                    X.loc[:,column].fillna(np.nan, inplace=True)
                    num_columns.append(column)
        if cat_columns:
            X.loc[:,cat_columns] = X[cat_columns].astype('category', copy=False)
            if X_test is not None: 
                X_test.loc[:,cat_columns] = X_test[cat_columns].astype(
                    'category', copy=False)
        if num_columns:
            X.loc[:,num_columns] = X[num_columns].astype('float')
            transformer = ColumnTransformer([('continuous', SimpleImputer(
                missing_values=np.nan, strategy='median'), num_columns)])
            X.loc[:,num_columns] = transformer.fit_transform(X)
            if X_test is not None: 
                X_test.loc[:,num_columns] = X_test[num_columns].astype('float')
                X_test.loc[:,num_columns] = transformer.transform(X_test)
        if task_type == 'regression':
            label_transformer = None
        else:
            from sklearn.preprocessing import LabelEncoder
            label_transformer = LabelEncoder()
            y = label_transformer.fit_transform(y)
            if y_test is not None: 
                y_test = label_transformer.transform(y_test)
        return X, y, X_test, y_test

    @staticmethod 
    def split_data(task_type, split_type, split_ratio, n_splits,
     resampling_strategy, X_all, y_all):
        from sklearn.model_selection import train_test_split
        from sklearn.utils import shuffle
        from scipy.sparse import issparse
        if issparse(X_all): X_all = X_all.tocsr()
        X_all, y_all = shuffle(X_all, y_all, random_state=202020)        
        df = isinstance(X_all, pd.DataFrame)
        if df:
            X_all.reset_index(drop=True, inplace=True)
            if isinstance(y_all, pd.Series):
                y_all.reset_index(drop=True, inplace=True)
        kf = X_train = y_train = X_val = y_val = None     
        labels = np.unique(y_all) 
        if resampling_strategy == 'holdout':
            if task_type != 'regression':
                label_set, first = np.unique(y_all, return_index=True)
                rest = []
                last = 0
                first.sort()
                for i in range(len(label_set)):
                    rest.extend(range(last, first[i]))
                    last = first[i] + 1
                rest.extend(range(last, len(y_all)))

                X_first = X_all.iloc[first] if df else X_all[
                    first]
                X_rest = X_all.iloc[rest] if df else X_all[rest]
                y_rest = y_all.iloc[rest] if isinstance(
                    y_all, pd.Series) else y_all[rest]
                stratify = y_rest if split_type=='stratified' else None
            else:
                stratify = None
            X_train, X_val, y_train, y_val = train_test_split(
                X_rest, y_rest, test_size=split_ratio,
                stratify=stratify, random_state=1)                                                                
            if task_type != 'regression':
                X_train = AutoML.concat(X_first, X_train)
                y_train = AutoML.concat(label_set,
                    y_train) if df else np.concatenate([label_set, y_train])
                X_val = AutoML.concat(X_first, X_val)
                y_val = AutoML.concat(label_set,
                    y_val) if df else np.concatenate([label_set, y_val])
        else:          
            if task_type != 'regression' and split_type == "stratified":
                print("Using StratifiedKFold")
                kf = RepeatedStratifiedKFold(n_splits= n_splits,
                    n_repeats=1, random_state=202020)
            else:
                print("Using KFold")
                kf = RepeatedKFold(n_splits= n_splits, n_repeats=1,
                    random_state=202020)
        return X_all, y_all, X_train, y_train, X_val, y_val, kf, labels

    @staticmethod
    def concat(X1, X2):
        '''concatenate two matrices vertically
        '''
        if isinstance(X1, pd.DataFrame) or isinstance(X1, pd.Series):
            df = pd.concat([X1, X2], sort=False)
            df.reset_index(drop=True, inplace=True)
            return df
        from scipy.sparse import vstack, issparse
        if issparse(X1):
            return vstack((X1, X2))
        else:
            return np.concatenate([X1, X2])