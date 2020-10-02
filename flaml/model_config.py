'''!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under MICROSOFT RESEARCH LICENSE TERMS. 
'''


class ConfigSearchInfo:
    '''The class of the search space of a hyperparameters:

    Attributes:
        name: A string of the name of the hyperparameter
        type: data type of the hyperparameter
        lower: A number of the lower bound of the value
        upper: A number of the upper bound of the value
        init: A number of the initial value. For hyperparameters related to
            complexity, the init value needs to correspond to the lowest
            complexity
        change_tpe: A string of the change type, 'linear' or 'log'
        min_change: A number of the minimal change required. Could be inf if
            no such requirement
    '''

    def __init__(self, name, type, lower, upper, init, change_type = 'log',
     complexity_related = True, min_change = None):
        self.name = name  
        self.type = type  
        self.lower = lower 
        self.upper = upper 
        self.init = init  
        self.change_type = change_type
        self.complexity_related = complexity_related
        # default setting of min_change: if type is int, min_change 
        # should be 1, otherwise +inf
        if min_change is None:
            if self.type == int:
                self.min_change = 1.0 #minimum change required, 
            else:
                self.min_change = float('+inf')
        else:
            self.min_change = min_change

def GET_ESTIMATOR_CS(estimator, data_size, objective_name = "regression"):
    CS = {}
    n_estimators_upper = min(32768,int(data_size))
    max_leaves_upper = min(32768,int(data_size))
    # exp_max_depth_upper = min(32768,data_size)
    if 'xgboost' in estimator:
        CS['n_estimators'] = ConfigSearchInfo(name = 'n_estimators',
         type = int, lower = 4, init = 4, upper = n_estimators_upper, 
         change_type = 'log', complexity_related = True)
        CS['max_leaves'] = ConfigSearchInfo(name = 'max_leaves', type =int,
         lower = 4, init = 4, upper = max_leaves_upper , change_type = 'log', 
         complexity_related = True)
        CS['min_child_weight'] = ConfigSearchInfo(name = 'min_child_weight',
         type = float, lower = 0.001, init = 20.0, upper = 20.0, 
         change_type = 'log', complexity_related = True)

        CS['learning_rate'] = ConfigSearchInfo(name = 'learning_rate',
         type = float, lower = 0.01, init = 0.1, upper = 1.0, 
         change_type = 'log', complexity_related = True)
        CS['subsample'] = ConfigSearchInfo(name = 'subsample', type = float,
         lower = 0.6, init = 1.0, upper = 1.0, change_type = 'linear', 
         complexity_related = True)
        CS['reg_alpha'] = ConfigSearchInfo(name = 'reg_alpha', type = float,
         lower = 1e-10, init = 1e-10, upper = 1.0, change_type = 'log',
         complexity_related = True)
        CS['reg_lambda'] = ConfigSearchInfo(name = 'reg_lambda', type = float,
         lower = 1e-10, init = 1.0, upper = 1.0, change_type = 'log', 
         complexity_related = True)
        CS['colsample_bylevel'] = ConfigSearchInfo(name = 'colsample_bylevel',
         type = float, lower = 0.6, init = 1.0, upper = 1.0, 
         change_type = 'linear', complexity_related = True)
        CS['colsample_bytree'] = ConfigSearchInfo(name = 'colsample_bytree',
         type = float, lower = 0.7, init = 1.0, upper = 1.0, 
         change_type = 'linear', complexity_related = True)
    elif estimator in ('rf', 'extra_tree'):
        n_estimators_upper = min(2048, n_estimators_upper)
        # max_leaves_upper = min(2048, max_leaves_upper)
        CS['n_estimators'] = ConfigSearchInfo(name = 'n_estimators',
         type = int, lower = 4, init = 4, upper = n_estimators_upper, 
         change_type = 'log', complexity_related = True)
        if objective_name != 'regression':
            CS['criterion'] = ConfigSearchInfo(name = 'criterion',
            type = int, lower = 1, init = 1, upper = 2, 
            change_type = 'log', complexity_related = True)
        
        # CS['max_leaves'] = ConfigSearchInfo(name = 'max_leaves', type =int,
        #  lower = 4, init = 4, upper = max_leaves_upper, change_type = 'log',
        #  complexity_related = True)
        
        CS['max_features'] = ConfigSearchInfo(name = 'max_features', type = float,
         lower = 0.1, init = 1.0, upper = 1.0, change_type = 'log', 
         complexity_related = True)
        # CS['min_samples_split'] = ConfigSearchInfo(name = 'min_samples_split',
        #  type = int, lower = 2, init = 2, upper = 20, change_type = 'log', 
        #  complexity_related = True)
        # CS['min_samples_leaf'] = ConfigSearchInfo(name = 'min_samples_leaf',
        #  type = int, lower = 1, init = 1, upper = 20, change_type = 'log', 
        #  complexity_related = True)
    elif 'lgbm' in estimator:
        CS['n_estimators'] = ConfigSearchInfo(name = 'n_estimators', type = int,
         lower = 4, init = 4, upper = n_estimators_upper, change_type = 'log', 
         complexity_related = True)
        CS['max_leaves'] = ConfigSearchInfo(name = 'max_leaves', type = int,
         lower = 4, init = 4, upper = max_leaves_upper, change_type = 'log', 
         complexity_related = True)
        CS['min_child_weight'] = ConfigSearchInfo(name = 'min_child_weight',
         type = float, lower = 0.001, init = 20, upper = 20.0, 
         change_type = 'log', complexity_related = True)

        CS['learning_rate'] = ConfigSearchInfo(name = 'learning_rate',
         type = float, lower = 0.01, init = 0.1, upper = 1.0, 
         change_type = 'log', complexity_related = True)
        CS['subsample'] = ConfigSearchInfo(name = 'subsample', type = float,
         lower = 0.6, init = 1.0, upper = 1.0, change_type = 'log',
         complexity_related = True)
        CS['log_max_bin'] = ConfigSearchInfo(name = 'log_max_bin', type = int,
         lower = 3, init = 8, upper = 10, change_type = 'log',
         complexity_related = True)
        CS['reg_alpha'] = ConfigSearchInfo(name = 'reg_alpha', type = float,
         lower = 1e-10, init = 1e-10, upper = 1.0, change_type = 'log',
         complexity_related = True)
        CS['reg_lambda'] = ConfigSearchInfo(name = 'reg_lambda', type = float,
         lower = 1e-10, init = 1.0, upper = 1.0, change_type = 'log', 
         complexity_related = True)
        CS['colsample_bytree'] = ConfigSearchInfo(name = 'colsample_bytree',
         type = float, lower = 0.7, init = 1.0, upper = 1.0, 
         change_type = 'log', complexity_related = True)
    elif 'lr' in estimator:
        CS['C'] = ConfigSearchInfo(name = 'C', type =float, lower = 0.03125,
          init = 1.0, upper = 32768.0, change_type = 'log', 
          complexity_related = True)
    elif 'catboost' in estimator:
        # CS['n_estimators'] = ConfigSearchInfo(name = 'n_estimators', type = int,
        #  lower = 4, init = 64,  upper = n_estimators_upper, change_type = 'log', 
        #  complexity_related = True)
        early_stopping_rounds = max(min(round(1500000/data_size),150), 10)
        CS['rounds'] = ConfigSearchInfo(name = 'rounds', type = int,
         lower = 10, init = 10, 
         upper = early_stopping_rounds, change_type = 'log', 
         complexity_related = True)
        # CS['exp_max_depth'] = ConfigSearchInfo(name = 'exp_max_depth', type = int,
        #  lower = 32, init = 64,  upper = 256, change_type = 'log', 
        #  complexity_related = True)

        CS['learning_rate'] = ConfigSearchInfo(name = 'learning_rate',
         type = float, lower = 0.005,  init = 0.1,  upper = .2, 
         change_type = 'log', complexity_related = True)
        # CS['l2_leaf_reg'] = ConfigSearchInfo(name = 'l2_leaf_reg',
        #  type = float, lower = 1,  init = 3, upper = 5, 
        #  change_type = 'log', complexity_related = True)
    elif 'nn' == estimator:
        CS['learning_rate'] = ConfigSearchInfo(name = 'learning_rate',
         type = float, lower = 1e-4, init = 3e-4, upper = 3e-2, 
         change_type = 'log', complexity_related = True)
        CS['weight_decay'] = ConfigSearchInfo(name = 'weight_decay',
         type = float, lower = 1e-12, init = 1e-6, upper = .1, 
         change_type = 'log', complexity_related = True)
        CS['dropout_prob'] = ConfigSearchInfo(name = 'dropout_prob',
         type = float, lower = 1.0, init = 1.1, upper = 1.5, 
         change_type = 'log', complexity_related = True)
    elif 'kneighbor' in estimator:
        n_neighbors_upper = min(512,int(data_size/2))
        CS['n_neighbors'] = ConfigSearchInfo(name = 'n_neighbors', type = int,
         lower = 1, init = 5, upper = n_neighbors_upper, change_type = 'log', 
         complexity_related = True)        
    else:
        raise NotImplementedError

    return CS


def GET_ESTIMATOR_SIZE(config, estimator):
    if estimator in ['xgboost', 'lgbm', 'rf', 'extra_tree']:
        try:
            max_leaves = int(round(config['max_leaves']))
            n_estimators = int(round(config['n_estimators']))
            model_size = float((max_leaves*3 + (max_leaves-1)*4 + 1)*
                n_estimators*8) 
        except:
            model_size = 0
        return model_size
    elif 'catboost' in estimator:
        # if config is None: raise Exception("config is none")
        n_estimators = int(round(config.get('n_estimators',8192)))
        max_leaves = int(round(config.get('exp_max_depth',64)))
        model_size = float((max_leaves*3 + (max_leaves-1)*4 + 1)*
            n_estimators*8) 
        return model_size
    else:
        model_size = 1.0
        # raise NotImplementedError
    return model_size
