'''!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under MICROSOFT RESEARCH LICENSE TERMS. 
'''

import numpy as np
from random import gauss
from .config import N_SPLITS, SPLIT_RATIO
from scipy.sparse import vstack, issparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize
from sklearn.preprocessing._encoders import _BaseEncoder
import collections 
from pandas import api

import logging
logger = logging.getLogger(__name__)


class save_info_helper:


    def __init__(self, automl_name, save_file_name):
        self.automl_name = automl_name
        self.save_file_name = save_file_name
        self.file_save = open(save_file_name, 'w')
        self.current_best_loss_info = None
        self.current_best_loss = float('+inf')
        self.current_sample_size = None
        self.file_save.write('iter\tlogged_metric\ttrain_time\tobjective2minimize\t'
        'time_from_start\tconfig\tprevious_best_val_loss\tprevious_best_config'
        '\tmove\tsample_size\tbase\tconfig_sig\n')

    def add_res_more(self, it_counter, train_loss, train_time, all_time,
     i_config, val_loss, best_val_loss, best_config, current_config_arr_iter, 
     move, sample_size, base='None', config_sig='None', write_to_file=True):
        if val_loss != None:
            line_info = str(it_counter) + '\t' + str(train_loss) + '\t' + str(
                train_time) + '\t' + str(val_loss) + '\t' + str(
                    all_time) + '\t' + str(i_config) + '\t' + str(
                best_val_loss) + '\t' + str(best_config) + '\t' + str(
                    move) + '\t' + str(sample_size) + '\t' + str(
                base) + '\t' + str(config_sig)
            if val_loss < self.current_best_loss or \
                val_loss == self.current_best_loss and \
                    sample_size > self.current_sample_size:
                self.current_best_loss = val_loss
                self.current_sample_size = sample_size
                self.current_best_loss_info = line_info
            if write_to_file:
                self.file_save.write(line_info)
                self.file_save.write('\n')
                self.file_save.flush()
        else:
            print('TEST LOSS NONE ERROR!!!')

    def update_best(self):
        if self.current_best_loss_info:
            self.file_save.write('best:' + '\t' + self.current_best_loss_info)
            self.file_save.write('\n')
        self.file_save.flush()
        self.file_save.close()
        return self.current_best_loss_info

    def __del__(self):
        self.file_save.close()


def get_classification_objective(num_labels: int) -> str:
    if num_labels == 2:
        objective_name = 'binary:logistic'
    else:
        objective_name = 'multi:softmax'
    return objective_name


def rand_vector_unit_sphere(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


def rand_vector_gaussian(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    return vec


def softmax(df):
    if len(df.shape) == 1:
        df[df > 20] = 20
        df[df < -20] = -20
        ppositive = 1 / (1 + np.exp(-df))
        ppositive[ppositive > 0.999999] = 1
        ppositive[ppositive < 0.0000001] = 0
        return np.transpose(np.array((1 - ppositive, ppositive)))
    else:
        tmp = df - np.max(df, axis=1).reshape((-1, 1))
        tmp = np.exp(tmp)
        return tmp / np.sum(tmp, axis=1).reshape((-1, 1))


def get_config_values(config_dic, config_type_dic):
    value_list = []
    for k in config_dic.keys():
        org_v = config_dic[k]
        if config_type_dic[k] == int:
            v = int(round(org_v))
            value_list.append(v)
        else:
            value_list.append(org_v)
    return value_list


def load_openml_dataset(dataset_id, data_dir=None, random_state=0):
    '''Load dataset from open ML. 

    If the file is not cached locally, download it from open ML.

    Args:
        dataset_id: An integer of the dataset id in openml
        data_dir: A string of the path to store and load the data
        random_state: An integer of the random seed for splitting data

    Returns:
        X_train: A 2d numpy array of training data
        X_test:  A 2d numpy array of test data
        y_train: A 1d numpy arrya of labels for training data
        y_test:  A 1d numpy arrya of labels for test data        
    '''
    import os
    import openml
    import pickle
    from sklearn.model_selection import train_test_split

    filename = 'openml_ds' + str(dataset_id) + '.pkl'
    filepath = os.path.join(data_dir, filename)
    if os.path.isfile(filepath):
        print('load dataset from', filepath)
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
    else:
        print('download dataset from openml')
        dataset = openml.datasets.get_dataset(dataset_id)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    print('Dataset name:', dataset.name)
    X, y, * \
        __ = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format='array')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state)
    print(
        'X_train.shape: {}, y_train.shape: {};\nX_test.shape: {}, y_test.shape: {}'.format(
            X_train.shape, y_train.shape, X_test.shape, y_test.shape,
        )
    )
    return X_train, X_test, y_train, y_test


def load_openml_task(task_id, data_dir):
    '''Load task from open ML. 

    Use the first fold of the task. 
    If the file is not cached locally, download it from open ML.

    Args:
        task_id: An integer of the task id in openml
        data_dir: A string of the path to store and load the data

    Returns:
        X_train: A 2d numpy array of training data
        X_test:  A 2d numpy array of test data
        y_train: A 1d numpy arrya of labels for training data
        y_test:  A 1d numpy arrya of labels for test data        
    '''
    import os
    import openml
    import pickle
    task = openml.tasks.get_task(task_id)
    filename = 'openml_task' + str(task_id) + '.pkl'
    filepath = os.path.join(data_dir, filename)
    if os.path.isfile(filepath):
        print('load dataset from', filepath)
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
    else:
        print('download dataset from openml')
        dataset = task.get_dataset()
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    X, y, _, _ = dataset.get_data(task.target_name, dataset_format='array')
    train_indices, test_indices = task.get_train_test_split_indices(
        repeat=0,
        fold=0,
        sample=0,
    )
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    print(
        'X_train.shape: {}, y_train.shape: {},\nX_test.shape: {}, y_test.shape: {}'.format(
            X_train.shape, y_train.shape, X_test.shape, y_test.shape,
        )
    )
    return X_train, X_test, y_train, y_test


def get_output_from_log(filename, time_budget):
    '''Get output from log file

    Args:
        filename: A string of the log file name
        time_budget: A float of the time budget in seconds

    Returns:
        training_time_list: A list of the finished time of each logged iter
        best_error_list: 
            A list of the best validation error after each logged iter
        error_list: A list of the validation error of each logged iter
        config_list: 
            A list of the estimator, sample size and config of each logged iter
        logged_metric_list: A list of the logged metric of each logged iter 
    '''
    import ast

    best_config = None
    best_learner = None
    best_val_loss = float('+inf')
    training_duration = 0.0

    training_time_list = []
    config_list = []
    best_error_list = []
    error_list = []
    logged_metric_list = []
    best_config_list = []
    with open(filename) as file_:
        for line in file_:
            data = line.split('\t')
            if data[0][0] in ('b','i'):
                continue
            time_used = float(data[4])
            training_duration = time_used
            val_loss = float(data[3])
            config = ast.literal_eval(data[5])
            learner = data[8].split('_')[0]
            sample_size = data[9]
            train_loss = data[1]

            if time_used < time_budget:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_config = config
                    best_learner = learner
                    best_config_list.append(best_config)
                training_time_list.append(training_duration)
                best_error_list.append(best_val_loss)
                logged_metric_list.append(train_loss)
                error_list.append(val_loss)
                config_list.append({"Current Learner": learner,
                                    "Current Sample": sample_size,
                                    "Current Hyper-parameters": data[5],
                                    "Best Learner": best_learner,
                                    "Best Hyper-parameters": best_config})

    return (training_time_list, best_error_list, error_list, config_list,
     logged_metric_list)


def concat(X1, X2):
    '''concatenate two matrices vertically
    '''
    if isinstance(X1, pd.DataFrame) or isinstance(X1, pd.Series):
        if isinstance(X1, pd.DataFrame):
            cat_columns = X1.select_dtypes(
                include='category').columns
        df = pd.concat([X1, X2], sort=False)
        df.reset_index(drop=True, inplace=True)
        if isinstance(X1, pd.DataFrame) and len(cat_columns):
            df[cat_columns] = df[cat_columns].astype('category')
        return df
    if issparse(X1):
        return vstack((X1, X2))
    else:
        return np.concatenate([X1, X2])


class DataTransformer:
    '''transform X, y
    '''


    def fit_transform(self, X, y, objective):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            n = X.shape[0]
            cat_columns, num_columns = [], []
            for column in X.columns:
                if X[column].dtype.name in ('object', 'category'):
                    if X[column].nunique()==1 or X[column].nunique(
                        dropna=True)==n-X[column].isnull().sum():
                        X.drop(columns=column, inplace=True)
                    elif X[column].dtype.name == 'category':
                        current_categories = X[column].cat.categories
                        if '__NAN__' not in current_categories:
                            X[column] = X[column].cat.add_categories(
                                '__NAN__').fillna('__NAN__')
                        cat_columns.append(column)
                    else:                        
                        X[column].fillna('__NAN__', inplace=True)
                        cat_columns.append(column)                
                else:
                    # print(X[column].dtype.name)
                    if X[column].nunique(dropna=True)<2:
                        X.drop(columns=column, inplace=True)
                    else:
                        X[column].fillna(np.nan, inplace=True)
                        num_columns.append(column)
            X = X[cat_columns+num_columns]
            if cat_columns:
                X[cat_columns] = X[cat_columns].astype('category')
            if num_columns:
                from sklearn.impute import SimpleImputer
                from sklearn.compose import ColumnTransformer
                # X[num_columns] = X[num_columns].astype('float')
                self.transformer = ColumnTransformer([(
                    'continuous', 
                    SimpleImputer(missing_values=np.nan, strategy='median'), 
                    num_columns)])
                X[num_columns] = self.transformer.fit_transform(X)
            self.cat_columns, self.num_columns = cat_columns, num_columns
            
        if objective == 'regression':
            self.label_transformer = None
        else:
            from sklearn.preprocessing import LabelEncoder
            self.label_transformer = LabelEncoder()
            y = self.label_transformer.fit_transform(y)
        return X, y

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            cat_columns, num_columns = self.cat_columns, self.num_columns
            X = X[cat_columns+num_columns].copy()
            for column in cat_columns:
                # print(column, X[column].dtype.name)
                if X[column].dtype.name == 'object':
                    X[column].fillna('__NAN__', inplace=True)
                elif X[column].dtype.name == 'category':
                    current_categories = X[column].cat.categories
                    if '__NAN__' not in current_categories:
                        X[column] = X[column].cat.add_categories(
                            '__NAN__').fillna('__NAN__')
            if cat_columns:
                X[cat_columns] = X[cat_columns].astype('category')
            if num_columns:
                X[num_columns].fillna(np.nan, inplace=True)
                X[num_columns] = self.transformer.transform(X)
        return X

    