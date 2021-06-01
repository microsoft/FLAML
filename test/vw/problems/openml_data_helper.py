import logging
RANDOM_SEED = 20201234
import argparse
import openml
import os
import numpy as np
import string
import pandas as pd
import scipy
import math
OPENML_REGRESSION_LIST = [201, 1191, 215, 344, 537, 564, 1196, 1199, 1203, 1206, 
5648, 23515, 41506, 41539, 42729, 42496]
NS_LIST = list(string.ascii_lowercase) + list(string.ascii_uppercase)
# NS_LIST = list(string.ascii_lowercase)[:10]

OML_target_attribute_dict = {
    42236: 'pm2.5'
}

# from ..vw_benchmark.config import QW_OML_API_KEY, VW_DS_DIR
VW_DS_DIR = './test/vw/vw_benchmark/data/openml_vwdatasets/'
QW_OML_API_KEY = '8c4eebcda506ae1065902c2b224369b9'
#TODO: how to get these info from config.py

class OpenML2VWData:
    VW_DS_DIR = VW_DS_DIR
    def __init__(self, did, max_ns_num, task_type='regression'):
        self._did = did
        self._task_type = task_type
        self._is_regression = False
        self.vw_x_dic_list = []
        self.Y = []
        if 'regression' in self._task_type:
            self._is_regression = True        
        self.vw_examples = self.load_vw_dataset(did, OpenML2VWData.VW_DS_DIR, self._is_regression, max_ns_num)
        print( 'number of samples', len(self.vw_examples))
        for i, e in enumerate(self.vw_examples):
            self.Y.append(float(e.split('|')[0]))
        print( self.Y[0:5])
        logging.info('y label%s', self.Y[0:5])

    @staticmethod
    def load_vw_dataset(did, ds_dir, is_regression, max_ns_num):
        import os
        data_list = []
        if is_regression:
            fname = 'ds_{}_{}_{}.vw'.format(did, max_ns_num, 0) # the second field specifies the largest number of namespaces using.
            vw_dataset_file = os.path.join(ds_dir, fname)
            if not os.path.exists(vw_dataset_file) or os.stat(vw_dataset_file).st_size < 1000:
                get_oml_to_vw(did, max_ns_num)
            print(ds_dir, vw_dataset_file)
            if not os.path.exists(ds_dir): os.makedirs(ds_dir)
            with open(os.path.join(ds_dir, fname), 'r') as f:
                vw_content = f.read().splitlines()
                print(type(vw_content), len(vw_content))
            return vw_content


# target # of ns: 10-26.
# TODO: split features into 10-26 ns:(1) look at the prefix (10<# of unique prefix< 26); (2) sequentially.

def oml_to_vw_no_grouping(X, y, ds_dir, fname):
    print('no feature grouping')
    with open(os.path.join(ds_dir, fname), 'w') as f:
        if isinstance(X, pd.DataFrame):
            for i in range(len(X)):
                ns_line =  '{} |{}'.format(str(y[i]), '|'.join('{} {}:{:.6f}'.format(NS_LIST[j], j, val) for 
                    j, val in enumerate(X.iloc[i].to_list()) ))
                f.write(ns_line)
                f.write('\n')
        elif isinstance(X, np.ndarray):
            for i in range(len(X)):
                ns_line =  '{} |{}'.format(str(y[i]), '|'.join('{} {}:{:.6f}'.format(NS_LIST[j], j, val) for 
                        j, val in enumerate(X[i]) ))
                f.write(ns_line)
                f.write('\n')
        elif isinstance(X, scipy.sparse.csr_matrix):
            print('NotImplementedError for sparse data')
            NotImplementedError

def oml_to_vw_w_grouping(X, y, ds_dir, fname, orginal_dim, group_num, grouping_method='sequential'):
    all_indexes = [i for i in range(orginal_dim)]
    print('grouping', group_num)
    # split all_indexes into # group_num of groups
    max_size_per_group = math.ceil(orginal_dim/float(group_num))
    # Option 1: sequential grouping
    if grouping_method == 'sequential':
        
        group_indexes = [] # lists of lists
        print('indexes', group_num)
        for i in range(group_num):
            print('indexes', group_num, max_size_per_group)
            indexes = [ind for ind in range(i*max_size_per_group, min( (i+1)*max_size_per_group, orginal_dim)) ]
            print('indexes', group_num, indexes)
            if len(indexes)>0: group_indexes.append(indexes)
            print(group_indexes)
        print(group_indexes)
    else: 
        NotImplementedError
    if group_indexes:
        print('group_indexes')
        with open(os.path.join(ds_dir, fname), 'w') as f:
            if isinstance(X, pd.DataFrame):
                raise NotImplementedError
            elif isinstance(X, np.ndarray):
                for i in range(len(X)):
                    # ns_content = '{} {}:{:.6f}'.format(NS_LIST[j], j, val) for j, val in enumerate(X[i]) 
                    NS_content = []
                    for zz in range(len(group_indexes)):
                        ns_features = ' '.join('{}:{:.6f}'.format(ind, X[i][ind]) for ind in group_indexes[zz])       
                        NS_content.append(ns_features)
                    ns_line = '{} |{}'.format(str(y[i]), '|'.join('{} {}'.format(NS_LIST[j], NS_content[j]) for
                                              j in range(len(group_indexes)) ))
                    f.write(ns_line)
                    f.write('\n')
            elif isinstance(X, scipy.sparse.csr_matrix):
                print('NotImplementedError for sparse data')
                NotImplementedError


def save_vw_dataset_w_ns(X, y, did, ds_dir, max_ns_num, is_regression):
    """ convert openml dataset to vw example and save to file
    """
    print('is_regression',is_regression)
    if is_regression:
        fname = 'ds_{}_{}_{}.vw'.format(did, max_ns_num, 0)
        print('dataset size', X.shape[0], X.shape[1])
        print('saving data', did, ds_dir, fname)
        dim = X.shape[1]
        # do not do feature grouping
        from os import path
        # if not path.exists(os.path.join(ds_dir, fname)):
        if dim < max_ns_num:
            oml_to_vw_no_grouping(X, y, ds_dir, fname)
        else:
            oml_to_vw_w_grouping(X, y, ds_dir, fname, dim, group_num=max_ns_num)
        
def shuffle_data(X, y, seed):
    try:
        n = len(X)
    except:
        n = X.getnnz()
    perm = np.random.RandomState(seed=seed).permutation(n)
    X_shuf = X[perm, :]
    y_shuf = y[perm]
    return X_shuf, y_shuf

def get_oml_to_vw(did, max_ns_num, ds_dir=VW_DS_DIR):
    success = False
    print('-----getting oml dataset-------', did)
    ds = openml.datasets.get_dataset(did)
    target_attribute = ds.default_target_attribute
    if target_attribute is None and did in OML_target_attribute_dict:
        target_attribute = OML_target_attribute_dict[did]

    print('target=ds.default_target_attribute', target_attribute)
    data = ds.get_data(target=target_attribute, dataset_format='array')
    X, y = data[0], data[1] # return X: pd DataFrame, y: pd series
    import scipy
    if scipy.sparse.issparse(X):
        X = scipy.sparse.csr_matrix.toarray(X)
        print('is sparse matrix')
    if data and isinstance(X, np.ndarray):
        print('-----converting oml to vw and and saving oml dataset-------')
        save_vw_dataset_w_ns(X, y, did, ds_dir, max_ns_num, is_regression=True)
        success = True
    else:
        print('---failed to convert/save oml dataset to vw!!!----')
    try:
        X, y = data[0], data[1] # return X: pd DataFrame, y: pd series
        if data and isinstance(X, np.ndarray):
            print('-----converting oml to vw and and saving oml dataset-------')
            save_vw_dataset_w_ns(X, y, did, ds_dir, max_ns_num, is_regression = True)
            success = True
        else:
            print('---failed to convert/save oml dataset to vw!!!----')
    except:
        print('-------------failed to get oml dataset!!!', did)
    return success

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openML to vw converter')
    parser.add_argument('-dataset', type=int, default=None, help='dataset id')
    parser.add_argument('-ns_num', '--ns_num', metavar='ns_num', type = int, 
        default=10, help="max name space number")
    parser.add_argument('-min_sample_size', type=int, default=10000, help='minimum sample size')
    parser.add_argument('-max_sample_size', type=int, default=None, help='maximum sample size')
    args = parser.parse_args()
    openml.config.apikey =  QW_OML_API_KEY
    openml.config.set_cache_directory('./data/omlcache/')

    print('loaded openML')
    if not os.path.exists(VW_DS_DIR): os.makedirs(VW_DS_DIR)
    if args.dataset is not None:
        dids = [args.dataset]
    else:
        if args.min_sample_size >=10000 and args.max_sample_size is None:
            dids = OPENML_REGRESSION_LIST
    failed_datasets = []
    for did in sorted(dids):
        print('processing did', did)
        print('getting data,', did)
        success = get_oml_to_vw(did, args.ns_num)
        if not success:
            failed_datasets.append(did)
    print('-----------failed datasets', failed_datasets)
## command line:
# python openml_data_helper.py -min_sample_size 10000
# failed datasets [1414, 5572, 40753, 41463, 42080, 42092, 42125, 42130, 42131, 42160, 42183, 42207, 
# 42208, 42362, 42367, 42464, 42559, 42635, 42672, 42673, 42677, 42688, 42720, 42721, 42726, 42728, 42729, 42731]