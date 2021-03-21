import matplotlib.pyplot as plt
import numpy as np
import os
import json

def get_lc_from_log(log_file_name):
    x = []
    y = [] #minimizing 
    if os.path.isfile(log_file_name):
        best_obj = np.inf 
        total_time = 0
        with open(log_file_name) as f:
            for line in f:
                record = json.loads(line)
                total_time = float(record['total_search_time'])
                obj = float(record['obj'])
                if obj < best_obj: 
                    best_obj = obj
                    x.append(total_time)
                    y.append(best_obj)
            x.append(total_time)
            y.append(best_obj)
    else:
        print('log_file_name', log_file_name)
        print('File does not exist')
    assert len(x) == len(y) 
    return x, y
    
# discretize curve_dic
def discretize_learning_curves(self, time_budget, curve_dic):
    # curve_dic: key: method name, value: list of multiple run
    # each run is a list of tuples run_0 = [ (time, loss) ]

    self.discretized_curve = {}
    for method, curve_folds in curve_dic.items():
        discretized_curve_folds = []
        for curve in curve_folds: # curve in each fold
            # discretize the time domain into intervals of 1s
            discretized_curve = []
            pos = 0
            # discretized_obj = np.iinfo(int).max
            for discretized_t in range(int(time_budget) + 1): # 3600 or 14400
                discretized_obj = np.iinfo(int).max
                while pos < len(curve) and curve[pos][1] <= discretized_t:
                    obj, t, i = curve[pos]
                    discretized_obj = obj
                    pos += 1
                if discretized_obj != np.iinfo(int).max:
                    discretized_curve.append((discretized_obj, discretized_t, -1))
            if len(discretized_curve) == 0: discretized_curve.append((discretized_obj, discretized_t, -1))
            discretized_curve_folds.append(discretized_curve)   
        self.discretized_curve[method] = discretized_curve_folds
    #print (self.discretized_curve)

    
def plot_lc(log_file_name, y_min=0, y_max=0.5, name=''):
    x_list, y_list = get_lc_from_log(log_file_name)
    plt.step(x_list, y_list, where='post', label=name)
    # plt.ylim([y_min,y_max])
    plt.yscale('log')

    print('plot lc')

def get_agg_lc_from_file(log_file_name_alias, method_alias, index_list=list(range(0,10))):
    log_file_list = []
    list_x_list = []
    list_y_list = []
    for index in index_list:
        log_file_name = log_file_name_alias + '_' + str(index) + '.log'
        try:
            x_list, y_list = get_lc_from_log(log_file_name)
            list_x_list.append(x_list)
            list_y_list.append(y_list)
        except:
            print('Fail to get lc from log')
    
    plot_agg_lc(list_x_list, list_y_list, method_alias= method_alias, y_max=1,)

def plot_agg_lc(list_x_list, list_y_list, y_max, method_alias=''):

    def get_x_y_from_index(current_index, x_list, y_list, y_max):
        if current_index ==0:
            current_index_x, current_index_y =  x_list[current_index], y_max
        else: 
            current_index_x, current_index_y = x_list[current_index], y_list[current_index-1]
        return current_index_x, current_index_y

    import itertools
    all_x_list = list(itertools.chain.from_iterable(list_x_list))
    all_x_list.sort()
    all_y_list = []
    for fold_index, y_list in enumerate(list_y_list):
        if len(y_list) == 0:
            continue
        all_y_list_fold = []
        x_list = list_x_list[fold_index]
        current_index = 0
        
        for x in all_x_list:
            current_index_x, best_y_before_current_x = get_x_y_from_index(current_index, x_list, y_list, y_max)
            if x < current_index_x:
                all_y_list_fold.append(best_y_before_current_x)
            else: 
                all_y_list_fold.append(y_list[current_index])
                if current_index < len(y_list)-1:
                    current_index +=1
        all_y_list.append(all_y_list_fold)
    all_y_list_arr = np.array(all_y_list)
    all_y_list_mean = np.mean(all_y_list_arr, axis=0)
    all_y_list_std = np.std(all_y_list_arr, axis=0)

    # plt.plot(all_x_list, all_y_list_mean, label = method_alias)
    plt.step(all_x_list, all_y_list_mean, where='post', label = method_alias)
    plt.fill_between(all_x_list, all_y_list_mean - all_y_list_std, all_y_list_mean + all_y_list_std, alpha=0.4)
    plt.yscale('log')