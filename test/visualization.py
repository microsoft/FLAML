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


def log_file_name(problem, dataset, method, budget, run):
    '''log file name
    '''
    return f'logs/{problem}/{problem}_1_1_{dataset}_{budget}_{method}_{run}.log'


def agg_final_result(problems, datasets, methods, budget, run):
    '''aggregate the final results for problems * datasets * methods
    
    Returns:
        A dataframe with aggregated results
        e.g., problems = ["xgb_blendsearch", "xgb_cfo", "xgb_hpolib"]
        output = df
index | xgb_blendsearch | xgb_cfo         | xgb_hpolib         | winner          | winner_method
blood | (0.2, 1900, BS) | (0.3, 109, CFO) | (0.4, 800, Optuna) | xgb_blendsearch | CFO
        ...
        the numbers correspond to the best loss and minimal time used 
            by all the methods for
            blood * xgb_blendsearch, blood * xgb_cfo, blood * xgb_hpolib etc.
    '''
    results = {}
    columns = problems
    for i, problem in enumerate(problems):
        for j, dataset in enumerate(datasets):
            for method in methods:
                key = (j, i)
                x, y = get_lc_from_log(log_file_name(
                    problem, dataset, method, budget, run))
                if len(y)>0: 
                    result = results.get(key, [])
                    if not result: results[key] = result                    
                    result.append((y[-1],x[y.index(y[-1])],method))
    import pandas as pd
    agg_results = pd.DataFrame(columns=columns, index=datasets, dtype=object)
    for key, value in results.items():
        row_id, col_id = key
        agg_results.iloc[row_id, col_id] = min(value)
    best_obj = agg_results.min(axis=1)
    winner = []
    winner_method = []
    for dataset in datasets:
        for problem in problems:
            if agg_results[problem][dataset] == best_obj[dataset]:
                    winner.append(problem)
                    winner_method.append(agg_results[problem][dataset][2])
                    break
    agg_results['winner'] = winner
    agg_results['winner_method'] = winner_method
    return agg_results


def final_result(problem, datasets, methods, budget, run):
    '''compare the final results for problem on each dataset across methods
    
    Returns:
        A dataframe for each dataset
        e.g., methods = ["CFO", "BlendSearch+Optuna", "Optuna"]
        output = df
            index | CFO | BlendSearch+Optuna | Optuna      | winner
            blood | (0.2, 312) | (0.2, 350)  | (0.4, 1800) | CFO
            ...
        the numbers correspond to the best loss and nimial time used 
            among all the methods for each dataset
    '''
    results = {}
    columns = methods
    import pandas as pd
    agg_results = pd.DataFrame(columns=columns, index=datasets)
    for row_id, dataset in enumerate(datasets):
        for col_id, method in enumerate(methods):
            x, y = get_lc_from_log(log_file_name(
                problem, dataset, method, budget, run))
            if len(y)>0: 
                agg_results.iloc[row_id, col_id] = (y[-1],x[y.index(y[-1])])
    best_obj = agg_results.min(axis=1)
    winner = []
    for dataset in datasets:
        for method in methods:
            if agg_results[method][dataset] == best_obj[dataset]:
                    winner.append(f'{method}')
                    break
    agg_results['winner'] = winner
    return agg_results


def test_agg_final_result():
    print(agg_final_result(['xgb_blendsearch', 'xgb_cfo', 'xgb_hpolib'],
        ['Australian', 'blood', 'car', 'credit', 'kc1', 'kr', 'phoneme', 'segment'], 
        ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
        3600.0, 0))
    print(agg_final_result(['xgb_blendsearch', 'xgb_cfo', 'xgb_hpolib'],
        ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine',], 
        ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
        14400.0, 0))


def test_final_result():
    print('xgb_cfo')
    print(final_result('xgb_cfo',
        ['Australian', 'blood', 'car', 'credit', 'kc1', 'kr', 'phoneme', 'segment'], 
        ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
        3600.0, 0))
    print(final_result('xgb_cfo',
        ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine',], 
        ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
        14400.0, 0))


if __name__ == "__main__":
    test_agg_final_result()
    test_final_result()