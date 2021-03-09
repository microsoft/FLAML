import os
import numpy as np
import ast
import json
def get_lc_from_log(log_file_name):
    x = []
    y = [] #minimizing 
    if os.path.isfile(log_file_name):
        best_obj = np.inf
        total_time = 0
        best_config = None
        with open(log_file_name, "r") as ins:
            for line in ins:
                data = json.loads(line)
                try:
                    total_time = float(data['total_search_time'])
                    obj = float(data['validation_loss'])
                    config = data['config']
                    if obj < best_obj: 
                        best_obj = obj
                        x.append(total_time)
                        y.append(best_obj)
                        best_config = config
                except:
                    print('end')
    else:
        print('log_file_name', log_file_name)
        print('File does not exist')
    assert len(x) == len(y) 
    print(x, y, best_config)
    return x, y


def plot_lc(log_file_name, y_min=0, y_max=0.5, name=''):
    import matplotlib.pyplot as plt
    x_list, y_list = get_lc_from_log(log_file_name)
    plt.step(x_list, y_list, where='post', label=name)
    # plt.ylim([y_min,y_max])
    plt.yscale('log')

    print('plot lc')

file_name = './test/xgboost2d_credit-g_grid_223.log'
file_name_2 = './test/xgboost2d_credit-g_grid_446.log'
file_name_3 = './test/xgboost2d_credit-g_grid_670.log'
file_name_4 = './test/xgboost2d_credit-g_grid_335.log'

get_lc_from_log(file_name)
get_lc_from_log(file_name_4)
get_lc_from_log(file_name_2)
get_lc_from_log(file_name_3)