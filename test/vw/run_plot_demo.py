import numpy as np
import argparse
import yaml
import json
import time 
from datetime import datetime
import os
from vowpalwabbit import pyvw
from flaml import AutoVW
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from vw_benchmark.config import LOG_DIR, PLOT_DIR, MAIN_RES_LOG_DIR, RESOURCE_DIR, AGGREGATE_RES_DIR
import logging
import matplotlib.pyplot as plt
from vw_benchmark.result_log import ResultLogWriter
from vw_benchmark.vw_utils import get_y_from_vw_example
from vw_benchmark.vw_utils import get_ns_feature_dim_from_vw_example
from vw_benchmark.utils import plot_progressive_loss, plot_progressive_loss_demo
from flaml.tune.online_trial import VWOnlineTrial
from problems.vw_online_regression_problem import VWTuning, VWNSInteractionTuning, VW_NS_LR


logger = logging.getLogger(__name__)

def extract_method_name_from_alg_name(alg_name):
    
    print(alg_name)
    if '_' in alg_name:
        info = alg_name.split('_')
        # if 'pause' in alg_name or 'champion' in alg_name or 'aggressive' in alg_name and len(info) >= 2:
        if len(info) > 4 and 'ChaCha' in alg_name:
            return str(info[0]) + str(info[1])
        return str(info[0])
    else:
        return alg_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', type=str, nargs='?', default='final',
                        help="The benchmark type to run as defined by default \
                        in resources/benchmarks/{benchmark}.yaml")
    parser.add_argument('result_dir', type=str, nargs='?', default='result_dir',
                        help="The benchmark type to run as defined by default \
                        in test/vw/vw_benchmark/result_icml")
    parser.add_argument('-e', '--exp_config_list', dest='exp_config_list', nargs='*',
                        default=[], help="The exp_config list")
    parser.add_argument('-d', '--dataset', metavar='dataset',  default='5648')
    parser.add_argument('-m', '--method_list', dest='method_list', nargs='*',
                        default=[], help="The method list")
    parser.add_argument('-seed', '--seed_index', dest='seed_index', nargs='*', type=int,
                        default=[0], help="set config_oracle_random_seed seed_index")
    parser.add_argument('-shuffle', '--shuffle_data', action='store_true',
                        help='whether to force rerun.')
    parser.add_argument('-demo', '--demo', action='store_true',
                        help='whether to force rerun.')
    parser.add_argument('-log', '--use_log', action='store_true',
                        help='whether to use_log.')
    args = parser.parse_args()
    # setup logs
    RES_LOG_DIR = MAIN_RES_LOG_DIR
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    if not os.path.exists(RES_LOG_DIR):
        os.makedirs(RES_LOG_DIR)

    # **********parse method config, exp config, and dataset info from yaml files****
    # file_constraints = open(RESOURCE_DIR + 'exp_config.yaml', 'r', encoding="utf-8")
    basic_config_info = yaml.load(open(RESOURCE_DIR + 'config.yaml', 'r', encoding="utf-8"))
    benchmark_info = yaml.load(open(RESOURCE_DIR + 'benchmarks/' + args.benchmark + '.yaml',
                                    'r', encoding="utf-8"))
    method_data = yaml.load(open(RESOURCE_DIR + 'methods.yaml', 'r', encoding="utf-8"))
    
    
    # setup alg configs
    fixed_hp_config = {'alg': 'supervised', 'loss_function': 'squared'}
    random_seed_list = [basic_config_info['run_random_seeds'][i] for i in args.seed_index]
    log_file_name = LOG_DIR + 'plot_demo.log'
    logging.basicConfig(
        filename=log_file_name,
        format='%(asctime)s %(name)s:%(levelname)s:%(message)s',
        filemode="w", level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    scores_per_dataset_per_randomseed = {}
    ###testing
    dataset = args.dataset
    method_results = {}  # key: method name, value: result 
    method_cum_loss = {}  # key: method name, value: result 
    for exp_config in args.exp_config_list:
        current_exp_config = yaml.load(open(RESOURCE_DIR + 'exp_config.yaml', 'r',
                                       encoding="utf-8"))[exp_config]
        iter_num = current_exp_config['max_sample_num']
    
        problem_args = {"max_iter_num": current_exp_config['max_sample_num'],
                        "dataset_id": benchmark_info[args.dataset]['dataset_id'],
                        "ns_num": current_exp_config['namespace_num'],
                        "use_log": args.use_log,
                        "shuffle": args.shuffle_data,
                        "vw_format": True
                        }
        # get the problem
        hp_to_tune = current_exp_config.get('hp_to_tune','NS')
        if hp_to_tune == 'NS':
            vw_online_aml_problem = VWNSInteractionTuning(**problem_args)
        elif hp_to_tune == 'NS+LR':
            vw_online_aml_problem = VW_NS_LR(**problem_args)
        else:
            raise NotImplementedError
        # res_dir = RES_LOG_DIR + dataset + '/'
        res_dir = args.result_dir + dataset + '/'
        # key: random seed, value: a dic of alg's normalized score under the random seed
        alg_final_score_per_random_seed = {}
    
        for random_seed in random_seed_list:
            auto_alg_common_args = \
                {'problem': vw_online_aml_problem,
                    "max_live_model_num": current_exp_config['max_live_model_num'],
                    'min_resource_lease': 'auto',
                    'config_oracle_random_seed': random_seed
                    }
            # setup configs for the experiments to run
            alg_dic = {}
            
            for alg_name in args.method_list:
                if alg_name in method_data.keys():
                    alg_args = method_data[alg_name]
                    alg_alias = '_'.join([alg_name, exp_config])
                    if 'is_naive' not in alg_args or not alg_args['is_naive']:
                        autovw_args = auto_alg_common_args.copy()
                        autovw_args.update(alg_args['config'])
                        # use the method_name+current_exp_config as the alias for the algorithm
                        logger.info('alg_alias %s %s', alg_alias, alg_args)
                        logger.info('trial runner config %s %s', alg_args['config'], autovw_args)
                        alg_dic[alg_alias] = AutoVW(**autovw_args)
                    else:
                        vw_args = fixed_hp_config.copy()
                        if 'config' in alg_args and alg_args['config'] is not None:
                            vw_args.update(alg_args['config'])
                        alg_dic[alg_alias] = pyvw.vw(**vw_args)
                else:
                    print('alg name not in methods.yaml')
                    NotImplementedError
            # convert method names from input to the names in alg_dic
            for alg_name, alg in alg_dic.items():
                max_iter_num = vw_online_aml_problem.max_iter_num
                time_start = time.time()
                print('----------running', alg_name, '-----------')
                ### get result file name
                res_file_name = vw_online_aml_problem.problem_id + ('-').join([str(info) for info in [alg_name, random_seed]]) + '.json'
                if not os.path.exists(res_dir):
                    os.makedirs(res_dir)
                result_file_address = res_dir + res_file_name
                ### load result from file
                cumulative_loss_list = []
                if os.path.exists(result_file_address):
                    champion_detection = []
                    print('---result file exists and loading res from:', result_file_address)
                    with open(result_file_address) as f:
                        for line in f:
                            try:
                                cumulative_loss_list.append(json.loads(line)['loss'])
                            except:
                                logger.warn('result format error')
                    print('---finished loading')

                if cumulative_loss_list:
                    logger.debug('calling algname')
                    method_results[alg_name] = sum(cumulative_loss_list) / len(cumulative_loss_list)
                    if alg_name not in method_cum_loss:
                        method_cum_loss[alg_name] = []
                    method_cum_loss[alg_name].append(cumulative_loss_list)
                else:
                    print('alg not exist')

    if method_cum_loss:
        print(method_cum_loss.keys())
        fig_name = PLOT_DIR + vw_online_aml_problem.problem_id + 'all.pdf'
        fig_name_demo = PLOT_DIR + vw_online_aml_problem.problem_id + 'demo.pdf'
        if '1000000' in fig_name:
            result_interval = 1000
        else:
            result_interval = 100

        if args.demo:
            plot_progressive_loss_demo(method_cum_loss, fig_name_demo, result_interval)
        else:
            plot_progressive_loss(method_cum_loss, fig_name, result_interval)
        
        