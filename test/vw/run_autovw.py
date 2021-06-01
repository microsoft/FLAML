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
from vw_benchmark.utils import plot_progressive_loss
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


def get_loss(y_pred, y_true, loss_func='squared'):
    if 'squared' in loss_func:
        loss = mean_squared_error([y_pred], [y_true])
    elif 'absolute' in loss_func:
        loss = mean_absolute_error([y_pred], [y_true])
    else:
        loss = None
        raise NotImplementedError
    return loss


def online_learning_loop(iter_num, vw_examples, Y, vw_alg, loss_func,
                         method_name='', result_file_address='./res.json',
                         demo_champion_detection=False):
    """Implements the online learning loop.
    Args:
        iter_num (int): The total number of iterations
        vw_examples (list): A list of vw examples
        Y: ground-truth label
        alg (alg instance): An algorithm instance has the following functions:
            - alg.learn(example)
            - alg.predict(example)
        loss_func (str): loss function
        method_name (str): the name of the method
        result_file_address: where to save results
        demo_champion_detection (bool): whether to demo champion detection

    Outputs:
        cumulative_loss_list (list): the list of cumulative loss from each iteration.
            It is returned for the convenience of visualization.
        champion_detection (list): a list of time index where a new champion is 
            detected.
    """
    print('rerunning exp....', len(vw_examples), iter_num)
    result_log = ResultLogWriter(result_file_address, loss_metric=loss_func,
                                 method_name=method_name)
    result_log.open()
    loss_list = []
    y_predict_list = []
    old_champion = ''
    champion_detection_iter = []
    for i in range(iter_num):
        start_time = time.time()
        vw_x = vw_examples[i]
        y_true = get_y_from_vw_example(vw_x)
        # predict step
        y_pred = vw_alg.predict(vw_x)
        # learn step
        vw_alg.learn(vw_x)
        if demo_champion_detection and hasattr(vw_alg, 'get_champion_id'):
            champion_id = vw_alg.get_champion_id()
            if champion_id != old_champion:
                champion_detection_iter.append(i)
                old_champion = champion_id
        # calculate one step loss
        loss = get_loss(y_pred, y_true, loss_func)
        loss_list.append(loss)
        y_predict_list.append([y_pred, y_true])
        # save results
        result_log.append(record_id=i, y_predict=y_pred, y=y_true, loss=loss,
                          time_used=time.time() - start_time,
                          incumbent_config=None,
                          champion_config=None)
    result_log.close()
    return loss_list, champion_detection_iter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', type=str, nargs='?', default='final',
                        help="The benchmark type to run as defined by default \
                        in resources/benchmarks/{benchmark}.yaml")
    parser.add_argument('exp_config', type=str, nargs='?', default='test',
                        help="The constraint definition to use as defined by \
                        default in resources/constraints.yaml. Defaults to `test`.")
    parser.add_argument('-d', '--dataset_list', metavar='dataset_list', nargs='*',
                        default=[], help="The specific dataset id (as defined in \
                        the benchmark file) to run. " "When an OpenML reference is \
                        used as benchmark, the dataset name should be used instead. "
                        "If not provided, all datasets from the benchmark will be run.")
    parser.add_argument('-m', '--method_list', dest='method_list', nargs='*',
                        default=[], help="The method list")
    parser.add_argument('-seed', '--seed_index', dest='seed_index', nargs='*', type=int,
                        default=[0], help="set config_oracle_random_seed seed_index")
    parser.add_argument('-rerun', '--force_rerun', action='store_true',
                        help='whether to force rerun.')
    parser.add_argument('-shuffle', '--shuffle_data', action='store_true',
                        help='whether to force rerun.')
    parser.add_argument('-log', '--use_log', action='store_true',
                        help='whether to use_log.')
    parser.add_argument('-demo', '--demo_champion', action='store_true',
                        help='whether to demo_champion.')
    parser.add_argument('-p', '--show_plot', action='store_true',
                        help='whether to demo_champion.')
    parser.add_argument('-a', '--aggregate', action='store_true',
                        help='whether to aggregate results.')
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
    current_exp_config = yaml.load(open(RESOURCE_DIR + 'exp_config.yaml', 'r',
                                        encoding="utf-8"))[args.exp_config]
    
    # get the list of dataset from yaml
    if args.dataset_list:
        dataset_list = args.dataset_list
    else:
        dataset_list = list(benchmark_info.keys())
    iter_num = current_exp_config['max_sample_num']
    # setup alg configs
    fixed_hp_config = {'alg': 'supervised', 'loss_function': 'squared'}
    random_seed_list = [basic_config_info['run_random_seeds'][i] for i in args.seed_index]
    log_sig = '_'.join([args.benchmark] + [args.exp_config] + args.dataset_list
                       + args.method_list + [str(i) for i in args.seed_index])
   
    log_file_name = LOG_DIR + '_'.join([args.benchmark] + [args.exp_config, current_exp_config.get('hp_to_tune','NS')]
                                       + args.dataset_list + args.method_list
                                       + [str(i) for i in args.seed_index]) + '.log'
    if args.show_plot:
        log_file_name = LOG_DIR + 'plot.log'
    logging.basicConfig(
        filename=log_file_name,
        format='%(asctime)s %(name)s:%(levelname)s:%(message)s',
        filemode="w", level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    scores_per_dataset_per_randomseed = {}
    ###testing
    for dataset in dataset_list:
        problem_args = {"max_iter_num": current_exp_config['max_sample_num'],
                        "dataset_id": benchmark_info[dataset]['dataset_id'],
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
        res_dir = RES_LOG_DIR + dataset + '/'
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
            method_results = {}  # key: method name, value: result 
            method_cum_loss = {}  # key: method name, value: result 
            for alg_name in args.method_list:
                if alg_name in method_data.keys():
                    alg_args = method_data[alg_name]
                    alg_alias = '_'.join([alg_name, args.exp_config])
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
            method_final_losses = {}
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
                if os.path.exists(result_file_address) and not args.force_rerun:
                    champion_detection = []
                    print('---result file exists and loading res from:', result_file_address)
                    with open(result_file_address) as f:
                        for line in f:
                            try:
                                cumulative_loss_list.append(json.loads(line)['loss'])
                            except:
                                logger.warn('result format error')
                    print('---finished loading')
                elif not args.aggregate:
                    # run online experiment
                    logger.info('---------------Start running %s on dataset %s--------------', alg_name, dataset)
                    cumulative_loss_list, champion_detection = online_learning_loop(
                        max_iter_num, vw_online_aml_problem.vw_examples, vw_online_aml_problem.Y, alg,
                        loss_func=fixed_hp_config['loss_function'],
                        method_name=alg_name, result_file_address=result_file_address,
                        demo_champion_detection=args.demo_champion)
                    logger.critical('%ss running time: %s, total iter num is %s',
                                    alg_name, time.time() - time_start, iter_num)
                else:
                    print('NEED to return file', result_file_address)
                # save result
                if cumulative_loss_list:
                    logger.debug('calling algname')
                    method_results[alg_name] = sum(cumulative_loss_list) / len(cumulative_loss_list)
                    if alg_name not in method_cum_loss:
                        method_cum_loss[alg_name] = []
                    method_cum_loss[alg_name].append(cumulative_loss_list)
                    method_final_losses[alg_name] = sum(cumulative_loss_list) / len(cumulative_loss_list)
                else:
                    print('alg not exist')

            if method_cum_loss and args.show_plot and not args.aggregate:
                exp_info = [args.benchmark, args.exp_config, random_seed]
                fig_name = PLOT_DIR + vw_online_aml_problem.problem_id \
                                    + '-' + str(random_seed) + '.pdf'
                plot_progressive_loss(method_cum_loss, fig_name)
            alg_final_losses = {}
            normalized_scores = {}
            for alg_name, final_loss in method_final_losses.items():
                m_name = extract_method_name_from_alg_name(alg_name)
                alg_final_losses[m_name] = final_loss
                print('alg final loss', m_name, alg_name, final_loss)
            if 'Vanilla' in alg_final_losses and 'Exhaustive' in alg_final_losses:
                for m_name in alg_final_losses:
                    if 'Exhaustive' != m_name and 'Vanilla' != m_name:
                        if (alg_final_losses['Vanilla'] - alg_final_losses["Exhaustive"]) == 0:
                            normalized_scores[m_name] = np.inf
                        else:
                            normalized_scores[m_name] = \
                                (alg_final_losses['Vanilla'] - alg_final_losses[m_name])/(alg_final_losses['Vanilla'] - alg_final_losses["Exhaustive"])
                            
            alg_final_score_per_random_seed[random_seed] = normalized_scores

        print(alg_final_score_per_random_seed)
        print(datetime.now().strftime("%m%d-%H"))
        time_string = datetime.now().strftime("%m%d-%H")
        scores_per_dataset_per_randomseed[dataset] = alg_final_score_per_random_seed
    print('score per', scores_per_dataset_per_randomseed)
    agg_res_alias = 'normalized_scores' + ('-').join(
        [args.benchmark, args.exp_config, str(len(dataset_list)),
         str(len(random_seed_list))]) + time_string
    
    if len(dataset_list) < 5:
        agg_res_alias = agg_res_alias + '-' +'-'.join([str(d) for d in dataset_list])
    with open(AGGREGATE_RES_DIR + agg_res_alias + ".json", "w") as outfile: 
        json.dump(scores_per_dataset_per_randomseed, outfile)
