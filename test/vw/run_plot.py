import numpy as np
import json
import argparse
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
from vw_benchmark.config import LOG_DIR, PLOT_DIR, MAIN_RES_LOG_DIR, RESOURCE_DIR, AGGREGATE_RES_DIR
from run_autovw import extract_method_name_from_alg_name
FONT_size_label = 18
FINAL_METHOD_alias = {}
FINAL_METHOD_color = {}
FINAL_METHOD_line = {}
FINAL_METHOD_marker = {}
FINAL_METHOD_hatch = {}
LEGEND_properties = {"size": 14}

FINAL_METHOD_alias = {
    'Exhaustive': 'Exhaustive',
    'Random': 'Random',
    'Vanilla': 'Vanilla',
    'ChaCharemove': 'ChaCha',
    'ChaCha': 'ChaCha-noremove',
    'ChaChawo': 'ChaCha-w/o-Champion',
    'ChaChaaggressive': 'ChaCha-AggressiveScheduling'
}

FINAL_METHOD_alias_key_list = ['Random', 'Vanilla', 'ChaCha', 'Exhaustive']

FINAL_METHOD_color = {
    'Random': 'tab:red',
    'ChaCha': 'tab:blue',
    'ChaCha-AggressiveScheduling': 'tab:green',
    'ChaCha-w/o-Champion': 'tab:orange'
    # 'ChaCha': 'tab:green',
}

FINAL_METHOD_line = {
    'Random': '-',
    'ChaCha': '--',
}

FINAL_METHOD_marker = {
    'Random': 'o',
    'naiveVW': 'x',
}


FINAL_METHOD_hatch= {
        'Exhaustive': '/',
        'Random': '+',
        'Vanilla': '-',
        'ChaCharemove': '+',
        'ChaCha': '/',
        'ChaChawo': 'x',
        'ChaChaaggressive': 'x'
}



def normalized_scores_barplot(res_dic, alias='', method_list_to_show = [], error_bar=False, print_table=False,name_0='naive', name_1='fixed-50'):
    all_names = [ 'fixed-5-VW', 'ChaCha-CB','ChaCha-Org','ChaCha', 'ChaCha-Final','ChaCha-nochampion-top0', 'ChaCha-ucb-top0','Chambent-Hybrid', 'Chambent-Van-ucb-tophalf']
    
    table_method_names = ['fixed-5-VW','Chambent-Hybrid', 'ChaCha-nochampion-top0', 'ChaCha-ucb-top0']
    table_name_label = {
        'fixed-5-VW': '\RandomInit' ,
        'Chambent-Hybrid': '\Chambent' , 
        'ChaCha-nochampion-top0':'\Chambent-w/o-Champion', 
        'ChaCha-ucb-top0': '\Chambent-AggressiveScheduling',
    }
    table_lines = []
    table_line_0 = 'Dataset id &' + (' & ').join([table_name_label[n] for n in table_method_names]) + ' \\\ \hline'
    # print 
    print('resdic', res_dic)
    if not error_bar: 
        result = [res_dic]
    else:
        assert type(res_dic)==list 
        result = res_dic
    # print(error_bar, type(result), type(result[0]))
    datasets = list(result[0].keys())
    labels = datasets 
    all_res1=[]
    all_res2=[]
    print(datasets)
    fig, ax = plt.subplots()
    if len(method_list_to_show) <3: 
        barWidth = 0.25
        loc_num = 2
        blk = False
    else: 
        barWidth = 0.2
        loc_num = 1
        blk = True

    r_position = []
    r0 = np.arange(len(labels))
    r_position.append(r0)
    table_method_mean ={}
    table_method_std = {}
    for i,m_name in enumerate(all_names):
        all_res = []
        for res_dic in result:
            normalized_score = get_normalized_score(res_dic, m_name, datasets)
            all_res.append(normalized_score)
        m_mean = np.mean(all_res, axis=0)
        m_std = np.std(all_res, axis=0)
        m_alias = FINAL_METHOD_alias[m_name] if m_name in FINAL_METHOD_alias else 'None'
        m_color = FINAL_METHOD_color[m_name] if m_name in FINAL_METHOD_alias else 'red'
        m_hatch = FINAL_METHOD_hatch[m_name] if m_name in FINAL_METHOD_alias else 'x'
        if m_name in method_list_to_show:
            r_new = [x + barWidth for x in r_position[-1]]
            r_position.append(r_new) 
            plt.bar(r_position[-1], m_mean, width=barWidth, yerr=m_std, label=m_alias, color=m_color, ecolor='grey' ) #hatch = m_hatch,
        if m_name in table_method_names:
            table_method_mean[m_name] = m_mean
            table_method_std[m_name] = m_std
    ###generate table lines
    if print_table:
        print(table_line_0) 
        for i, d_id in enumerate(labels):
            mean_list = [table_method_mean[method][i] for method in table_method_names ]
            max_mean = max(mean_list)
            print_mean_list = {}
            for j, mean in enumerate(mean_list):
                if mean != max_mean:
                    print_mean = '{:.2f}'.format(float(mean))
                else: print_mean = '\\textbf{'+ '{:.2f}'.format(float(mean)) + '}'
                print_mean_list[table_method_names[j]] = print_mean
            res_list = [print_mean_list[method]+ ' $\pm$ ' + \
                 '{:.2f}'.format(float(table_method_std[method][i])) for method in table_method_names]
            data_line = str(d_id) + ' & ' +  (' & ').join(res_list) + ' \\\ '
            print(data_line)
    # x = np.arange(len(labels))
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized score', fontsize=FONT_size_label)
    # ax.set_title('Normalzied Scores (naive=0, ExhaustInit=1)')
    ax.set_xticks(r0)
    ax.set_xticklabels(labels,rotation=50)
    ax.set_ylim(-0.5,2.0)
    if blk:
        ax.legend(loc='upper left', ncol= loc_num, prop={'weight':'bold' }) #prop=LEGEND_properties
    else:
        ax.legend(loc='upper left', ncol= loc_num,prop=LEGEND_properties) #prop=LEGEND_properties
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # autolabel(rects1)
    # autolabel(rects2)
    # fig.tight_layout()
    # plt.axhline(y=1.0, color='r', linestyle='--')
    fig_name = PLOT_DIR  + 'bar_plot_' + alias+ '.pdf'
    plt.savefig(fig_name)
    plt.close()


def normalized_scores_barplot(res_dic, method_list=[], alias='test'):
    # labels are the datasets
    datasets = list(res_dic.keys())
    datasets.sort()
    random_seeds = list(res_dic[datasets[0]].keys())
    method_names = list(res_dic[datasets[0]][random_seeds[0]].keys()) if not method_list else method_list
    method_names.sort()
    method_names = method_names[::-1]
    logger.info('dataset num: %s', len(datasets))
    logger.info('datasets %s', datasets)
    logger.info('random seeds %s', random_seeds)
    logger.info('method names %s', method_names)
    labels = [d.split('_')[1] for d in datasets]
    r_position = []
    r0 = np.arange(len(labels))
    r_position.append(r0)
    if len(method_names) < 3: 
        barWidth = 0.25
        loc_num = 2
        blk = False
    else: 
        barWidth = 0.15
        loc_num = 1
        blk = True
    dataset_results_mean = {}
    dataset_results_std = {}
    for dataset_name, res in res_dic.items():
        dataset_method_results = {}
        dataset_method_results_std = {}
        for method_name in method_names:
            result_list = [res[seed][method_name] for seed in random_seeds if \
                (method_name in res[seed] and isinstance(res[seed][method_name], float)  )]
            dataset_method_results[method_name] = np.mean(result_list, axis=0)
            print(result_list, np.std(result_list, axis=0))
            dataset_method_results_std[method_name] = np.std(result_list, axis=0)
        dataset_results_mean[dataset_name] = dataset_method_results
        dataset_results_std[dataset_name] = dataset_method_results_std
    print('es', res_dic.keys(), labels)
    fig, ax = plt.subplots()
    table_method_mean ={}
    table_method_std = {}
    for method_name, mean_std in dataset_method_results.items():
        if method_name in method_list:
            m_mean = [dataset_results_mean[d_name][method_name] for d_name in datasets]
            m_std = [dataset_results_std[d_name][method_name] for d_name in datasets]
            print('m_std', method_name, m_mean, m_std, )

            m_alias = FINAL_METHOD_alias[method_name] if method_name in FINAL_METHOD_alias else method_name
            m_color = FINAL_METHOD_color[m_alias] if m_alias in FINAL_METHOD_color else 'red'
            m_hatch = FINAL_METHOD_hatch[method_name] if method_name in FINAL_METHOD_alias else 'x'
            # m_mean, m_std = mean_std[0], mean_std[1]
            r_new = [x + barWidth for x in r_position[-1]]
            r_position.append(r_new) 
            # plt.bar(r_position[-1], m_mean, width=barWidth, yerr=m_std, label=m_alias, color=m_color, ecolor='grey' ) #hatch = m_hatch,
            plt.bar(r_position[-1], m_mean, width=barWidth, yerr=m_std, label=m_alias, color=m_color, ecolor='grey') #hatch = m_hatch,
        if method_name in table_method_names:
            table_method_mean[method_name] = m_mean
            table_method_std[method_name] = m_std
    logger.info('dataset_method_results %s', dataset_method_results)
    # labels = datasets
    if print_table:
        print(table_line_0) 
        for i, d_id in enumerate(labels):
            mean_list = [table_method_mean[method][i] for method in table_method_names ]
            max_mean = max(mean_list)
            print_mean_list = {}
            for j, mean in enumerate(mean_list):
                if mean != max_mean:
                    print_mean = '{:.2f}'.format(float(mean))
                else: print_mean = '\\textbf{'+ '{:.2f}'.format(float(mean)) + '}'
                print_mean_list[table_method_names[j]] = print_mean
            res_list = [print_mean_list[method]+ ' $\pm$ ' + \
                 '{:.2f}'.format(float(table_method_std[method][i])) for method in table_method_names]
            data_line = str(d_id) + ' & ' +  (' & ').join(res_list) + ' \\\ '
            print(data_line)
    
    # x = np.arange(len(labels))
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized score', fontsize=FONT_size_label)
    # ax.set_title('Normalzied Scores (naive=0, ExhaustInit=1)')
    ax.set_xticks(r0)
    ax.set_xticklabels(labels, rotation=50)
    ax.set_ylim(-0.5, 2.0)
    if blk:
        ax.legend(loc='upper left', ncol= loc_num, prop={'weight':'bold' }) #prop=LEGEND_properties
    else:
        ax.legend(loc='upper left', ncol= loc_num, prop=LEGEND_properties) #prop=LEGEND_properties
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2.5),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    fig_name = AGGREGATE_RES_DIR + 'bar_plot_' + alias + '.pdf'
    plt.savefig(fig_name)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='?', default='final',
                        help="")
    parser.add_argument('alias', type=str, nargs='?', default='test',
                        help="")
    parser.add_argument('-m', '--method_list', dest='method_list', nargs='*',
                        default=[], help="The method list")
    args = parser.parse_args()
    final_method_list = []
    for name in args.method_list:
        final_method_list.append(extract_method_name_from_alg_name(name))
    # 'normalized_scoresfinal-10k_1_5-15-5.json'
    file_address = AGGREGATE_RES_DIR + str(args.filename)
    # Opening JSON file
    with open(file_address) as json_file:
        data = json.load(json_file)
        # Print the type of data variable
        print("Type:", type(data), data)
        normalized_scores_barplot(data, final_method_list, args.alias)

# python test/vw/run_plot.py  normalized_scoresfinal_test-100k_1_5_ns+lr-15-50506-15.json   ns+lr -m ChaCha Random