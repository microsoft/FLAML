
VW_DS_DIR = './test/vw/vw_benchmark/data/openml_vwdatasets/'
LOG_DIR = './test/vw/vw_benchmark/result/log/'

STDOUT_DIR = './test/vw/vw_benchmark/result/stdout/'
PLOT_DIR = './test/vw/vw_benchmark/result/plots/'
AGGREGATE_RES_DIR = './test/vw/vw_benchmark/result/'
MAIN_RES_LOG_DIR = './test/vw/vw_benchmark/result/result_log/'

AB_RES_LOG_DIR = './test/vw/vw_benchmark/result/result_log/'
RESOURCE_DIR = './test/vw/resources/'

RANDOM_SEED = 20201234
QW_OML_API_KEY = '8c4eebcda506ae1065902c2b224369b9'
WARMSTART_NUM = 50
CANDIDATE_SIZE = 10
ORACLE_RANDOM_SEED = 2345
FONT_size_label = 18
FONT_size_stick_label = 12
fixed_hp_config = {'alg': 'supervised', 'loss_function': 'squared'}  
CSFONT = {'fontname': 'Times New Roman'}
LEGEND_properties = {"size": 14}

significant_dataset_ids = [201, 1191, 215, 344, 537, 564, 1196, 1199, 1203, 1206, 
5648, 23515, 41506, 41539, 42729, 42496] #42495 (missing value),
ICML_DATASET_10NS = [201, 1191, 215, 344, 537, 564, 1196, 1199, 1203, 1206, 
5648, 23515, 41506, 41539, 42729, 42496]

dataset_ids = [573, 1201, 1195, 344, 1192, 201, 216, 41065, 42731, 42712, 4545, 42688, 1196, 23515, 1206, 
1193, 42721, 42571, 42713, 537, 42724, 41540, 4549, 296, 574, 218, 5648, 215, 41539, 1199, 1203, 1191, 564, 
1208, 42225, 42728, 42705, 42729]

# no difference
no_diff_id = [1195]
small = [573]

other_no_diff = [204]
not_regression = [1202, 23395, 23397, 5889, 5587, 41928, 42110, 42130, 42111, 42112,\
    42113, 42677, 42669, 42636, 42207, 42131, 1198, 42720, 41463, 1213, 
    1194, 1197, 1200, 1204, 1189, 1188, 1190]              

OPENML_REGRESSION_LIST_larger_than_1k = [572, 545, 1595, 1575, 1433, 1414, 1202, 1196, 23395, 23397, 5889, 5648, \
     5587, 41539, 42559, 42545, 1195, 42092, 42130, 42677, 42669, 42635, 197, 225, 227, 42207, 42208, 42131, 42183, 218,\
          189, 198, 564, 562, 1198, 344, 573, 42728, 42712, 42713, 42720, 503, 688, 215, 41463, 1213, 1203, 1194, 1201, \
               1197, 1192, 1193, 1200, 1204, 1191, 1189, 1188, 1199, 1190, 537, 42721, 42225, 42731, 42729, 287, 216, \
                   41540, 42688, 574, 23515]
                   
OPENML_REGRESSION_LIST_inst_larger_than_5k = [572, 545, 1595, 1575, 1433, 1414, 1202, 1196, 23395, 23397, 5889, 5648, \
     5587, 41539, 42559, 42545, 1195, 42092, 42130, 42677, 42669, 42635, 197, 225, 227, 42207, 42208, 42131, 42183, 218,\
          189, 198, 564, 562, 1198, 344, 573, 42728, 42712, 42713, 42720, 503, 688, 215, 41463, 1213, 1203, 1194, 1201, \
               1197, 1192, 1193, 1200, 1204, 1191, 1189, 1188, 1199, 1190, 537, 42721, 42225, 42731, 42729, 287, 216, \
                   41540, 42688, 574, 23515]

 
OPENML_REGRESSION_LIST_larger_than_5k_no_missing = [572, 545, 1595, 1575, 1433, 1414, 1202, 1196, 23395, 23397, 5889, 5648, \
     5587, 41539, 42559, 42545, 1195, 42092, 42130, 42677, 42669, 42635, 197, 225, 227, 42207, 42208, 42131, 42183, 218,\
          189, 198, 564, 562, 1198, 344, 573, 42728, 42712, 42713, 42720, 503, 688, 215, 41463, 1213, 1203, 1194, 1201, \
               1197, 1192, 1193, 1200, 1204, 1191, 1189, 1188, 1199, 1190, 537, 42721, 42225, 42731, 42729, 287, 216, \
                   41540, 42688, 574, 23515]
                   
# qualities.NumberOfInstances:>10000 qualities.NumberOfFeatures:<27 qualities.NumberOfClasses:0
# total num: 62
OPENML_REGRESSION_LIST_inst_larger_than_10k = [11595, 1575, 1414, 1202, 1196, 23395, 23397, 5889, 5648, 5587, \
    40753, 41539, 41506, 42559, 42496, 42495, 1195, 42092, 42080, 42130, 42677, 42669, 42635, 42207, 42208, 42131, \
        42160, 42183, 218, 564, 1198, 344, 42728, 42712, 42713, 42720, 215, 41463, 1213, 1203, 1194, 1201, 1197, 1192, \
            1193, 1200, 1204, 1191, 1189, 1188, 1199, 1190, 537, 42721, 42225, 42731, 42729, 216, 41540, 42688, 574, 23515
]

# qualities.NumberOfInstances:>100000 qualities.NumberOfFeatures:<27 qualities.NumberOfClasses:0
# total num: 33


FINAL_METHOD_alias = {
    'Exhaustive': 'Exhaustive',
    'Vanilla': 'Vanilla',
    'Random': 'Random',
    'ChaCha': 'ChaCha-noremove',
    'ChaCha_remove': 'ChaCha',
    'Vanilla_100k_1_5': 'Vanilla',
    'Vanilla_100k_1_5_ns+lr': 'Vanilla',
    'Vanilla_1000k_1_5': 'Vanilla',
    'Vanilla_1000k_1_5_ns+lr': 'Vanilla',
    'Exhaustive_100k_1_5': 'Exhaustive',
    'Exhaustive_100k_1_5_ns+lr': 'Exhaustive',
    'Exhaustive_1000k_1_5': 'Exhaustive',
    'Exhaustive_1000k_1_5_ns+lr': 'Exhaustive',
    'Random_100k_1_5': 'Random:NI',
    'Random_100k_1_5_ns+lr': 'Random:NI+LR',
    'Random_1000k_1_5': 'Random:NI',
    'Random_1000k_1_5_ns+lr': 'Random:NI+LR',
    'ChaCha_100k_1_5': 'ChaCha:NI',
    'ChaCha_100k_1_5_ns+lr': 'ChaCha:NI+LR',
    'ChaCha_1000k_1_5': 'ChaCha:NI',
    'ChaCha_1000k_1_5_ns+lr': 'ChaCha:NI+LR',
    'ChaCha_remove_100k_1_5': 'ChaCha:NI',
    'ChaCha_remove_100k_1_5_ns+lr': 'ChaCha:NI+LR',
    'ChaCha_remove_1000k_1_5': 'ChaCha:NI',
    'ChaCha_remove_1000k_1_5_ns+lr': 'ChaCha:NI+LR',
    'OfflineVW': 'Offline'
}

FINAL_METHOD_alias_key_list = ['Vanilla', 'Random', 'Random:NI', 'Random:NI+LR', 'Exhaustive',
                                'ChaCha', 'ChaCha:NI','ChaCha:NI+LR',
                               ]

FINAL_METHOD_color = {'Vanilla': 'tab:brown',
                      'Exhaustive': 'black',
                      'Exhaustive:NI': 'black',
                      'Exhaustive:NI+LR': 'black',
                      'Random': 'tab:red',
                      'Random:NI': 'tab:red',
                      'Random:NI+LR': 'tab:red',
                      'ChaCha': 'tab:blue',
                      'ChaCha:NI': 'tab:blue',
                      'ChaCha:NI+LR': 'tab:blue',
                      'OfflineVW': 'tab:pink'
                    }

FINAL_METHOD_line = {
    'Vanilla': '-',
    'Exhaustive': '-',
    'Exhaustive:NI': '-',
    'Exhaustive:NI+LR': '-',
    'Random': '-',
    'Random:NI': '-',
    'Random:NI+LR': '-',
    'ChaCha': '-',
    'ChaCha:NI': '-',
    'ChaCha:NI+LR': '-',
    'OfflineVW': '-',
}

FINAL_METHOD_marker = {
    'Vanilla': '',
    'Exhaustive': 'x',
    'Exhaustive:NI': 'x',
    'Exhaustive:NI+LR': 'x',
    'Random': 'o',
    'Random:NI': 'o',
    'Random:NI+LR': 's',
    'ChaCha': 's',
    'ChaCha:NI': 'o',
    'ChaCha:NI+LR': 's',
    'OfflineVW': 's',
}


FINAL_METHOD_hatch= {
    'Vanilla': '',
    'Exhaustive': '',
    'Exhaustive:NI': '-',
    'Exhaustive:NI+LR': '-',
    'Random': 'x',
    'Random:NI': 'x',
    'Random:NI+LR': 'x',
    'ChaCha': '/',
    'ChaCha:NI': '/',
    'ChaCha:NI+LR': '/',
    'OfflineVW': '/',
}