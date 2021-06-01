import openml
from openml.tasks import TaskType
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')

tasks = openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_REGRESSION)
logger.info('Example task %s', tasks[146542]) # tasks[14949]
lower_data = 10000
upper_data = 10000000
feature_upper_bound = 50
all_regression = set()
all_regression_numerical = set()
for k, v in tasks.items():
    if 'NumberOfInstances' in v and v['NumberOfInstances'] > lower_data and v['NumberOfInstances'] < upper_data:
        if 'NumberOfFeatures' in v and v['NumberOfFeatures'] < feature_upper_bound:
            all_regression.add(v['did'])
            if  'NumberOfNumericFeatures' in v and v['NumberOfFeatures'] == v['NumberOfNumericFeatures']:
                all_regression_numerical.add(v['did'])
            
logger.info('Total dataset number %s', len(all_regression))
logger.info('Dataset ids: %s', all_regression)
logger.info('Total dataset number numerical %s', len(all_regression_numerical))
logger.info('Dataset ids: %s', all_regression_numerical)


# generate yaml file
import yaml
yaml_file_address = './test/vw/resources/benchmarks/all_oml_regression.yaml'
logger.info('Dumping datasets into yaml file: %s', yaml_file_address)
info_dict = {}
for did in all_regression:
    key = 'oml_' + str(did) + '_10'
    info_dict[key] = {
        'type': "openml_regression",
        "dataset_id": did,
        "vw_namespace_num": 10,
        "obj_name": "mse",
        "shuffle": False,
        }
with open(yaml_file_address, 'w') as yaml_file:
    yaml.dump(info_dict, yaml_file, default_flow_style=False)

yaml_file_address_num = './test/vw/resources/benchmarks/all_oml_regression_numerical.yaml'
info_dict_numeric = {}
for did in all_regression_numerical:
    key = 'oml_' + str(did) + '_10'
    info_dict_numeric[key] = {
        'type': "openml_regression",
        "dataset_id": did,
        "vw_namespace_num": 10,
        "obj_name": "mse",
        "shuffle": False,
        }
with open(yaml_file_address_num, 'w') as yaml_file:
    yaml.dump(info_dict_numeric, yaml_file, default_flow_style=False)

# python test/vw/run_datasets.py