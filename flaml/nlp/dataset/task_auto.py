# https://github.com/huggingface/datasets/blob/master/metrics/glue/glue.py

from collections import OrderedDict

task_mapping_glue = {
    "cola": "seq-classification",
    "mnli": "seq-classification",
    "mrpc": "seq-classification",
    "qnli": "seq-classification",
    "qqp": "seq-classification",
    "rte": "seq-classification",
    "sst2": "seq-classification",
    "stsb": "regression",
    "wnli": "seq-classification"
}

task_mapping_hate_offensive = "seq-classification"

task_mapping_squad = "question-answering"

task_mapping_dbpedia = "seq-classification"

task_mapping_super_glue = {
    "wic": "seq-classification",
    "rte": "seq-classification"
}

task_mapping_imdb = "seq-classification"

TASK_MAPPING = OrderedDict(
    [
        ("squad", task_mapping_squad),
        ("glue", task_mapping_glue),
        ("dbpedia_14", task_mapping_dbpedia),
        ("imdb", task_mapping_imdb),
        ("super_glue", task_mapping_super_glue),
        ("hate_offensive", task_mapping_hate_offensive),
        ("yelp_review_full", "regression"),
        ("amazon_polarity", task_mapping_imdb),
        ("amazon_reviews_multi", "regression"),
        ("yelp_polarity", task_mapping_imdb)
    ]
)


def get_default_task(dataset_name_list: list, subdataset_name=None):
    from ..result_analysis.azure_utils import JobID
    dataset_name = JobID.dataset_list_to_str(dataset_name_list)
    assert dataset_name in TASK_MAPPING.keys(), "The dataset is not in {}, you must explicitly specify " \
                                                "the custom_metric_name and custom_metric_mode_name".format(
        ",".join(TASK_MAPPING.keys()))
    eval_name_mapping = TASK_MAPPING[dataset_name]
    if isinstance(eval_name_mapping, dict):
        assert subdataset_name and subdataset_name in eval_name_mapping, \
            "dataset_name and subdataset_name not correctly specified"
        default_task = eval_name_mapping[subdataset_name]
    else:
        #assert isinstance(eval_name_mapping, list), "dataset_name and subdataset_name not correctly specified"
        default_task = eval_name_mapping
    return default_task
