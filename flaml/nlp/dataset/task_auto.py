# https://github.com/huggingface/datasets/blob/master/metrics/glue/glue.py

from collections import OrderedDict
from typing import Tuple

task_mapping_glue_stsb = "regression"

task_mapping_glue_other = "seq-classification"

task_mapping_anli = "seq-classification"

task_mapping_hate_speech18 = "seq-classification"

task_mapping_squad = "question-answering"

task_mapping_dbpedia = "seq-classification"

task_mapping_super_glue_other = "seq-classification"

task_mapping_imdb = "seq-classification"

task_mapping_sentiment140 = "regression"

task_mapping_hyperpartisan = "seq-classification"

TASK_MAPPING = OrderedDict(
    [
        (("squad", ""), task_mapping_squad),
        (("glue", "stsb"), task_mapping_glue_stsb),
        (("glue", "cola"), task_mapping_glue_other),
        (("glue", "mnli"), task_mapping_glue_other),
        (("glue", "mrpc"), task_mapping_glue_other),
        (("glue", "qnli"), task_mapping_glue_other),
        (("glue", "qqp"), task_mapping_glue_other),
        (("glue", "rte"), task_mapping_glue_other),
        (("glue", "sst2"), task_mapping_glue_other),
        (("glue", "wnli"), task_mapping_glue_other),
        (("dbpedia_14", ""), task_mapping_dbpedia),
        (("imdb", ""), task_mapping_imdb),
        (("super_glue", "rte"), task_mapping_super_glue_other),
        (("super_glue", "wic"), task_mapping_super_glue_other),
        (("hate_speech18", ""), task_mapping_hate_speech18),
        (("sentiment140", ""), "regression"),
        (("yelp_review_full", ""), "regression"),
        (("amazon_polarity", ""), task_mapping_imdb),
        (("amazon_reviews_multi", ""), "regression"),
        (("yelp_polarity", ""), task_mapping_imdb),
        (("anli", ""), task_mapping_anli),
        (("hyperpartisan_news_detection", ""), task_mapping_hyperpartisan),
    ]
)


def get_default_task(dataset_name_list: Tuple, custom_task):
    if custom_task is not None:
        return custom_task

    assert dataset_name_list in TASK_MAPPING, (
        "The dataset is not in {}, you must explicitly specify "
        "the custom_metric_name and custom_metric_mode_name".format(
            ", ".join(["-".join(x) for x in TASK_MAPPING.keys()])
        )
    )
    eval_name_mapping = TASK_MAPPING[dataset_name_list]
    return eval_name_mapping
