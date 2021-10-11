# https://github.com/huggingface/datasets/blob/master/metrics/glue/glue.py
from collections import OrderedDict
from typing import Tuple

metric_mode_mapping_glue_cola = [("matthews_correlation", "max")]

metric_mode_mapping_glue_mrpc = [("accuracy", "max"), ("f1", "max")]

metric_mode_mapping_glue_qqp = [("accuracy", "max"), ("f1", "max")]

metric_mode_mapping_glue_stsb = [("pearson", "max"), ("spearmanr", "max")]

metric_mode_mapping_glue_other = [("accuracy", "max")]

metric_mode_mapping_squad = [("exact_match", "max"), ("f1", "max")]

metric_mode_mapping_super_glue_axb = [("matthews_correlation", "max")]

metric_mode_mapping_super_glue_cb = [("accuracy", "max"), ("f1", "max")]

metric_mode_mapping_super_glue_other = [("accuracy", "max")]

metric_mode_mapping_imdb = [("accuracy", "max")]

metric_mode_mapping_dbpedia = [("accuracy", "max")]

metric_mode_mapping_yelp = [("accuracy", "max")]

metric_mode_mapping_hate_speech18 = [("accuracy", "max")]

metric_mode_mapping_anli = [("accuracy", "max")]

metric_mode_mapping_hyperpartisan = [("accuracy", "max")]

METRIC_MAPPING = OrderedDict(
    [
        (("squad", ""), metric_mode_mapping_squad),
        (("glue", "cola"), metric_mode_mapping_glue_cola),
        (("glue", "mrpc"), metric_mode_mapping_glue_mrpc),
        (("glue", "qqp"), metric_mode_mapping_glue_qqp),
        (("glue", "stsb"), metric_mode_mapping_glue_stsb),
        (("glue", "mnli"), metric_mode_mapping_glue_other),
        (("glue", "qnli"), metric_mode_mapping_glue_other),
        (("glue", "rte"), metric_mode_mapping_glue_other),
        (("glue", "sst2"), metric_mode_mapping_glue_other),
        (("glue", "wnli"), metric_mode_mapping_glue_other),
        (("hate_speech18", ""), metric_mode_mapping_hate_speech18),
        (("super_glue", "axb"), metric_mode_mapping_super_glue_axb),
        (("super_glue", "cb"), metric_mode_mapping_super_glue_cb),
        (("super_glue", "copa"), metric_mode_mapping_super_glue_other),
        (("super_glue", "rte"), metric_mode_mapping_super_glue_other),
        (("super_glue", "wic"), metric_mode_mapping_super_glue_other),
        (("super_glue", "wsc"), metric_mode_mapping_super_glue_other),
        (("super_glue", "wsc.fixed"), metric_mode_mapping_super_glue_other),
        (("super_glue", "boolq"), metric_mode_mapping_super_glue_other),
        (("super_glue", "axg"), metric_mode_mapping_super_glue_other),
        (("imdb", ""), metric_mode_mapping_imdb),
        (("dbpedia_14", ""), metric_mode_mapping_dbpedia),
        (("yelp_review_full", ""), metric_mode_mapping_yelp),
        (("amazon_reviews_multi", ""), metric_mode_mapping_yelp),
        (("amazon_polarity", ""), metric_mode_mapping_yelp),
        (("yelp_polarity", ""), metric_mode_mapping_yelp),
        (("anli", ""), metric_mode_mapping_anli),
        (("hyperpartisan_news_detection", ""), metric_mode_mapping_hyperpartisan),
    ]
)


def get_default_and_alternative_metric(
    dataset_name_list: Tuple,
    custom_metric_name=None,
    custom_metric_mode_name=None,
):
    if dataset_name_list not in METRIC_MAPPING.keys():
        assert custom_metric_name and custom_metric_mode_name, (
            "The dataset is not in {}, you must explicitly specify "
            "the custom_metric_name and custom_metric_mode_name".format(
                ",".join(METRIC_MAPPING.keys())
            )
        )
        return None, None, None, None
    eval_name_mapping = METRIC_MAPPING[dataset_name_list]

    default_metric, default_mode = eval_name_mapping[0]
    all_metrics, all_mode = [x[0] for x in eval_name_mapping] + ["loss"], [
        x[1] for x in eval_name_mapping
    ] + ["min"]

    return default_metric, default_mode, all_metrics, all_mode
