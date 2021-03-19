# name mapping for evaluation metric for glue

eval_name_mapping = {
    "cola": ("mcc"),
    "mnli": ("acc"),
    "mrpc": ("acc", "f1"),
    "qnli": ("acc"),
    "qqp":  ("acc", "f1"),
    "rte":  ("acc"),
    "sst2": ("acc"),
    "stsb": ("pearson", "spearman"),
    "wnli": ("eval_acc")
}
