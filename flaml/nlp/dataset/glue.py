
sentence_key_mapping = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

eval_name_mapping = {
    "cola": ("eval_mcc"),
    "mnli": ("eval_mnli/acc"),
    "mrpc": ("eval_acc"),
    "qnli": ("eval_acc"),
    "qqp":  ("eval_acc"),
    "rte":  ("eval_acc"),
    "sst2": ("eval_acc"),
    "stsb": ("eval_pearson"),
    "wnli": ("eval_acc")
}

foldername_exceptions = {
    "mnli": ("train", "validation_matched", "test_matched"),
}