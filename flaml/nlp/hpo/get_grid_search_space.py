# lookup table for the grid configs in each pre-trained language huggingface for different tasks
import copy

def get_space_union_and_unique(search_space_common, search_space_unique, this_case_tags: list):
    search_space_union = copy.deepcopy(search_space_common)
    this_search_space = copy.deepcopy(search_space_common)
    # enumerate over each case where the search space is different
    # this difference can be the dataset or model size, etc.
    is_included = False
    from ..utils import merge_dicts
    for each_case in search_space_unique.keys():
        from ..utils import _check_dict_keys_overlaps
        if each_case in this_case_tags:
            is_included = True
            assert not _check_dict_keys_overlaps(this_search_space, search_space_unique[each_case]), \
                "the hyperparameters of common and unique search spaces should not have overlaps"
            this_search_space.update(search_space_unique[each_case])
        search_space_union = merge_dicts(search_space_union, search_space_unique[each_case])
    if not is_included:
        if "other" in search_space_unique.keys():
            this_search_space.update(search_space_unique["other"])
            search_space_union = merge_dicts(search_space_union, search_space_unique["other"])
    return search_space_union, this_search_space

def get_longformer_space(model_size_type = None,
                   dataset_name = None,
                   subdataset_name = None):
    """
        Longformer: The Long-Document Transformer
    """
    search_space_dict = {}
    if dataset_name == "glue":
        return

def get_funnel_space(model_size_type = None,
                   dataset_name = None,
                   subdataset_name = None):
    search_space_common = {"learning_rate":[1e-5, 2e-5, 3e-5],
                         "hidden_dropout": [0.1],
                         "activation_dropout": [0.0],
                         "attention_dropout": [0.1],
                         "weight_decay": [0.01],
                         "warmup_ratio": [0.1],
                         "adam_epsilon": [1e-6],
                         }
    search_space_unique = {
        "yelp_review_full": {
            "per_device_train_batch_size": [64],
            "num_train_epochs": [3]
        }
    }
    return get_space_union_and_unique(search_space_common, search_space_unique, dataset_name)

def get_bert_space(model_size_type = None,
                   dataset_name = None,
                   subdataset_name = None):
    """
        BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        https://arxiv.org/pdf/1810.04805.pdf
    """
    search_space_common = {}
    search_space_unique = {
        # Section 4.1: We use a batch size of 32 and fine-tune for 3 epochs over the data for all GLUE tasks. For each
        # task, we selected the best fine-tuning learning rate (among 5e-5, 4e-5, 3e-5, and 2e-5) on the Dev set
        "glue": {
            "learning_rate": [5e-5, 4e-5, 3e-5, 2e-5],
            "per_device_train_batch_size": [32],
            "num_train_epochs": [3],
        },
        # Section 4.2: We fine-tune for 3 epochs with a learning rate of 5e-5 and a batch size of 32
        "squad": {
            "learning_rate": [5e-5],
            "per_device_train_batch_size": [32],
            "num_train_epochs": [2],
        },
        # Section 4.3: We fine-tuned for 2 epochs with a learning rate of 5e-5 and a batch size of 48.
        "squad_v2": {
            "learning_rate": [5e-5],
            "per_device_train_batch_size": [48],
            "num_train_epochs": [2],
        },
        # Section 4.4: We fine-tune the huggingface for 3 epochs with a learning rate of 2e-5 and a batch size of 16.
        "swag": {
            "learning_rate": [2e-5],
            "per_device_train_batch_size": [16],
            "num_train_epochs": [3],
        },
        # Appedix A. The optimal hyperparameter values are task-specific, but we found the following range of possible values to work well across all tasks:
        # - Batch size: 16, 32
        # - Learning rate (Adam): 5e-5, 3e-5, 2e-5
        # - Number of epochs: 2, 3, 4
        "other": {
            "learning_rate": [5e-5, 3e-5, 2e-5],
            "per_device_train_batch_size": [16, 32],
            "num_train_epochs": [2, 3, 4],
        }
    }
    return get_space_union_and_unique(search_space_common, search_space_unique, [dataset_name])

def get_roberta_space(model_size_type = None,
                      dataset_name = None,
                      subdataset_name = None):
    # RoBERTa: A Robustly Optimized BERT Pretraining Approach
    # https://arxiv.org/pdf/1907.11692.pdf
    search_space_common = {
        "warmup_ratio": [0.06],
    }
    search_space_unique = {
        # Table 10: Hyperparameters for finetuning RoBERTa-LARGE on RACE, SQuAD and GLUE.
        # We consider a limited hyperparameter
        # sweep for each task, with batch sizes ∈ {16, 32}
        # and learning rates ∈ {1e−5, 2e−5, 3e−5}, with a
        # linear warmup for the first 6% of steps followed by
        # a linear decay to 0.
        "glue": {
            "learning_rate": [1e-5, 2e-5, 3e-5],
            "per_device_train_batch_size": [16, 32],
            "weight_decay": [0.1],
            "num_train_epochs": [10],
        },
        "race":{
            "learning_rate": [1e-5],
            "per_device_train_batch_size": [16],
            "weight_decay": [0.1],
            "num_train_epochs": [4],
        },
        "squad": {
            "learning_rate": [1.5e-5],
            "per_device_train_batch_size": [48],
            "weight_decay": [0.01],
            "num_train_epochs": [2],
        }
    }
    return get_space_union_and_unique(search_space_common, search_space_unique, [dataset_name])

def get_electra_space(model_size_type = None,
                      dataset_name = None,
                      subdataset_name = None):
    """
        ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS
        https://arxiv.org/pdf/2003.10555.pdf
    """
    assert model_size_type in ("small", "base"), "Electra paper has only provided hyperparameter for the small and base huggingface"
    search_space_common = {
        "weight_decay": [0.0],
        "adam_epsilon": [1e-6],
        "warmup_ratio": [0.1],
        "per_device_train_batch_size": [32],
        "hidden_dropout_prob": [0.1],
        "attention_probs_dropout_prob": [0.1],
    }
    search_space_unique = {
        # Appendix B: For Basesized models we searched for a learning
        # rate out of [3e-5, 5e-5, 1e-4, 1.5e-4]
        "base": {
            "learning_rate": [3e-5, 5e-5, 1e-4, 1.5e-4],
        },
        # Appendix B: We found the small models benefit from a larger learning rate and searched for the best one
        # out of [1e-4, 2e-4, 3e-4, 5e-3]
        "small": {
            "learning_rate": [1e-4, 2e-4, 3e-4, 5e-3],
        },
        "squad": {
            "num_train_epochs": [2]
        },
        "squad_v2": {
            "num_train_epochs": [2]
        },
        "glue_stsb":{
            "num_train_epochs": [10],
        },
        "glue_rte": {
            "num_train_epochs": [10],
        },
        "glue_wnli": {
            "num_train_epochs": [3],
        },
        "glue_mrpc": {
            "num_train_epochs": [3],
        },
        "glue_cola": {
            "num_train_epochs": [3],
        },
        "glue_sst2": {
            "num_train_epochs": [3],
        },
        "glue_qnli": {
            "num_train_epochs": [3],
        },
        "glue_mnli": {
            "num_train_epochs": [3],
        },
        "glue_qqp": {
            "num_train_epochs": [3],
        }
    }
    from ..autotransformers import AutoTransformers
    return get_space_union_and_unique(search_space_common, search_space_unique,
        [AutoTransformers.get_full_data_name(dataset_name, subdataset_name), model_size_type])

def get_mobilebert_space(model_size_type = None,
                         dataset_name = None,
                         subdataset_name = None):
    """
        MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices
        https://arxiv.org/pdf/2004.02984.pdf
    """
    # To finetune the pre-trained models, we search the optimization hyperparameters
    # in a search space including different batch sizes (16/32/48), learning
    # rates ((1-10) * e-5), and the number of epochs (2-10)
    search_space_common = {
        "learning_rate": [x * 1e-5 for x in range(1, 11)],
        "per_device_train_batch_size": [4, 8, 16, 32, 48],
        "num_train_epochs": [x for x in range(2, 11)],
    }
    search_space_unique = {}
    return get_space_union_and_unique(search_space_common, search_space_unique, [])

def get_albert_space(model_size_type = None,
                         dataset_name = None,
                         subdataset_name = None):
    """
        Hyperparameters for downstream tasks are shown in Table 14. We adapt these hyperparameters
        from Liu et al. (2019), Devlin et al. (2019), and Yang et al. (2019).

        LR BSZ ALBERT DR Classifier DR TS WS MSL
        CoLA 1.00E-05 16 0 0.1 5336 320 512
        STS 2.00E-05 16 0 0.1 3598 214 512
        SST-2 1.00E-05 32 0 0.1 20935 1256 512
        MNLI 3.00E-05 128 0 0.1 10000 1000 512
        QNLI 1.00E-05 32 0 0.1 33112 1986 512
        QQP 5.00E-05 128 0.1 0.1 14000 1000 512
        RTE 3.00E-05 32 0.1 0.1 800 200 512
        MRPC 2.00E-05 32 0 0.1 800 200 512
        WNLI 2.00E-05 16 0.1 0.1 2000 250 512
        SQuAD v1.1 5.00E-05 48 0 0.1 3649 365 384
        SQuAD v2.0 3.00E-05 48 0 0.1 8144 814 512
        RACE 2.00E-05 32 0.1 0.1 12000 1000 512
    """
    search_space_dict = {}
    # To finetune the pre-trained models, we search the optimization hyperparameters
    # in a search space including different batch sizes (16/32/48), learning
    # rates ((1-10) * e-5), and the number of epochs (2-10)
    return  search_space_dict
