# lookup table for the grid configs in each pre-trained language huggingface for different tasks

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
    search_space_dict = {}
    search_space_dict["learning_rate"] =  [1e-5, 2e-5, 3e-5]
    search_space_dict["hidden_dropout"] = [0.1]
    search_space_dict["activation_dropout"] = [0.0]
    search_space_dict["attention_dropout"] = [0.1]
    search_space_dict["weight_decay"] = [0.01]
    search_space_dict["warmup_ratio"] = [0.1]
    search_space_dict["adam_epsilon"] = [1e-6]
    if dataset_name == "yelp_review_full":
        search_space_dict["per_device_train_batch_size"] = [64]
        search_space_dict["num_train_epochs"] = [3]
    return search_space_dict

def get_bert_space(model_size_type = None,
                   dataset_name = None,
                   subdataset_name = None):
    """
        BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        https://arxiv.org/pdf/1810.04805.pdf
    """
    search_space_dict = {}
    # Section 4.1: We use a batch size of 32 and fine-tune for 3 epochs over the data for all GLUE tasks. For each
    # task, we selected the best fine-tuning learning rate (among 5e-5, 4e-5, 3e-5, and 2e-5) on the Dev set
    if dataset_name == "glue":
        search_space_dict["learning_rate"] = [5e-5, 4e-5, 3e-5, 2e-5]
        search_space_dict["per_device_train_batch_size"] = [32]
        search_space_dict["num_train_epochs"] = [3]
    # Section 4.2: We fine-tune for 3 epochs with a learning rate of 5e-5 and a batch size of 32
    elif dataset_name == "squad":
        search_space_dict["learning_rate"] = [5e-5]
        search_space_dict["per_device_train_batch_size"] = [32]
        search_space_dict["num_train_epochs"] = [3]
    # Section 4.3: We fine-tuned for 2 epochs with a learning rate of 5e-5 and a batch size of 48.
    elif dataset_name == "squad_v2":
        search_space_dict["learning_rate"] = [5e-5]
        search_space_dict["per_device_train_batch_size"] = [48]
        search_space_dict["num_train_epochs"] = [2]
    # Section 4.4: We fine-tune the huggingface for 3 epochs with a learning rate of 2e-5 and a batch size of 16.
    elif dataset_name == "swag":
        search_space_dict["learning_rate"] = [2e-5]
        search_space_dict["per_device_train_batch_size"] = [16]
        search_space_dict["num_train_epochs"] = [3]
    # Appedix A. The optimal hyperparameter values are task-specific, but we found the following range of possible values to work well across all tasks:
    # - Batch size: 16, 32
    # - Learning rate (Adam): 5e-5, 3e-5, 2e-5
    # - Number of epochs: 2, 3, 4
    else:
        search_space_dict["learning_rate"] = [5e-5, 3e-5, 2e-5]
        search_space_dict["per_device_train_batch_size"] = [16, 32]
        search_space_dict["num_train_epochs"] = [2, 3, 4]
    return search_space_dict

def get_roberta_space(model_size_type = None,
                      dataset_name = None,
                      subdataset_name = None):
    # RoBERTa: A Robustly Optimized BERT Pretraining Approach
    # https://arxiv.org/pdf/1907.11692.pdf
    search_space_dict = {}
    # Table 10: Hyperparameters for finetuning RoBERTa-LARGE on RACE, SQuAD and GLUE.
    # We consider a limited hyperparameter
    # sweep for each task, with batch sizes ∈ {16, 32}
    # and learning rates ∈ {1e−5, 2e−5, 3e−5}, with a
    # linear warmup for the first 6% of steps followed by
    # a linear decay to 0.
    if dataset_name == "glue":
        search_space_dict["learning_rate"] = [1e-5, 2e-5, 3e-5]
        search_space_dict["per_device_train_batch_size"] = [16, 32]
        search_space_dict["weight_decay"] = [0.1]
        search_space_dict["num_train_epochs"] = [10]
        search_space_dict["warmup_ratio"] = [0.06]
    elif dataset_name == "race":
        search_space_dict["learning_rate"] = [1e-5]
        search_space_dict["per_device_train_batch_size"] = [16]
        search_space_dict["weight_decay"] = [0.1]
        search_space_dict["num_train_epochs"] = [4]
        search_space_dict["warmup_ratio"] = [0.06]
    elif dataset_name == "squad":
        search_space_dict["learning_rate"] = [1.5e-5]
        search_space_dict["per_device_train_batch_size"] = [48]
        search_space_dict["weight_decay"] = [0.01]
        search_space_dict["num_train_epochs"] = [2]
        search_space_dict["warmup_ratio"] = [0.06]
    return search_space_dict

def get_electra_space(model_size_type = None,
                      dataset_name = None,
                      subdataset_name = None):
    """
        ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS
        https://arxiv.org/pdf/2003.10555.pdf
    """
    assert model_size_type in ("small", "base"), "Electra paper has only provided hyperparameter for the small and base huggingface"
    search_space_dict = {}
    # Appendix B: For Basesized models we searched for a learning
    # rate out of [3e-5, 5e-5, 1e-4, 1.5e-4]
    if model_size_type == "base":
        search_space_dict["learning_rate"] = [3e-5, 5e-5, 1e-4, 1.5e-4]
    # Appendix B: We found the small models benefit from a larger learning rate and searched for the best one
    # out of [1e-4, 2e-4, 3e-4, 5e-3]
    elif model_size_type == "small":
        search_space_dict["learning_rate"] = [1e-4, 2e-4, 3e-4, 5e-3]
    search_space_dict["adam_epsilon"] = [1e-6]
    search_space_dict["warmup_ratio"] = [0.1]
    search_space_dict["per_device_train_batch_size"] = [32]
    if dataset_name == "squad" or dataset_name == "squad_v2":
        search_space_dict["num_train_epochs"] = [2]
    elif dataset_name == "glue" and subdataset_name and (subdataset_name == "stsb" or subdataset_name == "rte"):
        search_space_dict["num_train_epochs"] = [10]
    else:
        search_space_dict["num_train_epochs"] = [3]
    return search_space_dict

def get_mobilebert_space(model_size_type = None,
                         dataset_name = None,
                         subdataset_name = None):
    """
        MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices
        https://arxiv.org/pdf/2004.02984.pdf
    """
    search_space_dict = {}
    # To finetune the pre-trained models, we search the optimization hyperparameters
    # in a search space including different batch sizes (16/32/48), learning
    # rates ((1-10) * e-5), and the number of epochs (2-10)
    search_space_dict["learning_rate"] = [x * 1e-5 for x in range(1, 11)]
    search_space_dict["per_device_train_batch_size"] = [4, 8, 16, 32, 48]
    search_space_dict["num_train_epochs"] = [x for x in range(2, 11)]
    return  search_space_dict

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
