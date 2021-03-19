electra_glue_hpo = \
    {
    "learning_rate": {"l": 3e-5, "u": 1.5e-4, "space": "log"},
    "weight_decay": {"l": 0, "u": 0.3, "space": "linear"},
    "warmup_ratio": [0.1],
    "hidden_dropout_prob": [0.1],
    "attention_probs_dropout_prob": [0.1],
    "per_device_train_batch_size": [16, 32, 64, 128],
    "seed": [12, 22, 33, 42],
    "num_train_epochs": {"l": 1.00, "u": 9.00, "space": "log"},
    "max_grad_norm": [1.0]
}