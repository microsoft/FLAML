#ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS
# https://arxiv.org/pdf/2003.10555.pdf
# TABLE 7 and Section Appendix.A

electra_glue_grid = {
    "learning_rate": [3e-5, 5e-5, 1e-4, 1.5e-4],
    "weight_decay": [0],
    "warmup_ratio": [0.1],
    "hidden_dropout_prob": [0.1],
    "attention_probs_dropout_prob": [0.1],
    "per_device_train_batch_size": [32],
    "seed": [42],
    "num_train_epochs": [3.05],
    "max_grad_norm": [1.0]
}