from transformers import AutoModelForSequenceClassification
<<<<<<< HEAD
from flaml.nlp.autohf import AutoHuggingFace
from flaml.nlp.hpo_training_args import AutoHFArguments
=======

from flaml.nlp.hpo_trainer import HPOTrainer
from flaml.nlp.hpo_training_args import HPOTrainingArguments
>>>>>>> adding AutoHuggingFace
from flaml.nlp.utils import prepare_data
from flaml.nlp.utils import build_compute_metrics_fn

def _test_electra(method='bs'):
<<<<<<< HEAD
    wandb_key = "f38cc048c956367de27eeb2749c23e6a94519ab8"

    autohf = AutoHuggingFace()

    autohf.prepare_data(dataset_config = {"task": "text-classification",
                                          "dataset": ["glue"],
                                          "subdataset_name": "qnli"},
                        submit_mode = "resplit",
                        output_path= "/data/xliu127/projects/hyperopt/data/",
                        model_name = ["electra", "base"],
                        split_portion={"train": (0.0, 0.8),
                                      "dev": (0.8, 0.9),
                                      "test": (0.9, 1.0)})

    autohf.fit(hpo_method=method,
               num_samples = 1000,
               time_budget = 7200,
               device_nums= {"gpu": 4, "cpu": 4})
=======
    abs_data_path = "/data/xliu127/projects/hyperopt/data/"
    submit_mode = "resplit"
    wandb_key = "f38cc048c956367de27eeb2749c23e6a94519ab8"
    task_name = "qnli"

    train_dataset, val_dataset, test_dataset \
        = prepare_data(submit_mode,
                       task_name,
                       split_portion={"train": (0.0, 0.8),
                                      "dev": (0.8, 0.9),
                                      "test": (0.9, 1.0)})
    #TODO: test MNLI see if there's a conflict

    NUM_LABELS = len(train_dataset.features["label"].names)

    autohf = HPOTrainer(task_name,
                        abs_data_path,
                        model_name_short = "electra",
                        hpo_method = method,
                        scheduler_name = "asha",
                        submit_mode = submit_mode,
                        split_portion = {"train": (0.0, 0.8),
                                  "dev": (0.8, 0.9),
                                  "test": (0.9, 1.0)},
                        wandb_key = wandb_key)

    training_args = HPOTrainingArguments(
        task_name=task_name,
        model_name_short= "electra",

    )

    this_model = AutoModelForSequenceClassification.from_pretrained(, num_labels=NUM_LABELS)

    tune_trainer = HPOTrainer(
        model= this_model,
        args= training_args,
        train_dataset= train_dataset,
        eval_dataset= val_dataset,
        test_dataset= test_dataset,
        compute_metrics= build_compute_metrics_fn(config["task_name"]),
    )

    autohf.prepare_glue_data()

    autohf.hyperparameter_search(num_samples = 1000,
                                 time_budget = 7200,
                                 device_nums= {"gpu": 4, "cpu": 4})
>>>>>>> adding AutoHuggingFace

if __name__ == "__main__":
    _test_electra()