from flaml.nlp.autohf import AutoHuggingFace

def _test_electra(method='bs'):
    wandb_key = "f38cc048c956367de27eeb2749c23e6a94519ab8"

    autohf = AutoHuggingFace()

    train_dataset, eval_dataset, test_dataset =\
        autohf.prepare_data(dataset_config = {"task": "text-classification",
                                          "dataset": ["glue"],
                                          "subdataset_name": "qnli"},
                        submit_mode = "resplit",
                        output_path= "/data/xliu127/projects/hyperopt/data/",
                        model_name = "google/electra-base-discriminator",
                        split_portion={"train": (0.0, 0.8),
                                      "dev": (0.8, 0.9),
                                      "test": (0.9, 1.0)})

    autohf.fit(train_dataset,
               eval_dataset,
               test_dataset,
               metric_name = "acc",
               mode_name = "max",
               wandb_key = wandb_key,
               hpo_method = method,
               num_samples = 1000,
               time_budget = 7200,
               device_nums= {"gpu": 4, "cpu": 4})

if __name__ == "__main__":
    _test_electra()