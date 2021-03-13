import sys
sys.path.insert(0, "../../")
from flaml.nlp.autohf import AutoHuggingFace

def _test_electra(method='BlendSearch'):
    wandb_key = "f38cc048c956367de27eeb2749c23e6a94519ab8"

    autohf = AutoHuggingFace()

    train_dataset, eval_dataset, test_dataset =\
        autohf.prepare_data(dataset_config = {"task": "text-classification",
                                          "dataset_name": ["glue"],
                                          "subdataset_name": "rte"},
                        model_name = "google/electra-base-discriminator",
                        submit_mode="resplit",
                        output_path="/data/xliu127/projects/hyperopt/data/",
                        max_seq_length=128,
                        split_portion={"train": (0.0, 0.8),
                                      "dev": (0.8, 0.9),
                                      "test": (0.9, 1.0)})

    search_algo_args = {"points_to_evaluate": [{
                         "num_train_epochs": 1,
                         "per_device_train_batch_size": 128,}]}

    autohf.fit(train_dataset,
               eval_dataset,
               metric_name = "accuracy",
               mode_name = "max",
               resources_per_trial={"gpu": 4, "cpu": 4},
               wandb_key = wandb_key,
               search_algo= method,
               num_samples = 4,
               time_budget = 7200,
               **search_algo_args,)

    predictions = autohf.predict(test_dataset)

    stop = 0

if __name__ == "__main__":
    _test_electra()