'''Require: pip install torch transformers datasets flaml[blendsearch,ray]
'''
import sys
sys.path.insert(0, "../../")
from flaml.nlp.autohf import AutoHuggingFace

def test_electra(method='BlendSearch'):
    wandb_key = "f38cc048c956367de27eeb2749c23e6a94519ab8"

    autohf = AutoHuggingFace()

    preparedata_setting = {
        "dataset_config": {"task": "text-classification",
                            "dataset_name": ["glue"],
                            "subdataset_name": "rte"},
        "model_name": "google/electra-base-discriminator",
        "split_mode": "resplit",
        "output_path": "../../../data/",
        "max_seq_length": 128,
        "resplit_portion": {"train": (0.0, 0.8),
                                      "dev": (0.8, 0.9),
                                      "test": (0.9, 1.0)}
    }

    train_dataset, eval_dataset, test_dataset =\
        autohf.prepare_data(**preparedata_setting)

    autohf_settings = {"metric_name": "accuracy",
                       "mode_name": "max",
                       "resources_per_trial": {"gpu": 4, "cpu": 4},
                       "wandb_key": wandb_key,
                       "search_algo": method,
                       "num_samples": 4,
                       "time_budget": 7200,
                       "points_to_evaluate": [{
                           "num_train_epochs": 1,
                           "per_device_train_batch_size": 128, }]
                       }

    autohf.fit(train_dataset,
               eval_dataset,
               **autohf_settings,)

    predictions = autohf.predict(test_dataset)

    stop = 0

if __name__ == "__main__":
    test_electra()