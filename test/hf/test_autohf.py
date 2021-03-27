'''Require: pip install torch transformers datasets flaml[blendsearch,ray]
'''
import ray

from flaml.nlp.autohf import AutoHuggingFace

def test_electra(method='BlendSearch'):
    # setting wandb key
    wandb_key = "f38cc048c956367de27eeb2749c23e6a94519ab8"

    autohf = AutoHuggingFace()

    preparedata_setting = {
        "dataset_config": {"task": "text-classification",
                           "dataset_name": ["glue"],
                           "subdataset_name": "rte"},
        "model_name": "google/mobilebert-uncased",
        "split_mode": "origin",
        "ckpt_path": "../../../data/checkpoint/",
        "result_path": "../../../data/result/",
        "log_path": "../../../data/result/",
        "max_seq_length": 128,
    }

    train_dataset, eval_dataset, test_dataset =\
        autohf.prepare_data(**preparedata_setting)

    autohf_settings = {"resources_per_trial": {"cpu": 1},
                       "wandb_key": wandb_key,
                       "search_algo_name": method,
                       "num_samples": 1,
                       "time_budget": 7200,
                       "fp16": False,
                       "search_algo_args_mode": "custom",
                       "points_to_evaluate": [{
                           "num_train_epochs": 0.05,
                           "per_device_train_batch_size": 1, }]
                       }

    autohf.fit(train_dataset,
               eval_dataset,
               **autohf_settings,)

    predictions = autohf.predict(test_dataset)

if __name__ == "__main__":
    test_electra()
