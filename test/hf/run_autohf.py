'''Require: pip install torch transformers datasets flaml[blendsearch,ray]
'''
from flaml.nlp.autohf import AutoHuggingFace

def _test_electra(method='BlendSearch'):
    # setting wandb key
    wandb_key = "f38cc048c956367de27eeb2749c23e6a94519ab8"

    autohf = AutoHuggingFace()

    preparedata_setting = {
        "dataset_config": {"task": "text-classification",
                            "dataset_name": ["glue"],
                            "subdataset_name": "qnli"},
        "model_name": "roberta-base",
        "split_mode": "origin",
        "output_path": "../../../data/",
        "max_seq_length": 128,
    }

    train_dataset, eval_dataset, test_dataset =\
        autohf.prepare_data(**preparedata_setting)

    search_algo = "CFO"

    autohf_settings = {"metric_name": "accuracy",
                       "mode_name": "max",
                       "resources_per_trial": {"gpu": 4, "cpu": 4},
                       "wandb_key": wandb_key,
                       "search_algo": search_algo,
                       "num_samples": 1000 if search_algo != "grid_search" else 1,
                       "time_budget": 3600,
                       "fp16": True,
                       "points_to_evaluate": [{
                           "num_train_epochs": 1,
                           "per_device_train_batch_size": 48, }]
                       }

    autohf.fit(train_dataset,
               eval_dataset,
               **autohf_settings,)

    predictions = autohf.predict(test_dataset)

if __name__ == "__main__":
    _test_electra()
