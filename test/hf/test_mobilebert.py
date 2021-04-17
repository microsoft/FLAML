'''Require: pip install torch transformers datasets wandb flaml[blendsearch,ray]
'''
import subprocess, platform
from flaml.nlp.autotransformers import AutoTransformers


def test_electra(method='BlendSearch'):
    # setting wandb key
    if not 'windows' in platform.system().lower():
        wandb_key = "f38cc048c956367de27eeb2749c23e6a94519ab8"
        subprocess.run(["wandb", "login", "--relogin", wandb_key])
        autohf = AutoTransformers()

        preparedata_setting = {
            "dataset_config": {"task": "text-classification",
                               "dataset_name": ["glue"],
                               "subdataset_name": "rte"},
            "model_name": "google/mobilebert-uncased",
            "server_name": "tmdev",
            "split_mode": "resplit",
            "resplit_portion": {
                "source": ["train", "validation"],
                "train": [0, 0.01],
                "validation": [0.01, 0.02],
                "test": [0.02, 0.03]},
            "ckpt_path": "data/checkpoint/",
            "result_path": "data/result/",
            "log_path": "data/result/",
            "max_seq_length": 128,
        }

        train_dataset, eval_dataset, test_dataset = autohf.prepare_data(
            **preparedata_setting)

        autohf_settings = {"resources_per_trial": {"cpu": 1},
                           "wandb_key": wandb_key,
                           "search_algo_name": method,
                           "custom_num_samples": 1,
                           "custom_time_budget": 7200,
                           "fp16": False,
                           "search_algo_args_mode": "custom",
                           "points_to_evaluate": [{
                               "num_train_epochs": 1,
                               "per_device_train_batch_size": 4, }]
                          }

        autohf.fit(train_dataset,
                   eval_dataset,
                   **autohf_settings)

        #predictions = autohf.predict(test_dataset)


if __name__ == "__main__":
    test_electra()
