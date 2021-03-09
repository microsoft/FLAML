from transformers import AutoModelForSequenceClassification
from flaml.nlp.autohf import AutoHuggingFace
from flaml.nlp.hpo_training_args import AutoHFArguments
from flaml.nlp.utils import prepare_data
from flaml.nlp.utils import build_compute_metrics_fn

def _test_electra(method='bs'):
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

if __name__ == "__main__":
    _test_electra()