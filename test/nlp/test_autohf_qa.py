def test_qa():
    from flaml import AutoML
    from datasets import load_dataset

    train_dataset = (
        load_dataset("squad", split="train[:1%]").to_pandas().iloc[0:10]
    )
    dev_dataset = (
        load_dataset("squad", split="train[1%:2%]").to_pandas().iloc[0:10]
    )

    custom_sent_keys = ['context', 'question']
    label_key = 'answers'

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key].apply(lambda x: str(x["answer_start"][0])+"###%d"%len(x["text"][0]))
    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key].apply(lambda x: str(x["answer_start"][0])+"###%d"%len(x["text"][0]))
    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 3,
        "time_budget": 10,
        "task": "question-answering",
        "metric": "accuracy",
        "log_file_name": "qa.log",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": 'google/electra-small-discriminator',
        "output_dir": "test/data/output/",
        "ckpt_per_epoch": 5,
        "fp16": False,
    }

    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        **automl_settings
    )

if __name__ == "__main__":
    test_qa()
