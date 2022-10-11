from utils import (
    get_toy_data_regression,
    get_toy_data_binclassification,
    get_toy_data_multiclassclassification,
    get_automl_settings,
)


def test_switch_classificationhead():
    from flaml import AutoML
    import requests

    automl = AutoML()

    data_list = [
        "get_toy_data_regression",
        "get_toy_data_binclassification",
        "get_toy_data_multiclassclassification",
    ]
    model_path_list = [
        "textattack/bert-base-uncased-SST-2",
        "textattack/bert-base-uncased-STS-B",
        "textattack/bert-base-uncased-MNLI",
    ]

    for each_data in data_list:
        for each_model_path in model_path_list:
            X_train, y_train, X_val, y_val = globals()[each_data]()
            automl_settings = get_automl_settings()
            automl_settings["model_path"] = each_model_path

            if each_data == "get_toy_data_regression":
                automl_settings["task"] = "seq-regression"
                automl_settings["metric"] = "pearsonr"
            else:
                automl_settings["task"] = "seq-classification"
                automl_settings["metric"] = "accuracy"

            try:
                automl.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    **automl_settings
                )
            except requests.exceptions.HTTPError:
                return


if __name__ == "__main__":
    test_switch_classificationhead()
