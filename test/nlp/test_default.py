from utils import get_toy_data_seqclassification, get_automl_settings


def test_points_to_evaluate():
    from flaml import AutoML

    X_train, y_train, X_val, y_val, _ = get_toy_data_seqclassification()

    automl = AutoML()
    automl_settings = get_automl_settings(estimator_name="transformer_ms")
    automl_settings["time_budget"] = 100
    automl_settings["gpu_per_trial"] = 1

    automl_settings["estimator_list"] = ["transformer_ms"]
    automl_settings["starting_points"] = "data"

    del automl_settings["custom_fit_kwargs"]["transformer_ms"]["model_path"]

    automl.fit(X_train, y_train, **automl_settings)


# TODO: implement _test_zero_shot_model
def test_zero_shot_nomodel():
    from flaml.default import preprocess_and_suggest_hyperparams

    location = "test/nlp/default"
    X_train, y_train, X_val, y_val, X_test = get_toy_data_seqclassification()

    automl_settings = get_automl_settings()
    del automl_settings["custom_fit_kwargs"]["transformer"]["model_path"]

    hyperparams, estimator_class, _, _, _, _ = preprocess_and_suggest_hyperparams(
        "seq-classification", X_train, y_train, "transformer_ms", location=location
    )

    model = estimator_class(
        **hyperparams
    )  # estimator_class is TransformersEstimatorModelSelection

    model.fit(X_train, y_train, **automl_settings)


if __name__ == "__main__":
    test_points_to_evaluate()
