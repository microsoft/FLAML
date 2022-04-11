from utils import get_toy_data_seqclassification, get_automl_settings


def _test_points_to_evaluate():
    from flaml import AutoML

    X_train, y_train, X_val, y_val = get_toy_data_seqclassification()

    automl = AutoML()
    automl_settings = get_automl_settings()
    automl_settings["starting_points"] = "data"

    automl.fit(X_train, y_train, **automl_settings)


def _test_zero_shot_model(model_path):
    from flaml.default import preprocess_and_suggest_hyperparams

    X_train, y_train, X_val, y_val, X_test = get_toy_data_seqclassification()

    hyperparams, estimator_class, _, _, _, _ = preprocess_and_suggest_hyperparams(
        "seq-classification_" + model_path, X_train, y_train, "transformer"
    )
    trainer = estimator_class(**hyperparams)  # estimator_class is Trainer
    trainer.train()  # LGBMClassifier can handle raw labels


def _test_zero_shot_nomodel():
    from flaml.default import preprocess_and_suggest_hyperparams

    X_train, y_train, X_val, y_val, X_test = get_toy_data_seqclassification()

    hyperparams, estimator_class, _, _, _, _ = preprocess_and_suggest_hyperparams(
        "seq-classification", X_train, y_train, "transformer_ms"
    )

    model = estimator_class(**hyperparams)  # estimator_class is Trainer
    model.fit(X_train, y_train)  # LGBMClassifier can handle raw labels
