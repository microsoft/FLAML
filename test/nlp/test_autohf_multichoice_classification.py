import os
import shutil
import sys

import pytest
from utils import get_automl_settings, get_toy_data_multiplechoiceclassification

try:
    import transformers

    _transformers_installed = True
except ImportError:
    _transformers_installed = False


@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"] or not _transformers_installed,
    reason="do not run on mac os or windows or transformers not installed",
)
def test_mcc():
    import requests

    from flaml import AutoML

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = get_toy_data_multiplechoiceclassification()
    automl = AutoML()

    automl_settings = get_automl_settings()
    automl_settings["task"] = "multichoice-classification"
    automl_settings["metric"] = "accuracy"

    try:
        automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **automl_settings)
    except requests.exceptions.HTTPError:
        return

    y_pred = automl.predict(X_test)
    proba = automl.predict_proba(X_test)
    print(str(len(automl.classes_)) + " classes")
    print(y_pred)
    print(y_test)
    print(proba)
    true_count = 0
    for i, v in y_test.items():
        if y_pred[i] == v:
            true_count += 1
    accuracy = round(true_count / len(y_pred), 5)
    print("Accuracy: " + str(accuracy))

    if os.path.exists("test/data/output/"):
        try:
            shutil.rmtree("test/data/output/")
        except PermissionError:
            print("PermissionError when deleting test/data/output/")


if __name__ == "__main__":
    test_mcc()
