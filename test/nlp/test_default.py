import sys
from pathlib import Path
from test.nlp.utils import get_automl_settings, get_toy_data_seqclassification

import pytest

from flaml.default import portfolio

try:
    import transformers

    _transformers_installed = True
except ImportError:
    _transformers_installed = False

_unsupported_training_runtime = (
    sys.platform.startswith("darwin")
    and sys.version_info >= (3, 11)
    or not _transformers_installed
    or sys.platform == "win32"
)

pytestmark = (
    pytest.mark.spark
)  # set to spark as parallel testing raised ValueError: Feature NonExisting not implemented.


def pop_args(fit_kwargs):
    fit_kwargs.pop("max_iter", None)
    fit_kwargs.pop("use_ray", None)
    fit_kwargs.pop("estimator_list", None)
    fit_kwargs.pop("time_budget", None)
    fit_kwargs.pop("log_file_name", None)


def test_build_portfolio(tmp_path, monkeypatch):
    source = Path(__file__).parent / "default"
    (tmp_path / "transformer_ms").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "portfolio.py",
            "--output",
            str(tmp_path),
            "--input",
            str(source),
            "--metafeatures",
            str(source / "all" / "metafeatures.csv"),
            "--task",
            "seq-classification",
            "--estimator",
            "transformer_ms",
            "--strategy",
            "greedy",
        ],
    )
    portfolio.main()
    assert (tmp_path / "transformer_ms" / "seq-classification.json").is_file()


def test_starting_point_not_in_search_space():
    """Regression test for invalid starting points and custom_hp.

    This test must not require network access to Hugging Face.
    """

    """
        test starting_points located outside of the search space, and custom_hp is not set
    """
    from flaml.automl.state import SearchState
    from flaml.automl.task.factory import task_factory

    this_estimator_name = "transformer"
    X_train, y_train, _, _, _ = get_toy_data_seqclassification()
    task = task_factory("seq-classification", X_train, y_train)
    estimator_class = task.estimator_class_from_str(this_estimator_name)
    estimator_class.init()

    # SearchState is where invalid starting points are filtered out when max_iter > 1.
    search_state = SearchState(
        learner_class=estimator_class,
        data=X_train,
        task=task,
        starting_point={"learning_rate": 2e-3},
        max_iter=3,
        budget=10,
    )
    assert search_state.init_config and search_state.init_config[0].get("learning_rate") != 2e-3

    """
        test starting_points located outside of the search space, and custom_hp is set
    """

    from flaml import tune

    X_train, y_train, _, _, _ = get_toy_data_seqclassification()

    this_estimator_name = "transformer_ms"
    task = task_factory("seq-classification", X_train, y_train)
    estimator_class = task.estimator_class_from_str(this_estimator_name)
    estimator_class.init()

    custom_hp = {
        "model_path": {
            "domain": "albert-base-v2",
        },
        "learning_rate": {
            "domain": tune.choice([1e-4, 1e-5]),
        },
        "per_device_train_batch_size": {
            "domain": 2,
        },
    }

    # Simulate a suggested starting point (e.g. from portfolio) which becomes invalid
    # after custom_hp constrains the space.
    invalid_starting_points = [
        {
            "learning_rate": 1e-5,
            "num_train_epochs": 1.0,
            "per_device_train_batch_size": 8,
            "seed": 43,
            "global_max_steps": 100,
            "model_path": "google/electra-base-discriminator",
        }
    ]

    search_state = SearchState(
        learner_class=estimator_class,
        data=X_train,
        task=task,
        starting_point=invalid_starting_points,
        custom_hp=custom_hp,
        max_iter=3,
        budget=10,
    )

    assert search_state.init_config, "Expected a non-empty init_config list"
    init_config0 = search_state.init_config[0]
    assert init_config0 is not None
    assert len(init_config0) == len(search_state._search_space_domain) - len(custom_hp), (
        "The search space is updated with the custom_hp on {} hyperparameters of "
        "the specified estimator without an initial value. Thus a valid init config "
        "should only contain the cardinality of the search space minus {}".format(
            len(custom_hp),
            len(custom_hp),
        )
    )
    assert search_state.search_space["model_path"] == "albert-base-v2"


@pytest.mark.skipif(_unsupported_training_runtime, reason="Online HF training is not supported on this runtime")
def test_points_to_evaluate():
    from flaml import AutoML

    X_train, y_train, X_val, y_val, _ = get_toy_data_seqclassification()

    automl = AutoML()
    automl_settings = get_automl_settings(estimator_name="transformer_ms")

    automl_settings["starting_points"] = "data:test/nlp/default/"

    automl_settings["custom_hp"] = {"transformer_ms": {"model_path": {"domain": "google/electra-small-discriminator"}}}

    try:
        automl.fit(X_train, y_train, **automl_settings)
    except OSError as e:
        message = str(e)
        if "Too Many Requests" in message or "rate limit" in message.lower():
            pytest.skip(f"Skipping HF model load/training: {message}")
        raise


# TODO: implement _test_zero_shot_model
@pytest.mark.skipif(_unsupported_training_runtime, reason="Online HF training is not supported on this runtime")
def test_zero_shot_nomodel():
    from flaml.default import preprocess_and_suggest_hyperparams

    estimator_name = "transformer_ms"

    location = "test/nlp/default"
    X_train, y_train, X_val, y_val, X_test = get_toy_data_seqclassification()

    automl_settings = get_automl_settings(estimator_name)

    (
        hyperparams,
        estimator_class,
        X_train,
        y_train,
        _,
        _,
    ) = preprocess_and_suggest_hyperparams("seq-classification", X_train, y_train, estimator_name, location=location)

    model = estimator_class(**hyperparams)  # estimator_class is TransformersEstimatorModelSelection

    fit_kwargs = automl_settings.pop("fit_kwargs_by_estimator", {}).get(estimator_name)
    fit_kwargs.update(automl_settings)
    pop_args(fit_kwargs)

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except OSError as e:
        message = str(e)
        if "Too Many Requests" in message or "rate limit" in message.lower():
            pytest.skip(f"Skipping HF model load/training: {message}")
        raise


def test_build_error_portfolio(tmp_path, monkeypatch):
    source = Path(__file__).parent / "default"
    (tmp_path / "transformer_ms").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "portfolio.py",
            "--output",
            str(tmp_path),
            "--input",
            str(source),
            "--metafeatures",
            str(source / "all" / "metafeatures_err.csv"),
            "--task",
            "seq-classification",
            "--estimator",
            "transformer_ms",
            "--strategy",
            "greedy",
        ],
    )
    portfolio.main()

    from flaml.default import preprocess_and_suggest_hyperparams

    estimator_name = "transformer_ms"

    location = str(tmp_path)
    X_train, y_train, X_val, y_val, X_test = get_toy_data_seqclassification()

    with pytest.raises(ValueError, match="not implemented"):
        (
            hyperparams,
            estimator_class,
            X_train,
            y_train,
            _,
            _,
        ) = preprocess_and_suggest_hyperparams(
            "seq-classification", X_train, y_train, estimator_name, location=location
        )
