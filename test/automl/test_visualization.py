from flaml import AutoML
from flaml.data import load_openml_dataset


def test_fi_lc():
    X_train, X_test, y_train, y_test = load_openml_dataset(
        dataset_id=1169, data_dir="./"
    )
    settings = {
        "time_budget": 10,  # total running time in seconds
        "metric": "accuracy",  # can be: 'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'roc_auc_ovr',
        # 'roc_auc_ovo', 'log_loss', 'mape', 'f1', 'ap', 'ndcg', 'micro_f1', 'macro_f1'
        "task": "classification",  # task type
        "log_file_name": "airlines_experiment.log",  # flaml log file
        "seed": 7654321,  # random seed
    }
    automl = AutoML(**settings)
    automl.fit(X_train=X_train, y_train=y_train)
    automl.visualize(type="feature_importance", plot_filename="feature_importance")
    automl.visualize(type="learning_curve", plot_filename="learning_curve")


if __name__ == "__main__":
    test_fi_lc()
