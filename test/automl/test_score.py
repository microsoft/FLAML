from flaml import AutoML
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml
)

class TestClassification:

    def test_forecast(self, budget=5):
        # using dataframe
        import statsmodels.api as sm

        data = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean()
        data = (
            data.fillna(data.bfill())
                .to_frame()
                .reset_index()
                .rename(columns={"index": "ds", "co2": "y"})
        )
        num_samples = data.shape[0]
        time_horizon = 12
        split_idx = num_samples - time_horizon
        X_test = data[split_idx:]["ds"]
        y_test = data[split_idx:]["y"]

        df = data[:split_idx]
        automl = AutoML()
        settings = {
            "time_budget": budget,  # total running time in seconds
            "metric": "mape",  # primary metric
            "task": "ts_forecast",  # task type
            "log_file_name": "test/CO2_forecast.log",  # flaml log file
            "eval_method": "holdout",
            "label": "y",
        }
        """The main flaml automl API"""
        try:
            import prophet

            automl.fit(dataframe=df,
                       estimator_list=["prophet", "arima", "sarimax"],
                       **settings,
                       period=time_horizon)
            automl.score(X_test, y_test)
        except ImportError:
            print("not using prophet due to ImportError")
            automl.fit(
                dataframe=df,
                **settings,
                estimator_list=["arima", "sarimax"],
                period=time_horizon,
            )

            try:
                automl.score(X_test, y_test)
            except NotImplementedError:
                pass

    def test_classification(self):
        X = pd.DataFrame(
            {
                "f1": [1, -2, 3, -4, 5, -6, -7, 8, -9, -10, -11, -12, -13, -14],
                "f2": [
                    3.0,
                    16.0,
                    10.0,
                    12.0,
                    3.0,
                    14.0,
                    11.0,
                    12.0,
                    5.0,
                    14.0,
                    20.0,
                    16.0,
                    15.0,
                    11.0,
                ],
                "f3": [
                    "a",
                    "b",
                    "a",
                    "c",
                    "c",
                    "b",
                    "b",
                    "b",
                    "b",
                    "a",
                    "b",
                    1.0,
                    1.0,
                    "a",
                ],
                "f4": [
                    True,
                    True,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                ],
            }
        )
        y = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

        automl = AutoML()
        for each_estimator in ["catboost", "lrl2", "lrl1", "rf", "lgbm", "extra_tree", "kneighbor", "xgboost"]:
            automl_settings = {
                "time_budget": 6,
                "task": "classification",
                "n_jobs": 1,
                "estimator_list": [each_estimator],
                "metric": "accuracy",
                "log_training_metric": True,
            }
        automl.fit(X, y, **automl_settings)

        try:
            automl.score(X, y)
        except NotImplementedError:
            pass

    def test_regression(self):
        automl_experiment = AutoML()

        X_train, y_train = fetch_california_housing(return_X_y=True)
        n = int(len(y_train) * 9 // 10)

        for each_estimator in ["lgbm", "xgboost", "rf", "extra_tree", "catboost", "kneighbor"]:
            automl_settings = {
                "time_budget": 2,
                "task": "regression",
                "log_file_name": "test/california.log",
                "log_training_metric": True,
                "estimator_list": [each_estimator],
                "n_jobs": 1,
                "model_history": True,
            }
            automl_experiment.fit(
                X_train=X_train[:n],
                y_train=y_train[:n],
                X_val=X_train[n:],
                y_val=y_train[n:],
                **automl_settings
            )

            try:
                automl_experiment.score(X_train[n:], y_train[n:])
            except NotImplementedError:
                pass

    def _test_rank(self):
        from sklearn.externals._arff import ArffException

        dataset = "credit-g"

        try:
            X, y = fetch_openml(name=dataset, return_X_y=True)
            y = y.cat.codes
        except (ArffException, ValueError):
            from sklearn.datasets import load_wine

            X, y = load_wine(return_X_y=True)

        import numpy as np

        automl = AutoML()
        n = 500

        for each_estimator in ["lgbm", "xgboost"]:
            automl_settings = {
                "time_budget": 2,
                "task": "rank",
                "log_file_name": "test/{}.log".format(dataset),
                "model_history": True,
                "groups": np.array(  # group labels
                    [0] * 200 + [1] * 200 + [2] * 100
                ),
                "learner_selector": "roundrobin",
                "estimator_list": [each_estimator],
            }
            automl.fit(X[:n], y[:n], **automl_settings)
            try:
                automl.score(X[n:], y[n:])
            except NotImplementedError:
                pass

    def test_transformers(self):
        train_data = {
            "sentence1": [
                'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
                "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .",
                "They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added .",
                "Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a record high of A $ 4.57 .",
            ],
            "sentence2": [
                'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
                "Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 .",
                "On June 10 , the ship 's owners had published an advertisement on the Internet , offering the explosives for sale .",
                "Tab shares jumped 20 cents , or 4.6 % , to set a record closing high at A $ 4.57 .",
            ],
            "label": [1, 0, 1, 0],
            "idx": [0, 1, 2, 3],
        }
        train_dataset = pd.DataFrame(train_data)

        dev_data = {
            "sentence1": [
                "The stock rose $ 2.11 , or about 11 percent , to close Friday at $ 21.51 on the New York Stock Exchange .",
                "Revenue in the first quarter of the year dropped 15 percent from the same period a year earlier .",
                "The Nasdaq had a weekly gain of 17.27 , or 1.2 percent , closing at 1,520.15 on Friday .",
                "The DVD-CCA then appealed to the state Supreme Court .",
            ],
            "sentence2": [
                "PG & E Corp. shares jumped $ 1.63 or 8 percent to $ 21.03 on the New York Stock Exchange on Friday .",
                "With the scandal hanging over Stewart 's company , revenue the first quarter of the year dropped 15 percent from the same period a year earlier .",
                "The tech-laced Nasdaq Composite .IXIC rallied 30.46 points , or 2.04 percent , to 1,520.15 .",
                "The DVD CCA appealed that decision to the U.S. Supreme Court .",
            ],
            "label": [1, 1, 0, 1],
            "idx": [4, 5, 6, 7],
        }
        dev_dataset = pd.DataFrame(dev_data)

        custom_sent_keys = ["sentence1", "sentence2"]
        label_key = "label"

        X_train = train_dataset[custom_sent_keys]
        y_train = train_dataset[label_key]

        X_val = dev_dataset[custom_sent_keys]
        y_val = dev_dataset[label_key]

        automl = AutoML()

        automl_settings = {
            "gpu_per_trial": 0,
            "max_iter": 3,
            "time_budget": 10,
            "task": "seq-classification",
            "metric": "accuracy",
            "log_file_name": "seqclass.log",
            "use_ray": False,
        }

        automl_settings["hf_args"] = {
            "model_path": "google/electra-small-discriminator",
            "output_dir": "test/data/output/",
            "ckpt_per_epoch": 5,
            "fp16": False,
        }
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            **automl_settings
        )

        try:
            automl.score(X_val, y_val)
        except NotImplementedError:
            pass

if __name__ == "__main__":
    test = TestClassification()
    test.test_transformers()