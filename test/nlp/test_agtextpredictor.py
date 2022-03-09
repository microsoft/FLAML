from flaml import AutoML
import pandas as pd
import requests
import sklearn
import numpy as np
import os
import json
from autogluon.text.text_prediction.infer_types import infer_column_problem_types
from sklearn.model_selection import train_test_split

from auto_mm_bench.datasets import dataset_registry, TEXT_BENCHMARK_ALIAS_MAPPING

def _get_benchmark_dataset(dataset_name):
        

        train_dataset = dataset_registry.create(dataset_name, 'train')
        test_dataset = dataset_registry.create(dataset_name, 'test')
        train_data = train_dataset.data
        test_data = test_dataset.data

        return  train_dataset, test_dataset, train_data, test_data

def default_holdout_frac(num_train_rows, hyperparameter_tune=False):
    """ Returns default holdout_frac used in fit().
        Between row count 5,000 and 25,000 keep 0.1 holdout_frac, as we want to grow validation set to a stable 2500 examples.
        Ref: https://github.com/awslabs/autogluon/blob/master/core/src/autogluon/core/utils/utils.py#L243
    """
    if num_train_rows < 5000:
        holdout_frac = max(0.1, min(0.2, 500.0 / num_train_rows))
    else:
        holdout_frac = max(0.01, min(0.1, 2500.0 / num_train_rows))

    if hyperparameter_tune:
        holdout_frac = min(0.2, holdout_frac * 2)  # We want to allocate more validation data for HPO to avoid overfitting

    return holdout_frac
    
def test_ag_text_predictor(dataset_name='imdb', seed=123):
    # default dataset = the smallest dataset in the benchmark
    # TODO: is the random seed also tunable?

    train_dataset, test_dataset, train_data, test_data = _get_benchmark_dataset(TEXT_BENCHMARK_ALIAS_MAPPING[dataset_name])
    feature_columns = train_dataset.feature_columns
    label_columns = train_dataset.label_columns
    metric = train_dataset.metric  # 'acc' or 'roc_auc' or 'r2', 
    # TODO: map acc to accuracy in flaml
    problem_type = train_dataset.problem_type  # 'binary' or 'multiclass' or 'regression' 
    # TODO: map multiclass to multi in flaml
    
    train_data1, tuning_data1 = sklearn.model_selection.train_test_split(
                                        train_data,
                                        test_size=0.05,
                                        random_state=np.random.RandomState(seed))

    column_types, inferred_problem_type = infer_column_problem_types(train_data1,
                                                                     tuning_data1,
                                                                     label_columns=label_columns,
                                                                     problem_type=problem_type)
    train_data = train_data[feature_columns + label_columns]
    
    # FORCE THE SAME TRAIN-VALID SPLIT IN & OUT THE PREDICTOR
    # ******* MODIFY HERE IF HIGHER HOLDOUT FRAC WANTED ******* #
    #  num_trials = hyperparameters['tune_kwargs']['num_trials']
    # if num_trials == 1:
    holdout_frac = default_holdout_frac(len(train_data), False)
            # else:
            # For HPO, we will need to use a larger held-out ratio
                # holdout_frac = default_holdout_frac(len(train_data), True)
    # ******* MODIFY ABOVE IF HIGHER HOLDOUT FRAC WANTED ******* #

    train_data_real, valid_data = train_test_split(train_data,
                                              test_size=holdout_frac,
                                              random_state=np.random.RandomState(seed))
    # the train_data_real will not be used since such split will redo in the predictor

    automl = AutoML()

    METRIC_MAPPING = {"acc": "accuracy", "roc_auc": "roc_auc", "r2": "r2"}
    TASK_MAPPING = {"binary": "binary", "multiclass": "multi", "regression": "regression" }

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 3,
        "time_budget": 5,
        "task": TASK_MAPPING[problem_type],  # TODO: change accordingly, 
        "metric": METRIC_MAPPING[metric],  # TODO: change accordingly
    }

    # the following will be add to estimator's self.fix_args
    automl_settings["custom_fix_args"] = {   
        "output_dir": "test/output/",
        # args for TextPredictor
        "text_backbone":"electra_base",
        "multimodal_fusion_strategy":"fuse_late", 
        "dataset_name": dataset_name, 
        "label_column": label_columns[0],
    }

         # ***** THE FOLLOWING ARE HPs TO TUNE IN THE ESTIMATOR'S SEARCH SPACE*****
        # 'model.backbone.name': 'google_electra_small',
        # 'model.network.agg_net.agg_type': 'concat',
        # 'model.network.aggregate_categorical': True,
        # 'model.use_avg_nbest': True,
        # 'optimization.batch_size': 128,
        # 'optimization.layerwise_lr_decay': 0.8,
        # 'optimization.lr': Categorical[0.0001],
        # 'optimization.nbest': 3,
        # 'optimization.num_train_epochs': 10,
        # 'optimization.per_device_batch_size': 8,
        # 'optimization.wd': 0.0001,
        # 'preprocessing.categorical.convert_to_text': False,
        # 'preprocessing.numerical.convert_to_text': False
    # DEBUG
    print("Before automl fit, train data shape: ", train_data.shape)
    print("The data pass to predictor as X_train and y_train:", train_data[feature_columns].shape, train_data[label_columns[0]].shape)
    print("FLAML's X_valid and y_valid: ", valid_data[feature_columns].shape, valid_data[label_columns[0]].shape)
    # END DEBUG
    try:
        automl.fit(
            # X_train=train_data[feature_columns],  # pass the whole train dataset to predictor
            # y_train=train_data[label_columns[0]],
            dataframe = train_data,
            label = label_columns[0],
            train_data=train_data,
            valid_data=valid_data,
            X_val=valid_data[feature_columns],  # TODO: add valid data in the autogluon way
            y_val=valid_data[label_columns[0]],
            estimator_list=["agtextpredictor"],  # TODO: add this choice to the automl.py
            **automl_settings
        )
    except requests.exceptions.HTTPError:
        return
    
    print('Begin to run inference on test set')
    save_dir = automl_settings["custom_fix_args"]["output_dir"]
    # predictions = automl.predict(test_data, as_pandas=True)
    # if problem_type == "multiclass" or problem_type == "binary":
    #     prediction_prob = automl.predict_proba(test_data, as_pandas=True)
    #     prediction_prob.to_csv(os.path.join(save_dir, 'test_prediction_prob.csv'))
    # predictions.to_csv(os.path.join(save_dir, 'test_prediction.csv'))
    # gt = test_data[label_columns[0]]
    # gt.to_csv(os.path.join(save_dir, 'ground_truth.csv'))
    
    score = automl.model.estimator.evaluate(test_data)
    with open(os.path.join(save_dir, 'test_score.json'), 'w') as of:
        json.dump({metric: score}, of)
    print(f"Inference on test set complete, {metric}: {score}")

# evaluation on test set should be implemented elsewhere


if __name__ == "__main__":
    test_ag_text_predictor(dataset_name='imdb')

