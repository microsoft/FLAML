from flaml.data import load_openml_dataset
from flaml.ml import ExtraTreesEstimator
from flaml import AutoML

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=1169, data_dir='./')
X_train = X_train.iloc[:1000]
y_train = y_train.iloc[:1000]


class ExtraTreesEstimatorSeeded(ExtraTreesEstimator):
    def __init__(self, task='binary', **config):
        'ExtraTreesEstimator for reproducible FLAML run'
        super().__init__(task, **config)
        self.params.update({'random_state': 0})


settings = {
    "time_budget": 1e10,   # total running time in seconds
    "max_iter": 3,   
    "metric": 'ap',        # average_precision
    "task": 'classification',  # task type
    "seed": 7654321,    # random seed
    "estimator_list": ['extra_trees_seeded'],
    "verbose": False
}

for trial_num in range(8):
    automl = AutoML()
    automl.add_learner(learner_name='extra_trees_seeded', learner_class=ExtraTreesEstimatorSeeded)
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    print(automl.best_loss)
    print(automl.best_config)