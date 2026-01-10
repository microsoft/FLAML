# Default - Flamlized Estimator

Flamlized estimators automatically use data-dependent default hyperparameter configurations for each estimator, offering a unique zero-shot AutoML capability, or "no tuning" AutoML.

## Flamlized LGBMRegressor

### Prerequisites

This example requires the \[autozero\] option.

```bash
pip install flaml[autozero] lightgbm openml
```

### Zero-shot AutoML

```python
from flaml.automl.data import load_openml_dataset
from flaml.default import LGBMRegressor
from flaml.automl.ml import sklearn_metric_loss_score

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=537, data_dir="./")
lgbm = LGBMRegressor()
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
print("flamlized lgbm r2", "=", 1 - sklearn_metric_loss_score("r2", y_pred, y_test))
print(lgbm)
```

#### Sample output

```
load dataset from ./openml_ds537.pkl
Dataset name: houses
X_train.shape: (15480, 8), y_train.shape: (15480,);
X_test.shape: (5160, 8), y_test.shape: (5160,)
flamlized lgbm r2 = 0.8537444671194614
LGBMRegressor(colsample_bytree=0.7019911744574896,
              learning_rate=0.022635758411078528, max_bin=511,
              min_child_samples=2, n_estimators=4797, num_leaves=122,
              reg_alpha=0.004252223402511765, reg_lambda=0.11288241427227624,
              verbose=-1)
```

### Suggest hyperparameters without training

```
from flaml.automl.data import load_openml_dataset
from flaml.default import LGBMRegressor
from flaml.ml import sklearn_metric_loss_score

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=537, data_dir="./")
lgbm = LGBMRegressor()
hyperparams, estimator_name, X_transformed, y_transformed = lgbm.suggest_hyperparams(X_train, y_train)
print(hyperparams)
```

#### Sample output

```
load dataset from ./openml_ds537.pkl
Dataset name: houses
X_train.shape: (15480, 8), y_train.shape: (15480,);
X_test.shape: (5160, 8), y_test.shape: (5160,)
{'n_estimators': 4797, 'num_leaves': 122, 'min_child_samples': 2, 'learning_rate': 0.022635758411078528, 'colsample_bytree': 0.7019911744574896, 'reg_alpha': 0.004252223402511765, 'reg_lambda': 0.11288241427227624, 'max_bin': 511, 'verbose': -1}
```

[Link to notebook](https://github.com/microsoft/FLAML/blob/main/notebook/zeroshot_lightgbm.ipynb) | [Open in colab](https://colab.research.google.com/github/microsoft/FLAML/blob/main/notebook/zeroshot_lightgbm.ipynb)

## Flamlized LGBMClassifier

### Prerequisites

This example requires the \[autozero\] option.

```bash
pip install flaml[autozero] lightgbm openml
```

### Zero-shot AutoML

```python
from flaml.automl.data import load_openml_dataset
from flaml.default import LGBMClassifier
from flaml.automl.ml import sklearn_metric_loss_score

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=1169, data_dir="./")
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
print(
    "flamlized lgbm accuracy",
    "=",
    1 - sklearn_metric_loss_score("accuracy", y_pred, y_test),
)
print(lgbm)
```

#### Sample output

```
load dataset from ./openml_ds1169.pkl
Dataset name: airlines
X_train.shape: (404537, 7), y_train.shape: (404537,);
X_test.shape: (134846, 7), y_test.shape: (134846,)
flamlized lgbm accuracy = 0.6745
LGBMClassifier(colsample_bytree=0.85, learning_rate=0.05, max_bin=255,
               min_child_samples=20, n_estimators=500, num_leaves=31,
               reg_alpha=0.01, reg_lambda=0.1, verbose=-1)
```

## Flamlized XGBRegressor

### Prerequisites

This example requires xgboost, sklearn, openml==0.10.2.

### Zero-shot AutoML

```python
from flaml.automl.data import load_openml_dataset
from flaml.default import XGBRegressor
from flaml.automl.ml import sklearn_metric_loss_score

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=537, data_dir="./")
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print("flamlized xgb r2", "=", 1 - sklearn_metric_loss_score("r2", y_pred, y_test))
print(xgb)
```

#### Sample output

```
load dataset from ./openml_ds537.pkl
Dataset name: houses
X_train.shape: (15480, 8), y_train.shape: (15480,);
X_test.shape: (5160, 8), y_test.shape: (5160,)
flamlized xgb r2 = 0.8542
XGBRegressor(colsample_bylevel=1, colsample_bytree=0.85, learning_rate=0.05,
             max_depth=6, n_estimators=500, reg_alpha=0.01, reg_lambda=1.0,
             subsample=0.9)
```

## Flamlized XGBClassifier

### Prerequisites

This example requires xgboost, sklearn, openml==0.10.2.

### Zero-shot AutoML

```python
from flaml.automl.data import load_openml_dataset
from flaml.default import XGBClassifier
from flaml.automl.ml import sklearn_metric_loss_score

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=1169, data_dir="./")
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print(
    "flamlized xgb accuracy",
    "=",
    1 - sklearn_metric_loss_score("accuracy", y_pred, y_test),
)
print(xgb)
```

#### Sample output

```
load dataset from ./openml_ds1169.pkl
Dataset name: airlines
X_train.shape: (404537, 7), y_train.shape: (404537,);
X_test.shape: (134846, 7), y_test.shape: (134846,)
flamlized xgb accuracy = 0.6729009388487608
XGBClassifier(base_score=0.5, booster='gbtree',
              colsample_bylevel=0.4601573737792679, colsample_bynode=1,
              colsample_bytree=1.0, gamma=0, gpu_id=-1, grow_policy='lossguide',
              importance_type='gain', interaction_constraints='',
              learning_rate=0.04039771837785377, max_delta_step=0, max_depth=0,
              max_leaves=159, min_child_weight=0.3396294979905001, missing=nan,
              monotone_constraints='()', n_estimators=540, n_jobs=4,
              num_parallel_tree=1, random_state=0,
              reg_alpha=0.0012362430984376035, reg_lambda=3.093428791531145,
              scale_pos_weight=1, subsample=1.0, tree_method='hist',
              use_label_encoder=False, validate_parameters=1, verbosity=0)
```

## Flamlized RandomForestRegressor

### Prerequisites

This example requires the \[autozero\] option.

```bash
pip install flaml[autozero] scikit-learn openml
```

### Zero-shot AutoML

```python
from flaml.automl.data import load_openml_dataset
from flaml.default import RandomForestRegressor
from flaml.automl.ml import sklearn_metric_loss_score

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=537, data_dir="./")
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("flamlized rf r2", "=", 1 - sklearn_metric_loss_score("r2", y_pred, y_test))
print(rf)
```

#### Sample output

```
load dataset from ./openml_ds537.pkl
Dataset name: houses
X_train.shape: (15480, 8), y_train.shape: (15480,);
X_test.shape: (5160, 8), y_test.shape: (5160,)
flamlized rf r2 = 0.8521
RandomForestRegressor(max_features=0.8, min_samples_leaf=2, min_samples_split=5,
                      n_estimators=500)
```

## Flamlized RandomForestClassifier

### Prerequisites

This example requires the \[autozero\] option.

```bash
pip install flaml[autozero] scikit-learn openml
```

### Zero-shot AutoML

```python
from flaml.automl.data import load_openml_dataset
from flaml.default import RandomForestClassifier
from flaml.automl.ml import sklearn_metric_loss_score

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=1169, data_dir="./")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(
    "flamlized rf accuracy",
    "=",
    1 - sklearn_metric_loss_score("accuracy", y_pred, y_test),
)
print(rf)
```

#### Sample output

```
load dataset from ./openml_ds1169.pkl
Dataset name: airlines
X_train.shape: (404537, 7), y_train.shape: (404537,);
X_test.shape: (134846, 7), y_test.shape: (134846,)
flamlized rf accuracy = 0.6701
RandomForestClassifier(max_features=0.7, min_samples_leaf=3, min_samples_split=5,
                       n_estimators=500)
```

## Flamlized ExtraTreesRegressor

### Prerequisites

This example requires the \[autozero\] option.

```bash
pip install flaml[autozero] scikit-learn openml
```

### Zero-shot AutoML

```python
from flaml.automl.data import load_openml_dataset
from flaml.default import ExtraTreesRegressor
from flaml.automl.ml import sklearn_metric_loss_score

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=537, data_dir="./")
et = ExtraTreesRegressor()
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
print("flamlized et r2", "=", 1 - sklearn_metric_loss_score("r2", y_pred, y_test))
print(et)
```

#### Sample output

```
load dataset from ./openml_ds537.pkl
Dataset name: houses
X_train.shape: (15480, 8), y_train.shape: (15480,);
X_test.shape: (5160, 8), y_test.shape: (5160,)
flamlized et r2 = 0.8534
ExtraTreesRegressor(max_features=0.75, min_samples_leaf=2, min_samples_split=5,
                    n_estimators=500)
```

## Flamlized ExtraTreesClassifier

### Prerequisites

This example requires the \[autozero\] option.

```bash
pip install flaml[autozero] scikit-learn openml
```

### Zero-shot AutoML

```python
from flaml.automl.data import load_openml_dataset
from flaml.default import ExtraTreesClassifier
from flaml.automl.ml import sklearn_metric_loss_score

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=1169, data_dir="./")
et = ExtraTreesClassifier()
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
print(
    "flamlized et accuracy",
    "=",
    1 - sklearn_metric_loss_score("accuracy", y_pred, y_test),
)
print(et)
```

#### Sample output

```
load dataset from ./openml_ds1169.pkl
Dataset name: airlines
X_train.shape: (404537, 7), y_train.shape: (404537,);
X_test.shape: (134846, 7), y_test.shape: (134846,)
flamlized et accuracy = 0.6698
ExtraTreesClassifier(max_features=0.7, min_samples_leaf=3, min_samples_split=5,
                     n_estimators=500)
```
