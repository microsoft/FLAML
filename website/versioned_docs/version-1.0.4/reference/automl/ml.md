---
sidebar_label: ml
title: automl.ml
---

#### sklearn\_metric\_loss\_score

```python
def sklearn_metric_loss_score(metric_name, y_predict, y_true, labels=None, sample_weight=None, groups=None)
```

Loss using the specified metric.

**Arguments**:

- `metric_name` - A string of the metric name, one of
  'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'roc_auc_ovr',
  'roc_auc_ovo', 'roc_auc_weighted', 'roc_auc_ovo_weighted', 'roc_auc_ovr_weighted',
  'log_loss', 'mape', 'f1', 'ap', 'ndcg', 'micro_f1', 'macro_f1'.
- `y_predict` - A 1d or 2d numpy array of the predictions which can be
  used to calculate the metric. E.g., 2d for log_loss and 1d
  for others.
- `y_true` - A 1d numpy array of the true labels.
- `labels` - A list or an array of the unique labels.
- `sample_weight` - A 1d numpy array of the sample weight.
- `groups` - A 1d numpy array of the group labels.
  

**Returns**:

- `score` - A float number of the loss, the lower the better.

#### norm\_confusion\_matrix

```python
def norm_confusion_matrix(y_true, y_pred)
```

normalized confusion matrix.

**Arguments**:

- `estimator` - A multi-class classification estimator.
- `y_true` - A numpy array or a pandas series of true labels.
- `y_pred` - A numpy array or a pandas series of predicted labels.
  

**Returns**:

  A normalized confusion matrix.

#### multi\_class\_curves

```python
def multi_class_curves(y_true, y_pred_proba, curve_func)
```

Binarize the data for multi-class tasks and produce ROC or precision-recall curves.

**Arguments**:

- `y_true` - A numpy array or a pandas series of true labels.
- `y_pred_proba` - A numpy array or a pandas dataframe of predicted probabilites.
- `curve_func` - A function to produce a curve (e.g., roc_curve or precision_recall_curve).
  

**Returns**:

  A tuple of two dictionaries with the same set of keys (class indices).
  The first dictionary curve_x stores the x coordinates of each curve, e.g.,
  curve_x[0] is an 1D array of the x coordinates of class 0.
  The second dictionary curve_y stores the y coordinates of each curve, e.g.,
  curve_y[0] is an 1D array of the y coordinates of class 0.

