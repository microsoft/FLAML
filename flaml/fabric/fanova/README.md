# Optuna fANOVA adapter

This folder contains FLAML's adapter around Optuna's pure-Python
`optuna.importance.FanovaImportanceEvaluator`.

`flaml.fabric.visualization.get_param_importance()` works with FLAML result objects, so
`evaluator.py` converts the collected hyperparameter DataFrame and score series into an
in-memory Optuna study before delegating to Optuna's public evaluator API.

No local Cython extension or `build_ext` step is required.
