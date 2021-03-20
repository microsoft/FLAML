
# HPO methods included in ray tune

# [Nevergrad](https://docs.ray.io/en/master/tune/api_docs/suggestion.html#nevergrad-tune-suggest-nevergrad-nevergradsearch)

- Installation and two successful test exps
```
pip install nevergrad

python test/test_automl_exp.py  -t 600.0 -l lgbm  -d KDDCup09   -t 600.0 -total_pu 1 -trial_pu 1  -r 0 -m  Nevergrad

python test/test_automl_exp.py  -t 600.0 -l xgb_cat  -d KDDCup09   -t 600.0 -total_pu 1 -trial_pu 1  -r 0 -m  Nevergrad
```

## [Ax](https://docs.ray.io/en/master/tune/api_docs/suggestion.html#tune-ax)

- Installation and two successful test exps
```
pip install ax-platform sqlalchemy

python test/test_automl_exp.py  -t 600.0 -l lgbm -d KDDCup09  -m Ax -t 600.0 -total_pu 1 -trial_pu 1  -r 0 

python test/test_automl_exp.py  -t 600.0 -l xgb_cat -d KDDCup09  -m Ax -t 600.0 -total_pu 1 -trial_pu 1  -r 0 
```

## [Hyperopt](https://docs.ray.io/en/master/tune/api_docs/suggestion.html#hyperopt-tune-suggest-hyperopt-hyperoptsearch)
- Installation and two successful test exps

```
pip install hyperopt

python test/test_automl_exp.py  -t 600.0  -d KDDCup09  -t 600.0 -total_pu 1 -trial_pu 1  -r 0 -l lgbm -m HyperOpt

python test/test_automl_exp.py  -t 600.0 -l xgb_cat -d KDDCup09  -m HyperOpt -t 600.0 -total_pu 1 -trial_pu 1
```
## [Skopt](https://docs.ray.io/en/master/tune/api_docs/suggestion.html#scikit-optimize-tune-suggest-skopt-skoptsearch)
- Installation and two successful test exps
```
pip install scikit-optimize==0.8.1

python test/test_automl_exp.py  -t 600.0 -l xgb_cat -d KDDCup09  -m SkOpt -t 600.0 -total_pu 1 -trial_pu 1  -r 0

python test/test_automl_exp.py  -t 600.0 -l lgbm -d KDDCup09  -m SkOpt -t 600.0 -total_pu 1 -trial_pu 1  -r 0
```
- Warning

SkOpt search does not support quantization. Dropped quantization.


## [Dragonfly](https://docs.ray.io/en/master/tune/api_docs/suggestion.html#dragonfly-tune-suggest-dragonfly-dragonflysearch)

- installation
```
pip install dragonfly-opt
```

- Error

Dragonfly only support parameters of type `Float` (defined in ray.tune.sample)

- Warning
Does not suppert quantization and specific sampling methods (quantization and sampler will be dropped)



# Zoopt
- installation
```
pip install -U zoopt
```

- Error

ValueError: ZOOpt does not support parameters with samplers of type `Quantized`
