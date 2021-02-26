'''Require: pip install flaml[blendsearch,ray]
pip install tensorflow deeptables[gpu]
'''
import time
import numpy as np
from flaml import tune
import pandas as pd
from functools import partial

DT_METRIC, MODE = 'log_loss', 'min'

DT_loss_metric = 'categorical_crossentropy'
try:
    import ray
    import flaml
    from flaml import BlendSearch
    from deeptables.models.deeptable import DeepTable, ModelConfig
    from deeptables.models.deepnets import DCN, WideDeep, DeepFM
except:
    print("pip install deeptables flaml[blendsearch,ray]")
    
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('test/tune_dt_para.log'))
logger.setLevel(logging.INFO)


def construct_dt_modelconfig(config:dict, y_train, objective_name):#->ModelConfig:
    # basic config of dt
    dropout = config.get('dropout', 0)
    learning_rate = config.get('learning_rate', 0.001)
    batch_norm = config.get('batch_norm', True)
    auto_discrete = config.get('auto_discrete', False)
    apply_gbm_features = config.get('apply_gbm_features', False)
    fixed_embedding_dim = config.get('fixed_embedding_dim', True)
    if not fixed_embedding_dim: embeddings_output_dim = 0
    else: embeddings_output_dim = 4
    stacking_op = config.get('stacking_op', 'add')

    if 'binary' in objective_name:
        # nets = DCN
        metrics, monitor = ['AUC'], 'val_auc'
    elif 'multi' in objective_name:
        # nets = WideDeep  
        metrics, monitor = [
            'categorical_crossentropy'], 'val_categorical_crossentropy'
    else:
        metrics, monitor = ['r2'], 'val_r2'
    l1, l2 = 256, 128 #128, 64
    max_width = 2096
    if 'regression' != objective_name:
        n_classes = len(np.unique(y_train))
        base_size = max(1, min(n_classes, 100)/50)
        l1 = min(l1*base_size, max_width)
        l2 = min(l2*base_size, max_width)
    dnn_params = {'hidden_units': ((l1, dropout, batch_norm), 
        (l2, dropout, batch_norm)), 'dnn_activation': 'relu'}
    net = config.get('nets', 'DCN')
    if net == 'DCN':
        nets = DCN
    elif net == 'WideDeep':
        nets = WideDeep
    elif net == 'DeepFM':
        nets = DeepFM
    elif net == 'dnn_nets':
        nets = [net]
    else: nets = net
    from tensorflow.keras.optimizers import Adam
    assert 'rounds' in config, 'rounds required'
    assert 'dense_dropout' in config, 'dense_dropout required' 
    dt_model_config = ModelConfig(nets=nets, earlystopping_patience=config[
                "rounds"], dense_dropout=config["dense_dropout"], 
                auto_discrete=auto_discrete, stacking_op=stacking_op,
                apply_gbm_features=apply_gbm_features,
                fixed_embedding_dim=fixed_embedding_dim,
                embeddings_output_dim=embeddings_output_dim,
                dnn_params=dnn_params,
                optimizer=Adam(learning_rate=learning_rate, clipvalue=100),
                metrics=metrics, monitor_metric=monitor)

    return dt_model_config

def get_test_loss(estimator = None, model=None, X_test = None, y_test = None, 
                            metric = 'r2', labels = None):
        from sklearn.metrics import mean_squared_error, r2_score, \
            roc_auc_score, accuracy_score, mean_absolute_error, log_loss
        if not estimator:
            loss = np.Inf
        else:
            if 'roc_auc' == metric:
                y_pred = estimator.predict_proba(X_test = X_test)
                if y_pred.ndim>1 and y_pred.shape[1]>1:
                    y_pred = y_pred[:,1]
                loss = 1.0 - roc_auc_score(y_test, y_pred)
            elif 'log_loss' == metric or 'categorical' in metric:
                print('yes',estimator )
                att_func = getattr(estimator, "predict_proba", None)
                if callable(att_func): print('yes')
                y_pred = estimator.predict_proba(X_test)
                loss = log_loss(y_test, y_pred, labels=labels)
            elif 'r2' == metric:
                y_pred = estimator.predict(X_test)
                loss = 1.0-r2_score(y_test, y_pred)
        return loss

def train_dt(config: dict, prune_attr: str, resource_schedule: list):
    """ implement the traininig function of dt model
        reference: blendsearch.problem.DeepTables
    """

    def preprocess(X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[str(x) for x in list(range(
                X.shape[1]))])
        return X

    oml_dataset = "shuttle"
    # get a multi-class dataset
    from sklearn.model_selection import train_test_split
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml(name=oml_dataset, return_X_y=True)
        logger.info(f"dataset={oml_dataset}")
    except:
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)
        logger.info("failed to fetch oml dataset, dataset=wine")
    objective_name = 'muti'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
        random_state=42)
    # setup deeptable learner
    from deeptables.models.deeptable import DeepTable, ModelConfig
    from deeptables.models.deepnets import DCN, WideDeep, DeepFM

    # clip 'rounds' according to the size of the data
    data_size = len(y_train)
    assert 'rounds' in config, 'rounds required'
    clipped_rounds = max(min(round(1500000/data_size),config['rounds']), 10)
    config['rounds'] = clipped_rounds
    dt_model_config = construct_dt_modelconfig(config, y_train, objective_name='multi')

    dt = DeepTable(dt_model_config)
    #TODO: check the fit API
    for epo in resource_schedule:
        log_batchsize = config.get('log_batchsize', 9)
        # train model 
        dt_model, _ = dt.fit(preprocess(X_train), y_train, verbose=0,
                epochs=int(round(epo)), batch_size=1<<log_batchsize)
        
        # evaluate model
        loss = get_test_loss(dt, dt_model, X_test, y_test, metric=DT_loss_metric)

        # report loss
        tune.report(prune_attr = epo, score=loss)
        # TODO: for schedulers: option1: write a loop; option2: check whether there is a callback function in deeptable.

#TODO: should we worry about the following error: the global search method raises error. Ignoring for this iteration.
def _test_dt_parallel(method='BlendSearch'):
    try:
        import ray
    except ImportError:
        return
    if 'BlendSearch' in method:
        from flaml import tune
    else:
        from ray import tune
    # specify the search space
    # get the search space from the AutoML class: AutoML.search_space.
    #TODO: need to check how to set data_size
    data_size = 100
    # early_stopping_rounds = max(min(round(1500000/data_size),150), 10)
    search_space = {
        # 'log_epochs': tune.loguniform(1, 10),
        # 'rounds': tune.lograndint(1,150), 
        'rounds': tune.randint(1,150), 
        'net': tune.choice(['DCN', 'dnn_nets']),
        "learning_rate": tune.loguniform(1e-4, 3e-2),
        'auto_discrete': tune.choice([False, True]),
        'apply_gbm_features': tune.choice([False, True]),
        'fixed_embedding_dim': tune.choice([False, True]),
        'dropout': tune.uniform(0,0.5),
        'dense_dropout': tune.uniform(0,0.5),
        "log_batchsize": tune.randint(4, 10),        
    }

    init_config = {
        # 'log_epochs':1,
        'rounds': 10,
        "learning_rate": 3e-4, #FIXME: how to set random init
        'net': 'DCN',
        'auto_discrete': False,
        'apply_gbm_features': False,
        'fixed_embedding_dim': False,
         # 'auto_discrete': None,
        # 'apply_gbm_features': None,
        # 'fixed_embedding_dim': None,
        'dropout': 0.1,
        'dense_dropout': 0,
        "log_batchsize": 9,  
        }
    default_epochs = 2**9
    max_gpu = 1
    # specify prune attribute
    for num_samples in [2]:
        time_budget_s = 120 #None
        for n_gpu in [max_gpu]:
            start_time = time.time()
            ray.init(num_cpus=1, num_gpus=n_gpu)
            if 'BlendSearch' in method:
                prune_attr='epochs'
                resource_schedule = [default_epochs]
                if 'ASHA' in method:
                    prune_attr='epochs-ASHA'
                    resource_schedule=  [2**i for i in range(1,5)]
                # the default search_alg is BlendSearch in flaml 
                # corresponding schedulers for BS are specified in flaml.tune.run
                analysis = tune.run(
                    partial(train_dt, prune_attr=prune_attr, resource_schedule=resource_schedule),
                    init_config = init_config,
                    # cat_hp_cost={
                    #     "net": [2,1], #TODO: check the cat_hp_cost
                    # },
                    metric='score', 
                    mode=MODE,
                    prune_attr=prune_attr,
                    max_resource=resource_schedule[-1],
                    min_resource=resource_schedule[0],
                    report_intermediate_result=True,
                    # You can add "gpu": 0.1 to allocate GPUs
                    resources_per_trial={"gpu": n_gpu},
                    config=search_space, 
                    local_dir='logs/',
                    num_samples=num_samples*n_gpu, 
                    time_budget_s=time_budget_s,
                    use_ray=True)
            else: 
                algo=None
                scheduler = None
                if 'ASHA' == method:
                    algo=None 
                elif 'BOHB' == method:
                    from ray.tune.schedulers import HyperBandForBOHB
                    from ray.tune.suggest.bohb import TuneBOHB
                    algo = TuneBOHB(max_concurrent=n_cpu)
                    scheduler = HyperBandForBOHB(max_t=max_iter)
                elif 'Optuna' == method:
                    from ray.tune.suggest.optuna import OptunaSearch
                    algo = OptunaSearch()
                elif 'CFO' == method:
                    from flaml import CFO
                    #TODO: revise points to eval
                    # algo = CFO(points_to_evaluate=[{
                    #     "max_depth": 1,
                    #     "min_child_weight": 3,
                    # }], cat_hp_cost={
                    #     "min_child_weight": [6, 3, 2],
                    # })
                    algo = None

                #TODO: check points_to_evaluate and init_config
                
                analysis = tune.run(
                    partial(train_dt, resource_schedule=resource_schedule),
                    metric='score', 
                    mode=MODE,
                    init_config = init_config,
                    # You can add "gpu": 0.1 to allocate GPUs
                    resources_per_trial={"gpu": n_gpu},
                    config=search_space, local_dir='logs/',
                    num_samples=num_samples*n_gpu, time_budget_s=time_budget_s,
                    # scheduler=scheduler, 
                    search_alg=algo)
            ray.shutdown()
    
            best_trial = analysis.get_best_trial('score',MODE,"all")
            # accuracy = 1. - best_trial.metric_analysis["eval-error"]["min"]
            logloss = best_trial.metric_analysis['score'][MODE]
            logger.info(f"method={method}")
            logger.info(f"n_samples={num_samples*n_gpu}")
            logger.info(f"time={time.time()-start_time}")
            logger.info(f"Best model eval loss: {logloss:.4f}")
            logger.info(f"Best model parameters: {best_trial.config}")

# def _test_distillbert_cfo():
#     _test_distillbert('CFO')


# def _test_distillbert_dragonfly():
#     _test_distillbert('Dragonfly')


# def _test_distillbert_skopt():
#     _test_distillbert('SkOpt')


# def _test_distillbert_nevergrad():
#     _test_distillbert('Nevergrad')


# def _test_distillbert_zoopt():
#     _test_distillbert('ZOOpt')


# def _test_distillbert_ax():
#     _test_distillbert('Ax')


# def __test_distillbert_hyperopt():
#     _test_distillbert('HyperOpt')


# def _test_distillbert_optuna():
#     _test_distillbert('Optuna')


# def _test_distillbert_asha():
#     _test_distillbert('ASHA')


# def _test_distillbert_bohb():
#     _test_distillbert('BOHB')


if __name__ == "__main__":
    _test_dt_parallel(method='BlendSearch-ASHA')