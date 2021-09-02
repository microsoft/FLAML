import numpy as np
from flaml.searcher.suggestion import OptunaSearch, Searcher, ConcurrencyLimiter
from flaml.tune import sample


def define_search_space(trial):
    trial.suggest_float("a", 6, 8)
    trial.suggest_float("b", 1e-4, 1e-2, log=True)


def test_searcher():
    searcher = Searcher()
    searcher = Searcher(metric=['m1', 'm2'], mode=['max', 'min'])
    searcher.set_search_properties(None, None, None)
    searcher.suggest = searcher.on_pause = searcher.on_unpause = lambda _: {}
    searcher.on_trial_complete = lambda trial_id, result, error: None
    searcher = ConcurrencyLimiter(searcher, max_concurrent=2, batch=True)
    searcher.suggest("t1")
    searcher.suggest("t2")
    searcher.on_pause("t1")
    searcher.on_unpause("t1")
    searcher.suggest("t3")
    searcher.on_trial_complete("t1", {})
    searcher.on_trial_complete("t2", {})
    searcher.set_state({})
    print(searcher.get_state())
    import optuna
    config = {
        "a": optuna.distributions.UniformDistribution(6, 8),
        "b": optuna.distributions.LogUniformDistribution(1e-4, 1e-2),
    }
    searcher = OptunaSearch(
        config, points_to_evaluate=[{"a": 6, "b": 1e-3}],
        evaluated_rewards=[{'m': 2}], metric='m', mode='max'
    )
    config = {
        "a": sample.uniform(6, 8),
        "b": sample.loguniform(1e-4, 1e-2)
    }
    searcher = OptunaSearch(
        config, points_to_evaluate=[{"a": 6, "b": 1e-3}],
        evaluated_rewards=[{'m': 2}], metric='m', mode='max'
    )
    searcher = OptunaSearch(
        define_search_space, points_to_evaluate=[{"a": 6, "b": 1e-3}],
        # evaluated_rewards=[{'m': 2}], metric='m', mode='max'
        mode='max'
    )
    searcher = OptunaSearch()
    # searcher.set_search_properties('m', 'min', define_search_space)
    searcher.set_search_properties('m', 'min', config)
    searcher.suggest('t1')
    searcher.on_trial_complete('t1', None, False)
    searcher.suggest('t2')
    searcher.on_trial_complete('t2', None, True)
    searcher.suggest('t3')
    searcher.on_trial_complete('t3', {'m': np.nan})
    searcher.save('test/tune/optuna.pickle')
    searcher.restore('test/tune/optuna.pickle')
