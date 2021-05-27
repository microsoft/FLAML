import numpy as np
from typing import Optional
import logging
from ..tune.trial import Trial
from ..tune.online_trial_runner import OnlineTrialRunner
from ..scheduler.online_scheduler import ChaChaScheduler
from ..searcher.online_searcher import ChampionFrontierSearcher
logger = logging.getLogger(__name__)


class AutoVW:
    """The AutoML class

    """
    WARMSTART_NUM = 100

    def __init__(self,
                 init_config: dict,
                 search_space: dict,
                 max_live_model_num: int,
                 min_resource_lease='auto',
                 automl_runner_args: dict = {},
                 scheduler_args: dict = {},
                 model_select_policy: str = 'threshold_loss_ucb',
                 metric='mae_clipped',
                 config_oracle_random_seed: Optional[int] = None,
                 model_selection_mode='min',
                 cb_coef: Optional[float] = None,
                 ):
        '''Constructor

        Args:
            init_config: A dictionary of a partial or full initial config,
                e.g. {'interactions': set(), 'learning_rate': 0.5}
            search_space: A dictionary of the search space. This search space includes both
                hyperparameters we want to tune and fixed hyperparameters. In the latter case,
                the value is a fixed value.
            max_live_model_num: The maximum number of 'live' models, which, in other words, is the 
                maximum number of models allowed to update in each learning iteraction.
            min_resource_lease: The minimum resource lease assigned to a particular model/trial. 
                If set as 'auto', it will be calculated automatically.
            automl_runner_args: A dictionary of configuration for the OnlineTrialRunner.
                If set {}, default values will be used.
            scheduler_args: A dictionary of configuration for the scheduler.
                If set {}, default values will be used.
            model_select_policy: A string to specify how to select one model to do prediction from the live model pool
            metric: A string to specify the name of the loss function used for calculating
                the progressive validation loss
            config_oracle_random_seed (int): An integer of the random seed used in ConfigOracle
            cb_coef (float): A float coefficient (optional) used in the sample complexity bound.
        '''
        self._max_live_model_num = max_live_model_num
        self._model_select_policy = model_select_policy
        self._model_selection_mode = model_selection_mode
        online_trial_args = {"metric": metric,
                             "min_resource_lease": min_resource_lease,
                             "cb_coef": cb_coef,
                             }
        # setup the arguments for searcher, which contains the ConfigOracle
        searcher_args = {"init_config": init_config,
                         "config_oracle_random_seed": config_oracle_random_seed,
                         'online_trial_args': online_trial_args,
                         'space': search_space,
                         }
        logger.info("search_space %s", search_space)
        searcher = ChampionFrontierSearcher(**searcher_args)
        scheduler = ChaChaScheduler(**scheduler_args)
        logger.info('scheduler_args %s', scheduler_args)
        logger.info('searcher_args %s', searcher_args)
        logger.info('automl_runner_args %s', automl_runner_args)
        self._trial_runner = OnlineTrialRunner(max_live_model_num=self._max_live_model_num,
                                               searcher=searcher,
                                               scheduler=scheduler,
                                               **automl_runner_args)
        self._best_trial = None
        # code for bebugging purpose
        self._prediction_trial_id = None
        self._iter = 0

    def predict(self, data_sample):
        """ Predict on the input example (e.g., vw example)
        """
        self._best_trial = self._best_trial_selection()
        self._y_predict = self._best_trial.predict(data_sample)
        # code for bebugging purpose
        if self._prediction_trial_id is None or \
           self._prediction_trial_id != self._best_trial.trial_id:
            self._prediction_trial_id = self._best_trial.trial_id
            logger.info('prediction trial id changed to %s at iter %s %s',
                        self._prediction_trial_id, self._iter, self._best_trial.result.resource_used)
        return self._y_predict

    def learn(self, data_sample):
        """Perform one online learning with the given data sample

        Args:
            data_sample (vw_example/str/list): one data sample on which the model gets updated
        """
        self._iter += 1
        self._trial_runner.step(self._max_live_model_num, data_sample, (self._y_predict, self._best_trial))

    def _best_trial_selection(self):
        best_score = float('+inf') if self._model_selection_mode == 'min' else float('-inf')
        new_best_trial = None
        running_trials = list(self._trial_runner.get_running_trials).copy()
        for trial in running_trials:
            if trial.result is not None and ('threshold' not in self._model_select_policy or \
               trial.result.resource_used >= AutoVW.WARMSTART_NUM):
                score = trial.result.get_score(self._model_select_policy)
                # logger.info('%s trial score %s', trial.trial_id, score)
                if ('min' == self._model_selection_mode and score < best_score) or \
                   ('max' == self._model_selection_mode and score > best_score):
                    best_score = score
                    new_best_trial = trial
        if new_best_trial is not None:
            logger.debug('best_trial._data_sample_size %s %s', new_best_trial._data_sample_size,
                         new_best_trial.result.resource_used)
            return new_best_trial
        else:
            if self._best_trial is not None and self._best_trial.status == Trial.RUNNING:
                logger.debug('old best trial%s ', self._best_trial.trial_id)
                return self._best_trial
            else:
                logger.debug('using champion trial: %s',
                             self._trial_runner.champion_trial.trial_id)
                return self._trial_runner.champion_trial
