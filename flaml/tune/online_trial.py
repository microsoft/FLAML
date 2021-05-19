import numpy as np
import logging
import time
import math
import copy
import collections
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from vowpalwabbit import pyvw
from .trial import Trial
logger = logging.getLogger(__name__)


class OnlineResult:
    """ Class for managing the result statistics of a trial
    """
    prob_delta = 0.1
    LOSS_MIN = 0.0
    LOSS_MAX = np.inf
    CB_COEF = 0.05  # 0.001 for mse

    def __init__(self, result_trial_id, result_type_name, cb_coef=None, init_loss=0.0,
                 init_cb=100.0, mode='min', sliding_window_size=100):
        self._result_trial_id = result_trial_id
        self._result_type_name = result_type_name  # for example 'mse' or 'mae'
        self._mode = mode
        self._init_loss = init_loss
        # statistics needed for alg
        self.loss_sum = 0.0
        self.observation_count = 0
        self.resource_used = 0.0
        self.loss_cb = init_cb  # a large number #TODO: this can be changed
        self._cb_coef = cb_coef if cb_coef is not None else self.CB_COEF
        # optional statistics
        self._sliding_window_size = sliding_window_size
        self._loss_queue = collections.deque(maxlen=self._sliding_window_size)

    def update_result(self, new_loss, new_resource_used, data_dimension,
                      bound_of_range=1.0, new_observation_count=1):
        self.observation_count += new_observation_count
        self.resource_used += new_resource_used
        self.loss_sum += new_loss
        self.loss_cb = self._update_loss_cb(bound_of_range, data_dimension)
        self._loss_queue.append(new_loss)

    def _update_loss_cb(self, bound_of_range, data_dim,
                        bound_name='sample_complexity_bound'):
        """Calculate bound coef
        """
        if bound_name == 'sample_complexity_bound':
            # set the coefficient in the loss bound
            if 'mae' in self.result_type_name:
                coef = self._cb_coef * bound_of_range
            else:
                coef = 0.001 * bound_of_range

            comp_F = math.sqrt(data_dim)
            n = self.observation_count
            return coef * comp_F * math.sqrt((np.log10(n / OnlineResult.prob_delta)) / n)
        else:
            raise NotImplementedError

    @property
    def result_trial_id(self):
        return self.result_trial_id

    @property
    def result_type_name(self):
        return self._result_type_name

    @property
    def loss_avg(self):
        return self.loss_sum / self.observation_count if \
            self.observation_count != 0 else self._init_loss

    @property
    def loss_lcb(self):
        return max(self.loss_avg - self.loss_cb, OnlineResult.LOSS_MIN)

    @property
    def loss_ucb(self):
        return min(self.loss_avg + self.loss_cb, OnlineResult.LOSS_MAX)

    @property
    def loss_avg_recent(self):
        return sum(self._loss_queue) / len(self._loss_queue) \
            if len(self._loss_queue) != 0 else self._init_loss

    def get_score(self, score_name, cb_ratio=1):
        if 'lcb' in score_name:
            return max(self.loss_avg - cb_ratio * self.loss_cb, OnlineResult.LOSS_MIN)
        elif 'ucb' in score_name:
            return min(self.loss_avg + cb_ratio * self.loss_cb, OnlineResult.LOSS_MAX)
        elif 'avg' in score_name:
            return self.loss_avg
        else:
            raise NotImplementedError


class BaseOnlineTrial(Trial):
    """A class for online trial. Important information in this Trial class:
    id of a trial: config

    Other information to keep track of:
    1. model  (and information needed to construct this model)
        Info needed:
        1.1 fixed_config
        
    2. result (not just a single value, but a collection of statistics. 
    Realized by the OnlineResult class)
        Info needed:
            1.2 namespace dimension dictionary. It is used to calculate the dimension
    """

    model_class = None

    def __init__(self,
                 config: dict = None,
                 min_resource_lease: float = None,
                 is_champion: bool = False,
                 is_checked_under_current_champion: bool = True,  # assuming the trial created is champion frontier by default
                 custom_trial_name='mae',
                 trial_id: Optional[str] = None,
                 ):
        # #======== basic variables
        self.config = config
        # self.trial_id = Trial.generate_id() if trial_id is None else trial_id
        self.trial_id = trial_id
        self.status = Trial.PENDING
        self.start_time = time.time()
        self.custom_trial_name = custom_trial_name
        # self.resources = Resources(cpu=1, gpu=0)

        # #==resource budget related variable
        self._min_resource_lease = min_resource_lease if min_resource_lease else 100.0
        self._resource_lease = copy.copy(self._min_resource_lease)
        # #======== champion related variables
        self._is_champion = is_champion
        # self._is_checked_under_current_champion_ is supposed to be always 1 when the trial is first created
        self._is_checked_under_current_champion = is_checked_under_current_champion

    @property
    def is_champion(self):
        return self._is_champion

    @property
    def is_checked_under_current_champion(self):
        return self._is_checked_under_current_champion

    @property
    def resource_lease(self):
        return self._resource_lease

    def set_champion_status(self, is_champion: bool):
        self._is_champion = is_champion

    def set_checked_under_current_champion(self, checked_under_current_champion: bool):
        """TODO: add documentation why this is needed. Brifly speacking, it is needed because we want to 
        know whether a trial has been paused since a new champion is promoted.
        We want to try to pause those running trials (even though they are not yet achieve the next scheduling
        check point according to resource used and resource lease), because a better trial is likely to be 
        in the new challengers generated by the new champion, so we want to try them as soon as possible.
        If we wait until we reach the next scheduling point, we may waste a lot of resource (depending
        on what is the current resource lease) on the old trials (note that new trials is not possible to be scheduled 
        to run until there is a slot openning). 

        Intuitively speaking, we want to squize an opening slot as soon as possible once a new champion is promoted,
        such that we are able to try newly generated challengers.
        """
        self._is_checked_under_current_champion = checked_under_current_champion

    def set_resource_lease(self, resource: float):
        self._resource_lease = resource

    def reset_resource_lease(self):
        self._resource_lease = self._min_resource_lease

    def set_status(self, status):
        """Sets the status of the trial and record the start time
        """
        self.status = status
        if status == Trial.RUNNING:
            if self.start_time is None:
                self.start_time = time.time()


class VWOnlineTrial(BaseOnlineTrial):
    """ Implement BaseOnlineTrial for VW
    Args:
        config (set): the config of the trial (note that the config is a set because the hyperparameters are )
        trial_id (str): id of the trial (if None, it will be generated in the constructor)
        is_champion (bool): indicates whether the trial is the current champion or not
        is_champion_frontier (bool): indicates whether the trial is the current champion frontier or not
    
    #NOTE about result: 
        1. training related results (need to be updated in the trainable class)
        2. result about resources lease (need to be updated externally)

    #NOTE about namespaces in vw:
        - Wiki in vw: 
        https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Namespaces
        - Namespace vs features: 
        https://stackoverflow.com/questions/28586225/in-vowpal-wabbit-what-is-the-difference-between-a-namespace-and-feature
        - Command-line options related to name-spaces:
            Option	Meaning
            --keep c	Keep a name-space staring with the character c
            --ignore c	Ignore a name-space starting with the character c
            --redefine a:=b	redefine namespace starting with b as starting with a
            --quadratic ab	Cross namespaces starting with a & b on the fly to 
                generate 2-way interacting features
            --cubic abc	Cross namespaces starting with a, b, & c on the fly to 
                generate 3-way interacting features
            --interactions {ab, cd, cde} Specify a set of namepace interactions
            
    Instance variables:
        config: set
        result: dict =
            {
            'resource_used': float,
            'data_sample_count': int,
            'loss_sum': float,
            'loss_avg': float, 
            'cb': float,
            'loss_ucb': float, 
            'loss_lcb': float,
            }
        [the other instance variables should be self-explainable]
    """
    model_class = pyvw.vw
    cost_unit = 1.0
    interactions_config_key = 'interactions'

    def __init__(self,
                 config: dict,
                 trainable_func,
                 metric: str = 'mae',
                 min_resource_lease: float = None,
                 is_champion: bool = False,
                 is_champion_frontier: bool = True,  # assuming the trial created is champion frontier by default
                 custom_trial_name: str = 'vw_mae_clipped',
                 trial_id: Optional[str] = None,
                 cb_coef: Optional[float] = None,
                 ):
        self.trial_id = self._config_to_id(config) if trial_id is None else trial_id
        logger.info('Create trial with trial_id: %s', self.trial_id)
        super().__init__(config, min_resource_lease, is_champion, is_champion_frontier,
                         custom_trial_name, self.trial_id)
        # variables that are needed during online training
        self._metric = metric
        self._y_min_observed = None
        self._y_max_observed = None
        self.model = None   # model is None until the config is scheduled to run
        self.result = None  # OnlineResult(self.trial_id, 'mae') #FIXME: need 'mae' info
        # application dependent info
        self._dim = None
        self._namespace_set = None
        self.trainable_func = trainable_func
        self.cb_coef = cb_coef

    @staticmethod
    def _config_to_id(config):
        """Generate an id for the provided config
        """
        # sort config keys
        sorted_k_list = sorted(list(config.keys()))
        config_id_full = ''
        for key in sorted_k_list:
            v = config[key]
            if isinstance(v, set):
                config_id = ''
                key_list = sorted(v)
                for k in key_list:
                    config_id = config_id + '_' + str(k)
            else:
                config_id = str(v)
            config_id_full = config_id_full + config_id
        return config_id_full

    def initialize_vw_model(self, vw_example):
        """Initialize a vw model using the trainable_fuc
        """
        # get the dimensionality of the feature according to the namespace configuration
        self._vw_config_tuned = self.config.copy()
        # ensure the feature interaction config is a list (required by VW)
        if VWOnlineTrial.interactions_config_key in self._vw_config_tuned.keys():
            self._vw_config_tuned[VWOnlineTrial.interactions_config_key] \
                = list(self._vw_config_tuned[VWOnlineTrial.interactions_config_key])
        self._namespace_set = self.config[VWOnlineTrial.interactions_config_key]
        namespace_feature_dim = self.get_ns_feature_dim_from_vw_example(vw_example)
        self._dim = self._get_dim_from_ns(namespace_feature_dim, self._namespace_set)
        # construct an instance of vw model using the input config and fixed config
        self.model = self.trainable_func(**self._vw_config_tuned)
        self.result = OnlineResult(self.trial_id, self._metric,
                                   cb_coef=self.cb_coef,
                                   init_loss=0.0, init_cb=100.0,)
        self._data_sample_size = 0

    def train_eval_model_online(self, data_sample, y_pred):
        """Train and eval model online
        """
        y = self._get_y_from_vw_example(data_sample)
        self._update_y_range(y)
        if self.model is None:
            # initialize self.model and self.result
            self.initialize_vw_model(data_sample)
        # do one step of learning
        self.model.learn(data_sample)
        # update training related results accordingly
        new_loss = self._get_loss(y, y_pred, self._metric,
                                  self._y_min_observed, self._y_max_observed)
        # udpate sample size, sum of loss, and cost
        data_sample_size = 1
        bound_of_range = self._y_max_observed - self._y_min_observed
        if bound_of_range == 0:
            bound_of_range = 1.0
        self.result.update_result(new_loss,
                                  VWOnlineTrial.cost_unit * data_sample_size,
                                  self._dim, bound_of_range)
        self._data_sample_size += data_sample_size

    def predict(self, x):
        """Predict using the model
        """
        if self.model is None:
            # initialize self.model and self.result
            self.initialize_vw_model(x)
        return self.model.predict(x)

    @property
    def get_result(self) -> OnlineResult:
        return self.result

    def _get_loss(self, y_true, y_pred, loss_func_name, y_min_observed, y_max_observed):
        """Get instantaneous loss from y_true and y_pred, and loss_func_name
            For mae_clip, we clip y_pred in the observed range of y
        """
        if 'mse' in loss_func_name or 'squared' in loss_func_name:
            loss_func = mean_squared_error
        elif 'mae' in loss_func_name or 'absolute' in loss_func_name:
            loss_func = mean_absolute_error
            if y_min_observed is not None and y_max_observed is not None and \
               'clip' in loss_func_name:
                # clip y_pred in the observed range of y
                y_pred = min(y_max_observed, max(y_pred, y_min_observed))
        else:
            raise NotImplementedError
        return loss_func([y_true], [y_pred])

    def _update_y_range(self, y):
        """Maintain running observed minimum and maximum target value
        """
        if self._y_min_observed is None or y < self._y_min_observed:
            self._y_min_observed = y
        if self._y_max_observed is None or y > self._y_max_observed:
            self._y_max_observed = y

    @staticmethod
    def _get_dim_from_ns(namespace_feature_dim: dict, namespace_set: set):
        """Get the dimensionality of the corresponding feature of input namespace set
        """
        total_dim = 0
        for f in namespace_set:
            ns_dim = 1.0
            for c in f:
                ns_dim *= namespace_feature_dim[c]
            total_dim += ns_dim
        total_dim += sum(namespace_feature_dim.values())
        return total_dim

    def _config2vwconfig(self, config: dict):
        """Convert the config into vw config format

        The 'interaction' hyperparamter in vw only takes namespace interaction, so we 
        need to strip the single namespace from the config, which includes both single and
        namespace interactions.
        #FIXME: maybe we should consider removing the single namespace from the config, which 
        requires revision of the ConfigOracle. Then this function is no longer needed.
        """
        assert isinstance(config, dict) or isinstance(config, set)
        if isinstance(config, dict):
            assert VWOnlineTrial.interactions_config_key in config.keys()
            namespace_set = config[VWOnlineTrial.interactions_config_key]
            vw_config = config.copy()
        else:
            namespace_set = config
            vw_config = {}
        namespace_interaction_config = [c for c in namespace_set if len(c) != 1]
        vw_config[VWOnlineTrial.interactions_config_key] = namespace_interaction_config
        print('namespace_set', config, namespace_set)
        return vw_config, namespace_set

    def clean_up_model(self):
        self.model = None
        self.result = None

    @staticmethod
    def _get_y_from_vw_example(vw_example):
        """ get y from a vw_example. this works for regression datasets.
        """
        return float(vw_example.split('|')[0])

    @staticmethod
    def get_ns_feature_dim_from_vw_example(vw_example) -> dict:
        """Get a dictionary of feature dimensionality for each namespace

        Assumption: assume the vw_example takes one of the following format
        depending on whether the example includes the feature names
 
        format 1: 'y | ns1 feature1:feature_value1 feature2:feature_value2 | 
                   ns2 feature3:feature_value3 feature4:feature_value4'
        format 2: 'y | ns1 feature_value1 feature_value2 |
                   ns2 feature_value3 feature_value4'

        The output of both cases are {'ns1': 2, 'ns2': 2}

        For more information about the input formate of vw example, please refer to 
        https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format
        """
        ns_feature_dim = {}
        data = vw_example.split('|')
        for i in range(1, len(data)):
            if ':' in data[i]:
                ns_w_feature = data[i].split(' ')
                ns = ns_w_feature[0]
                feature = ns_w_feature[1:]
                feature_dim = len(feature)
            else:
                data_split = data[i].split(' ')
                ns = data_split[0]
                feature_dim = len(data_split) - 1
                if len(data_split[-1]) == 0:
                    feature_dim -= 1
            if len(ns) == 1:
                ns_feature_dim[ns] = feature_dim
        logger.debug('name space feature dimension %s', ns_feature_dim)
        return ns_feature_dim
