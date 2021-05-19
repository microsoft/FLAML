import numpy as np
import logging
import itertools
from typing import Dict, Optional, List, Tuple
# from flaml.tune.sample import Float
from flaml.tune.sample import Categorical
from flaml.tune.sample import PolynomialExpansionSet, Float
from flaml.searcher import CFO
from ..tune.trial import Trial
from ..tune.online_trial import VWOnlineTrial

logger = logging.getLogger(__name__)


class BaseSearcher:
    """Implementation of the BaseSearcher

    """

    def __init__(self,
                 metric: str = None,
                 mode: str = None,
                 ):
        pass

    def set_search_properties(self, metric: Optional[str], mode: Optional[str],
                              config: dict) -> bool:
        if metric:
            self._metric = metric
        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'."
            self._mode = mode

    def next_trial(self):
        NotImplementedError

    def on_trial_result(self, trial_id: str, result: Dict):
        pass

    def on_trial_complete(self, trial):
        pass


class ChampionFrontierSearcher(BaseSearcher):
    """This class serves the role of ConfigOralce.
    
    NOTE about searcher_trial_id and trial_id:
    Every time we create a VW trial, we generate a searcher_trial_id.
    At the same time, we also record the trial_id of the VW trial. 
    Note that the trial_id is a unique signature of the configuraiton. 
    So if two VWTrial is associated with the same config, they will have the same trial_id
    (although not the same searcher_trial_id).
    searcher_trial_id will be used in suggest()
    """
    POLY_EXPANSION_ADDITION_NUM = 1
    POLY_EXPANSION_ORDER = 2
    NUMERICAL_NUM = 2

    CFO_SEARCHER_METRIC_NAME = 'pseudo_loss'
    CFO_SEARCHER_LARGE_LOSS = 100
    NUM_RANDOM_SEED = 111

    CHAMPION_TRIAL_NAME = 'champion_trial'

    def __init__(self,
                 init_config: dict,
                 metric: Optional[str] = None,
                 mode: Optional[str] = None,
                 config_oracle_random_seed: Optional[int] = 2345,
                 space: Optional[dict] = None,
                 online_trial_args={},
                 nonpoly_searcher_name='CFO' # or 'Niave'
                 ):
        self._init_config = init_config
        self._seed = config_oracle_random_seed
        self._space = space
        self._online_trial_args = online_trial_args
        self._nonpoly_searcher_name = nonpoly_searcher_name

        self._random_state = np.random.RandomState(self._seed)
        self._searcher_for_nonpoly_hp = {}
        self._space_of_nonpoly_hp = {}
        # dicts to remember the mapping between searcher_trial_id and trial_id
        self._searcher_trialid_to_trialid = {}  # key: searcher_trial_id, value: trial_id
        self._trialid_to_searcher_trial_id = {}  # value: trial_id, key: searcher_trial_id 
        self._new_challenger_list = []
        # initialize the search in set_search_properties
        self.set_search_properties(metric, mode, {self.CHAMPION_TRIAL_NAME: None}, init_call=True)
        logger.debug('using random seed %s in config oracle', self._seed)

    def set_search_properties(self, metric: Optional[str], mode: Optional[str],
                              config: dict, init_call=False) -> bool:
        """ construct search space with given config, and setup the search
        """
        super().set_search_properties(metric, mode, config)
        ## Now we are using self._new_challenger_list.pop() to do the sampling
        # *********Use ConfigOralce (i.e, self._generate_new_space to generate list of new challengers)
        # assert 'champion_trial' in config.keys()
        logger.info('champion trial %s', config)
        champion_trial = config.get(self.CHAMPION_TRIAL_NAME, None)
        if champion_trial is None:
            champion_trial = self._create_trial_from_config(self._init_config)
        # create a list of challenger trials
        print('champion_trial.config', champion_trial.trial_id, champion_trial.config)
        new_challenger_list = self._query_config_oracle(champion_trial.config,
                                                        champion_trial.trial_id,
                                                        self._trialid_to_searcher_trial_id[champion_trial.trial_id])
        # add the champion as part of the new_challenger_list
        # we check dup when calling next_trial()
        self._new_challenger_list = self._new_challenger_list + new_challenger_list
        if init_call:
            self._new_challenger_list.append(champion_trial)
        ### add the champion as part of the new_challenger_list when called initially
        logger.critical('Created challengers from champion %s', champion_trial.trial_id)
        logger.critical('New challenger size %s, %s', len(self._new_challenger_list),
                        [t.trial_id for t in self._new_challenger_list])

    def next_trial(self):
        """Return a trial from the _new_challenger_list
        """
        next_trial = None
        if self._new_challenger_list:
            next_trial = self._new_challenger_list.pop()
        return next_trial

    def _create_trial_from_config(self, config, searcher_trial_id=None):
        if searcher_trial_id is None:
            searcher_trial_id = Trial.generate_id()
        trial = VWOnlineTrial(config, **self._online_trial_args)
        self._searcher_trialid_to_trialid[searcher_trial_id] = trial.trial_id
        # only update the dict when the trial_id does not exist
        if trial.trial_id not in self._trialid_to_searcher_trial_id:
            self._trialid_to_searcher_trial_id[trial.trial_id] = searcher_trial_id
        return trial

    def _query_config_oracle(self, seed_config, seed_config_trial_id,
                             seed_config_searcher_trial_id=None) -> List[Trial]:
        """Give the seed config, generate a list of new configs (which are supposed to include 
        at least one config that has better performance than the input seed_config)
        """
        # group the hyperparameters according to whether the configs of them are independent with
        # with the other hyperparameters
        hyperparameter_config_groups = []
        searcher_trial_ids_groups = []
        nonpoly_config = {}
        for k, v in seed_config.items():
            config_domain = self._space[k]
            if isinstance(config_domain, PolynomialExpansionSet):
                # get candidate configs for hyperparameters which are independent with other hyperparamters
                partial_new_configs = self._generate_independent_hp_configs(k, v, config_domain)
                if partial_new_configs:
                    hyperparameter_config_groups.append(partial_new_configs) 
                    # does not have searcher_trial_ids
                    searcher_trial_ids_groups.append([])
            else:
                # otherwise we need to deal with them in group
                nonpoly_config[k] = v
                if k not in self._space_of_nonpoly_hp:
                    self._space_of_nonpoly_hp[k] = self._space[k]

        # -----------generate partial new configs for non-PolynomialExpansionSet hyperparameters
        if nonpoly_config:
            new_searcher_trial_ids = []
            if 'CFO' in self._nonpoly_searcher_name:
                if seed_config_trial_id not in self._searcher_for_nonpoly_hp:
                    self._searcher_for_nonpoly_hp[seed_config_trial_id] = CFO(space=self._space_of_nonpoly_hp,
                                                                              points_to_evaluate=[nonpoly_config],
                                                                              metric=self.CFO_SEARCHER_METRIC_NAME,
                                                                              )
                    # initialize the search in set_search_properties
                    self._searcher_for_nonpoly_hp[seed_config_trial_id].set_search_properties(
                        config={'metric_target': self.CFO_SEARCHER_LARGE_LOSS})
                    # We need to call this for once, such that the seed config in points_to_evaluate will be called
                    # to be tried
                    self._searcher_for_nonpoly_hp[seed_config_trial_id].suggest(seed_config_searcher_trial_id)
                # assuming minimization
                pseudo_loss = self.CFO_SEARCHER_LARGE_LOSS if self._searcher_for_nonpoly_hp[seed_config_trial_id].get_metric_target is None \
                                                            else self._searcher_for_nonpoly_hp[seed_config_trial_id].get_metric_target * 0.95
                pseudo_result_to_report = {}
                for k, v in nonpoly_config.items():
                    pseudo_result_to_report['config/' + str(k)] = v
                pseudo_result_to_report[self.CFO_SEARCHER_METRIC_NAME] = pseudo_loss
                pseudo_result_to_report['time_total_s'] = 1
                self._searcher_for_nonpoly_hp[seed_config_trial_id].on_trial_complete(seed_config_searcher_trial_id,
                                                                                      result=pseudo_result_to_report)
                partial_new_numerical_configs = []
                # for i in range(self.NUMERICAL_NUM):
                while len(partial_new_numerical_configs) < self.NUMERICAL_NUM:
                    # suggest multiple times
                    new_searcher_trial_id = Trial.generate_id()
                    new_searcher_trial_ids.append(new_searcher_trial_id)
                    suggestion = self._searcher_for_nonpoly_hp[seed_config_trial_id].suggest(new_searcher_trial_id)
                    if suggestion is not None:
                        partial_new_numerical_configs.append(suggestion)
                logger.info('partial_new_numerical_configs %s', partial_new_numerical_configs)
            else:
                # An alternative implementation of FLOW2
                partial_new_numerical_configs = self._generate_num_candidates(nonpoly_config,
                                                                              self.NUMERICAL_NUM)
            if partial_new_numerical_configs:
                hyperparameter_config_groups.append(partial_new_numerical_configs)
                searcher_trial_ids_groups.append(new_searcher_trial_ids)
        # ----------- coordinate generation of new challengers in the case of multiple groups
        new_trials = []
        for i in range(len(hyperparameter_config_groups)):
            logger.info('hyperparameter_config_groups[i] %s %s', len(hyperparameter_config_groups[i]), hyperparameter_config_groups[i])
            for j, new_partial_config in enumerate(hyperparameter_config_groups[i]):
                new_seed_config = seed_config.copy()
                print('new_partial_config',  new_partial_config)
                new_seed_config.update(new_partial_config)
                # for some groups of the hyperparamters, we may have already generated the searcher_trial_id, 
                # in that case, we only need to retrive the searcher_trial_id such that we don't need to genearte again
                # For the case, searcher_trial_id is not geneated, we set teh searcher_trial_id to be None, and when
                # creating a trial from a config, a searcher_trial_id will be geneated if None is provided.
                # TODO: An alternative option is to geneate a searcher_trial_id for each partial config
                if searcher_trial_ids_groups[i]:
                    new_searcher_trial_id = searcher_trial_ids_groups[i][j]
                else:
                    new_searcher_trial_id = None
                new_trial = self._create_trial_from_config(new_seed_config, new_searcher_trial_id)
                self._searcher_trialid_to_trialid[new_searcher_trial_id] = new_trial.trial_id
                new_trials.append(new_trial)
        logger.info('new_configs %s', [t.trial_id for t in new_trials])
        return new_trials

    def _generate_independent_hp_configs(self, hp_name, current_config_value, config_domain) -> List:
        if isinstance(config_domain, PolynomialExpansionSet):
            monomials = list(current_config_value) + list(config_domain.init_monomials)
            logger.info('current_config_value %s %s', current_config_value, monomials)
            configs = self._generate_poly_expansion_sets(monomials,
                        self.POLY_EXPANSION_ADDITION_NUM,
                        self.POLY_EXPANSION_ORDER
                        )
        else:
            configs = [current_config_value]
            raise NotImplementedError
        configs_w_key = [{hp_name: hp_config} for hp_config in configs]
        return configs_w_key

    def _generate_poly_expansion_sets(self, champion_config,
                                      interaction_num_to_add, order=2):
        champion_all_combinations = self._generate_all_comb(champion_config)
        space = sorted(list(itertools.combinations(
                       champion_all_combinations, interaction_num_to_add)))
        self._random_state.shuffle(space)
        candidate_configs = [set(champion_config) | set(item) for item in space]
        assert len(candidate_configs) <= len(champion_config) * (len(champion_config) - 1) / 2.0
        final_candidate_configs = []
        for c in candidate_configs:
            new_c = set([e for e in c if len(e) > 1])
            final_candidate_configs.append(new_c)
        return final_candidate_configs
        # return candidate_configs

    def _generate_num_candidates(self, config, num):
        """Use local search to generate new candidate
        """
        self._random = np.random.RandomState(self.NUM_RANDOM_SEED)
        logger.info('num %s', num)
        half_num = int(num / 2)
        new_candidate_list = []
        hp_vec = list(config.keys())
        hp_config_vec = [config[k] for k in hp_vec]
        dim = len(hp_config_vec)
        candidate_config_vecs = []
        for i in range(half_num):
            move_direction = self._rand_vector_unit_sphere(dim)
            c1 = [hp_config_vec[i] * (2**move_direction[i]) for i in range(dim)]
            c2 = [hp_config_vec[i] * (2**(-move_direction[i])) for i in range(dim)]
            projected_c1 = self._project(hp_vec, c1)
            projected_c2 = self._project(hp_vec, c2)
            logger.info('on_trial_complete(trial.trial_id) %s %s %s', projected_c1,
                        type(projected_c1), hp_config_vec)
            if not (projected_c1 == hp_config_vec).all():
                candidate_config_vecs.append(projected_c1)
            if not (projected_c2 == hp_config_vec).all():
                candidate_config_vecs.append(projected_c2)

        for c in candidate_config_vecs:
            new_config = {}
            for i in range(dim):
                new_config[hp_vec[i]] = c[i]
            new_candidate_list.append(new_config)
        logger.info('new_candidate_list %s %s', config, new_candidate_list)
        return new_candidate_list

    def _rand_vector_unit_sphere(self, dim) -> np.ndarray:
        vec = self._random.normal(0, 1, dim)
        mag = np.linalg.norm(vec)
        return vec / mag

    def _project(self, hp_key_vec, hp_value_vec):
        projected_vec = []
        for i in range(len(hp_key_vec)):
            config_domain = self._space[hp_key_vec[i]]
            logger.info('Float %s', config_domain)
            value = hp_value_vec[i]
            upper = config_domain.upper
            lower = config_domain.lower
            logger.info('config_domain %s', lower)
            print(config_domain, lower, type(lower))
            projected = min(max(value, lower), upper)
            projected_vec.append(projected)
        return np.array(projected_vec)

    @staticmethod
    def _generate_all_comb(champion_config, order=2):
        def convert_nested_tuple_to_tuple(test_tuple):
            res = ''
            if type(test_tuple) is int:
                return (test_tuple,)
            else:
                for ele in test_tuple:
                    if isinstance(ele, tuple):
                        res += convert_nested_tuple_to_tuple(ele)
                    else:
                        res += ele
                return res

        def get_unique_combinations(list1, list2, no_dup=True):
            """ get combinatorial list of tuples
            """
            new_list = []
            for i in list1:
                for j in list2:
                    if len(i) < len(j): 
                        shorter = i
                        longer = j 
                    else:
                        shorter = j
                        longer = i
                    if no_dup and i != j and shorter not in longer:
                        new_tuple = sorted(''.join(i) + ''.join(j))  # tuple(sorted([i,j]))
                        new_tuple = ''.join(new_tuple)
                        if new_tuple not in new_list:
                            new_list.append(new_tuple)
            return new_list

        seed = champion_config.copy()
        all_combinations = seed
        inter_order = order
        while inter_order >= order:
            all_combinations = get_unique_combinations(all_combinations, seed)
            inter_order -= 1
        all_combinations = [convert_nested_tuple_to_tuple(element) for element in all_combinations]
        # remove duplicate features
        all_combinations = [c for c in all_combinations if len(c) == len(set(c))]
        logger.debug('all_combinations %s', all_combinations)
        all_combinations_joined = [''.join(candidate) for candidate in all_combinations]
        return all_combinations_joined
