# cython: language_level=3
# cython: profile=True
import itertools

import numpy as np

cimport cython
cimport numpy as cnp
from sklearn.tree._tree cimport Node
from sklearn.tree._tree cimport Tree


cnp.import_array()


ctypedef cnp.npy_intp SIZE_t  # Type for indices and counters
ctypedef cnp.npy_bool BOOL_t  # for boolean

cdef extern from "math.h" nogil:
    bint isnan(double x)


cdef class FanovaTree:
    cdef:
        Tree _tree
        SIZE_t _n_features
        double _variance
        double [:,:] _statistics, _search_spaces
        BOOL_t[:,:] _subtree_active_features
        object _split_midpoints
        object _split_sizes

    def __cinit__(self, Tree tree, double[:,:] search_spaces):
        assert search_spaces.shape[0] == tree.n_features
        assert search_spaces.shape[1] == 2

        self._tree = tree
        self._search_spaces = search_spaces
        self._n_features = search_spaces.shape[0]
        self._statistics = _precompute_statistics(tree, search_spaces)

        split_midpoints, split_sizes = _precompute_split_midpoints_and_sizes(tree, search_spaces)
        subtree_active_features = self._precompute_subtree_active_features()
        self._split_midpoints = split_midpoints
        self._split_sizes = split_sizes
        self._subtree_active_features = subtree_active_features
        self._variance = -1.0  # Computed lazily and requires `self._statistics`.

    @property
    def variance(self) -> float:
        if self._variance == -1.0:
            leaf_node_indices = np.nonzero(self._tree.feature < 0)[0]
            statistics = np.asarray(self._statistics, order='C')[leaf_node_indices]
            values = statistics[:, 0]
            weights = statistics[:, 1]
            average_values = np.average(values, weights=weights)
            variance = np.average((values - average_values) ** 2, weights=weights)

            self._variance = variance

        return self._variance

    def get_marginal_variance(self, features: np.ndarray) -> float:
        cdef:
            SIZE_t i, n_loop = 1
            SIZE_t[:] active_nodes_buf
            double value, weight
            double[:, :, :] active_search_spaces_buf
            double[:] values, weights
            cnp.ndarray[BOOL_t, cast = True, ndim = 1] active_features

        assert features.size > 0

        # For each midpoint along the given dimensions, traverse this tree to compute the
        # marginal predictions.
        midpoints = [self._split_midpoints[f] for f in features]
        sizes = [self._split_sizes[f] for f in features]

        for m in midpoints:
            n_loop *= len(m)

        product_midpoints = itertools.product(*midpoints)
        product_sizes = itertools.product(*sizes)

        sample = np.full(self._n_features, fill_value=np.nan, dtype=np.float64)
        active_features = np.zeros_like(np.asarray(sample), dtype=np.bool_)

        active_nodes_buf = np.empty(shape=self._tree.node_count, dtype=np.int64)
        active_search_spaces_buf = np.empty(shape=(self._tree.node_count, self._search_spaces.shape[0], 2), dtype=np.float64)

        values = np.empty(shape=n_loop, dtype=np.float64)
        weights = np.empty_like(values)
        for i, (midpoints, sizes) in enumerate(zip(product_midpoints, product_sizes)):
            sample[features] = np.array(midpoints)

            value, weight = self._get_marginalized_statistics(
                sample, active_features,
                active_nodes_buf, active_search_spaces_buf
            )
            weight *= cy_prod(np.array(sizes, dtype=np.float64))

            values[i] = value
            weights[i] = weight

        average_values = np.average(values, weights=weights)
        variance = np.average((values - average_values) ** 2, weights=weights)

        assert variance >= 0.0
        return variance

    @cython.boundscheck(False)
    cdef inline BOOL_t _is_subtree_active(self, SIZE_t node_index, BOOL_t[:] active_features) nogil:
        cdef:
            SIZE_t i
            BOOL_t[:] subtree_active_feature = self._subtree_active_features[node_index]
        for i in range(active_features.shape[0]):
            if active_features[i] and subtree_active_feature[i]:
                return True
        return False

    @cython.boundscheck(False)
    cdef (double, double) _get_marginalized_statistics(
        self, double[:] feature_vector, BOOL_t[:] active_features, SIZE_t[:] active_nodes, double[:, :, :] active_search_spaces
    ) nogil:
        cdef:
            double[:,:] buf
            double response

            double sum_weighted_value = 0, sum_weight = 0, tmp_weight, weighted_average
            SIZE_t i, node_index, active_nodes_index
            Node* node

        # Start from the root and traverse towards the leafs.
        active_nodes_index = 0
        active_nodes[active_nodes_index] = 0
        active_search_spaces[active_nodes_index, ...] = self._search_spaces

        for i in range(feature_vector.shape[0]):
            if isnan(feature_vector[i]):
                active_search_spaces[active_nodes_index, i, 0] = 0.0
                active_search_spaces[active_nodes_index, i, 1] = 1.0
            else:
                active_features[i] = True

        while active_nodes_index >= 0:
            node_index = active_nodes[active_nodes_index]
            node = self._tree.nodes + node_index
            search_spaces = active_search_spaces[active_nodes_index]
            active_nodes_index -= 1

            if node.feature >= 0:  # Not leaf.
                # If node splits on an active feature, push the child node that we end up in.
                response = feature_vector[node.feature]
                if not isnan(response):
                    active_nodes_index += 1
                    buf = active_search_spaces[active_nodes_index]
                    if response <= node.threshold:
                        _get_node_left_child_subspaces(node, search_spaces, buf)
                        active_nodes[active_nodes_index] = node.left_child
                    else:
                        _get_node_right_child_subspaces(node, search_spaces, buf)
                        active_nodes[active_nodes_index] = node.right_child
                    continue

                # If subtree starting from node splits on an active feature, push both child nodes.
                if self._is_subtree_active(node_index, active_features) == 1:
                    active_nodes_index += 1
                    active_nodes[active_nodes_index] = node.left_child
                    active_search_spaces[active_nodes_index] = search_spaces

                    active_nodes_index += 1
                    active_nodes[active_nodes_index] = node.right_child
                    active_search_spaces[active_nodes_index] = search_spaces
                    continue

            # avg = sum(a * weights) / sum(weights)
            tmp_weight = self._statistics[node_index, 1] / _get_cardinality(search_spaces)
            sum_weighted_value += self._statistics[node_index, 0] * tmp_weight
            sum_weight += tmp_weight

        weighted_average = sum_weighted_value / sum_weight
        return weighted_average, sum_weight

    cdef BOOL_t[:,:] _precompute_subtree_active_features(self):
        cdef:
            SIZE_t node_index
            cnp.ndarray subtree_active_features = np.full((self._tree.node_count, self._n_features), fill_value=False)
            Node* node

        for node_index in reversed(range(self._tree.node_count)):
            node = self._tree.nodes + node_index
            if node.feature >= 0:
                subtree_active_features[node_index, node.feature] = True
                subtree_active_features[node_index] |= subtree_active_features[node.left_child]
                subtree_active_features[node_index] |= subtree_active_features[node.right_child]

        return subtree_active_features


@cython.boundscheck(False)
cdef double cy_prod(double[:] items) nogil:
    cdef:
        double r = 1
        SIZE_t i
    for i in range(items.shape[0]):
        r *= items[i]
    return r


@cython.boundscheck(False)
def _precompute_statistics(Tree tree, double[:,:] search_spaces):
    cdef:
        double[:,:] statistics
        double[:,:,:] subspaces
        SIZE_t node_index, n_nodes = tree.node_count
        double v1, v2, w1, w2
        Node* node = tree.nodes

    # Holds for each node, its weighted average value and the sum of weights.
    statistics = np.empty((n_nodes, 2), dtype=np.float64)
    subspaces = np.empty(shape=(n_nodes, search_spaces.shape[0], 2), dtype=np.float64)
    subspaces[0, ...] = search_spaces

    with nogil:
        # Compute marginals for leaf nodes.
        for node_index in range(n_nodes):
            node = tree.nodes + node_index
            if node.feature < 0:
                statistics[node_index][0] = tree.value[node_index]
                statistics[node_index][1] = _get_cardinality(subspaces[node_index])
            else:
                _get_node_left_child_subspaces(node, subspaces[node_index], subspaces[node.left_child])
                _get_node_right_child_subspaces(node, subspaces[node_index], subspaces[node.right_child])

        # Compute marginals for internal nodes.
        for node_index in reversed(range(n_nodes)):
            node = tree.nodes + node_index
            if node.feature >= 0:
                v1 = statistics[node.left_child, 0]
                w1 = statistics[node.left_child, 1]
                v2 = statistics[node.right_child, 0]
                w2 = statistics[node.right_child, 1]
                # avg = sum(a * weights) / sum(weights)
                statistics[node_index][0] = (v1 * w1 + v2 * w2) / (w1 + w2)
                statistics[node_index][1] = w1 + w2
    return statistics


cdef _precompute_split_midpoints_and_sizes(Tree tree, double[:,:] search_spaces):
    cdef:
        SIZE_t node_index, feature
        SIZE_t n_features = search_spaces.shape[0]
        Node* node

    all_split_values = [set() for _ in range(n_features)]
    for node_index in range(tree.node_count):
        node = tree.nodes + node_index
        if node.feature >= 0:  # Not leaf.
            all_split_values[node.feature].add(node.threshold)

    midpoints = []
    sizes = []

    for feature in range(n_features):
        feature_split_values = np.array(list(all_split_values[feature]), dtype=np.float64)
        feature_split_values.sort()

        feature_split_values = np.concatenate(
            (
                np.atleast_1d(search_spaces[feature, 0]),
                feature_split_values,
                np.atleast_1d(search_spaces[feature, 1]),
            )
        )
        midpoint = 0.5 * (feature_split_values[1:] + feature_split_values[:-1])
        size = feature_split_values[1:] - feature_split_values[:-1]

        midpoints.append(midpoint)
        sizes.append(size)

    return midpoints, sizes


@cython.boundscheck(False)
cdef inline double _get_cardinality(double[:,:] search_spaces) nogil:
    cdef double result = 1
    for i in range(search_spaces.shape[0]):
        result *= search_spaces[i, 1] - search_spaces[i, 0]
    return result


cdef inline void _get_node_left_child_subspaces(
    Node* node, double[:,:] search_spaces, double[:,:] buf
) nogil:
    _get_subspaces(search_spaces, 1, node.feature, node.threshold, buf)


cdef inline void _get_node_right_child_subspaces(
    Node* node, double[:,:] search_spaces, double[:,:] buf
) nogil:
    _get_subspaces(search_spaces, 0, node.feature, node.threshold, buf)


@cython.boundscheck(False)
cdef inline void _get_subspaces(
    double[:,:] search_spaces, SIZE_t search_spaces_column, SIZE_t feature, double threshold, double[:,:] buf
) nogil:
    buf[...] = search_spaces
    buf[feature, search_spaces_column] = threshold
