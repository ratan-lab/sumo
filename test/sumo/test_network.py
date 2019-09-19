from sumo.network import MultiplexNet
from sumo.utils import check_matrix_symmetry
import numpy as np
import pytest


def _create_test_adj(n: int):
    a = np.random.random((n, n))
    a *= a.T
    a /= 2
    return a


def test_init():
    a0 = _create_test_adj(10)
    a1 = _create_test_adj(10)
    sample_names = ['sample_{}'.format(i) for i in range(10)]

    graph = MultiplexNet(adj_matrices=[a0, a1], node_labels=np.array(sample_names))
    assert graph.layers == 2 and graph.nodes == 10
    assert np.sum(graph.connections) == a0.size * 2  # all values in both layers

    # missing samples
    a0[:, 0], a0[0, :] = np.nan, np.nan
    graph = MultiplexNet(adj_matrices=[a0, a1], node_labels=np.array(sample_names))
    assert 0 not in graph.samples[0] and 0 in graph.samples[1]

    # incorrect adjacency matrices
    with pytest.raises(TypeError):
        MultiplexNet(adj_matrices=[[2], [4]], node_labels=np.array(sample_names))

    a2 = a0.copy()
    a2[2, 3] = np.nan
    with pytest.raises(ValueError):
        MultiplexNet(adj_matrices=[a1, a2], node_labels=np.array(sample_names))

    with pytest.raises(ValueError):
        MultiplexNet(adj_matrices=[], node_labels=np.array(sample_names))

    # different number of nodes in layers
    with pytest.raises(ValueError):
        MultiplexNet(adj_matrices=[a0[:5, :5], a1], node_labels=np.array(sample_names))

    # not normalized data
    a2 = a0.copy()
    a2[0, 1], a2[1, 0] = -1, -1
    with pytest.raises(ValueError):
        MultiplexNet(adj_matrices=[a1, a2], node_labels=np.array(sample_names))

    a2[0, 1], a2[1, 0] = 2, 2
    with pytest.raises(ValueError):
        MultiplexNet(adj_matrices=[a1, a2], node_labels=np.array(sample_names))


def test_get_clustering_quality():
    a = np.array([
        [1, 1, 0, 0.1],
        [1, 1, 0, 0.1],
        [0, 0, 1, 0.8],
        [0.1, 0.1, 0.8, 1]
    ])
    samples = 4
    assert check_matrix_symmetry(a)
    sample_names = ['sample_{}'.format(i) for i in range(samples)]

    graph = MultiplexNet(adj_matrices=[a], node_labels=np.array(sample_names))
    qual = graph.get_clustering_quality(np.array([0, 0, 1, 1]))
    assert qual == (1 + 0.8) / (samples ** 2)

    graph = MultiplexNet(adj_matrices=[a, a], node_labels=np.array(sample_names))
    qual = graph.get_clustering_quality(np.array([0, 0, 1, 1]))
    assert qual == (1 + 0.8) / (samples ** 2) * 2

    # incorrect labels
    with pytest.raises(AssertionError):
        graph.get_clustering_quality(np.array([0, 1]))
