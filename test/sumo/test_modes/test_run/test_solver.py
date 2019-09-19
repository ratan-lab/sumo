from sumo.modes.run.solver import svd_h_init, svd_si_init, SumoNMF, SumoNMFResults
from sumo.network import MultiplexNet
from sumo.utils import check_matrix_symmetry
import numpy as np
import os
import pytest


def test_svd_si_init():
    a = np.random.random((20, 20))
    a = (a * a.T) / 2

    s = svd_si_init(a, k=3)
    assert check_matrix_symmetry(s)
    assert s.shape == (3, 3)

    a[0, 1], a[1, 0] = 0, 1
    with pytest.raises(ValueError):
        svd_si_init(a, k=3)


def test_svd_h_init():
    a = np.random.random((20, 20))
    a = (a * a.T) / 2

    h = svd_h_init(a, k=3)
    assert h.shape == (20, 3)

    h = svd_h_init(a, k=5)
    assert h.shape == (20, 5)

    a[0, 1], a[1, 0] = 0, 1
    with pytest.raises(ValueError):
        svd_h_init(a, k=3)


def test_init():
    a0 = np.random.random((10, 10))
    a0 = (a0 * a0.T) / 2
    a1 = np.random.random((10, 10))
    a1 = (a1 * a1.T) / 2
    sample_names = ['sample_{}'.format(i) for i in range(10)]

    with pytest.raises(ValueError):
        SumoNMF(a0)

    graph = MultiplexNet(adj_matrices=[a0, a1], node_labels=np.array(sample_names))
    nmf = SumoNMF(graph)
    assert np.array_equal((a0 + a1) / 2, nmf.avg_adj)

    # missing values in one layer
    a0[0, 1], a0[1, 0] = np.nan, np.nan
    graph = MultiplexNet(adj_matrices=[a0, a1], node_labels=np.array(sample_names))
    nmf = SumoNMF(graph)
    assert np.allclose(nmf.avg_adj[0, 1], a1[0, 1])

    # missing values in both layers
    a1[0, 1], a1[1, 0] = np.nan, np.nan
    graph = MultiplexNet(adj_matrices=[a0, a1], node_labels=np.array(sample_names))
    nmf = SumoNMF(graph)
    assert not np.isnan(nmf.avg_adj[0, 1])


def test_factorize():
    a0 = np.random.random((10, 10))
    a0 = (a0 * a0.T) / 2
    a1 = np.random.random((10, 10))
    a1 = (a1 * a1.T) / 2
    sample_names = ['sample_{}'.format(i) for i in range(10)]

    graph = MultiplexNet(adj_matrices=[a0, a1], node_labels=np.array(sample_names))
    nmf = SumoNMF(graph)

    # incorrect k value
    with pytest.raises(ValueError):
        nmf.factorize(sparsity_penalty=10.0, k=20)

    nmf.factorize(sparsity_penalty=10.0, k=5)


def test_extract_clusters():
    a = np.random.random((10, 10))
    a = (a * a.T) / 2
    sample_names = ['sample_{}'.format(i) for i in range(10)]

    graph = MultiplexNet(adj_matrices=[a], node_labels=np.array(sample_names))
    nmf = SumoNMF(graph)
    res = nmf.factorize(sparsity_penalty=10.0, k=2)

    assert res.labels is None
    res.extract_clusters(method="max_value")
    assert res.labels.shape[0] == 10
    assert all(res.labels < 2)

    # incorrect method
    with pytest.raises(ValueError):
        res.extract_clusters(method="method")
