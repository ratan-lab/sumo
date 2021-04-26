from sumo.constants import RUN_DEFAULTS
from sumo.modes.run.solvers.supervised_sumo import SupervisedSumoNMF
from sumo.modes.run.solvers.unsupervised_sumo import UnsupervisedSumoNMF
from sumo.network import MultiplexNet
import numpy as np
import pytest


def test_init():
    a0 = np.random.random((10, 10))
    a0 = (a0 * a0.T) / 2
    a1 = np.random.random((10, 10))
    a1 = (a1 * a1.T) / 2
    sample_names = ['sample_{}'.format(i) for i in range(10)]

    with pytest.raises(ValueError):
        UnsupervisedSumoNMF(a0, nbins=RUN_DEFAULTS['n'])

    graph = MultiplexNet(adj_matrices=[a0, a1], node_labels=np.array(sample_names))
    nmf = UnsupervisedSumoNMF(graph, nbins=RUN_DEFAULTS['n'])
    assert np.array_equal((a0 + a1) / 2, nmf.avg_adj)
    assert all([bin.size == 10 for bin in nmf.bins])

    # different bin size
    nmf = UnsupervisedSumoNMF(graph, nbins=RUN_DEFAULTS['n'], bin_size=9)
    assert all([bin.size == 9 for bin in nmf.bins])

    with pytest.raises(ValueError):
        # incorrect number of bins
        UnsupervisedSumoNMF(graph, nbins=0)
    with pytest.raises(ValueError):
        # too high bin size
        UnsupervisedSumoNMF(graph, nbins=RUN_DEFAULTS['n'], bin_size=20)

    # missing values in one layer
    a0[0, 1], a0[1, 0] = np.nan, np.nan
    graph = MultiplexNet(adj_matrices=[a0, a1], node_labels=np.array(sample_names))
    nmf = UnsupervisedSumoNMF(graph, nbins=RUN_DEFAULTS['n'])
    assert np.allclose(nmf.avg_adj[0, 1], a1[0, 1])

    # missing values in both layers
    a1[0, 1], a1[1, 0] = np.nan, np.nan
    graph = MultiplexNet(adj_matrices=[a0, a1], node_labels=np.array(sample_names))
    nmf = UnsupervisedSumoNMF(graph, nbins=RUN_DEFAULTS['n'])
    assert not np.isnan(nmf.avg_adj[0, 1])


def test_supervised_factorize():
    a0 = np.random.random((15, 15))
    a0 = (a0 * a0.T) / 2
    a1 = np.random.random((15, 15))
    a1 = (a1 * a1.T) / 2
    sample_names = ['sample_{}'.format(i) for i in range(15)]
    graph = MultiplexNet(adj_matrices=[a0, a1], node_labels=np.array(sample_names))

    priors = np.zeros((15, 3))
    priors[:5, 0] = 1
    priors[5:10, 1] = 1
    priors[10:, 2] = 1

    nmf = SupervisedSumoNMF(graph, nbins=RUN_DEFAULTS['n'], priors=priors)
    with pytest.raises(ValueError):
        nmf.factorize(sparsity_penalty=10, k=2)

    res = nmf.factorize(sparsity_penalty=10, k=3)
    assert res.d is not None and res.h0 is not None
