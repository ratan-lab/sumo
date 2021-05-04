from collections import Counter
from sumo.constants import RUN_DEFAULTS
from sumo.modes.run import solver
from sumo.modes.run.solvers.supervised_sumo import SupervisedSumoNMF
from sumo.modes.run.solvers.unsupervised_sumo import UnsupervisedSumoNMF
from sumo.network import MultiplexNet
import numpy as np
import pytest


def _create_test_graph(nsamples):
    a = np.random.random((nsamples, nsamples))
    a = (a * a.T) / 2
    sample_names = ['sample_{}'.format(i) for i in range(nsamples)]
    return MultiplexNet(adj_matrices=[a], node_labels=np.array(sample_names))


def test_factorize():
    class SomeSolver(solver.SumoSolver):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def factorize(self, **kwargs):
            super(SomeSolver, self).factorize(**kwargs)

    with pytest.raises(TypeError):
        SomeSolver()

    graph = _create_test_graph(nsamples=10)
    some_solver = SomeSolver(graph=graph, nbins=RUN_DEFAULTS['n'])
    with pytest.raises(TypeError):
        some_solver.factorize()
    with pytest.raises(NotImplementedError):
        some_solver.factorize(sparsity_penalty=0.1, k=2, max_iter=RUN_DEFAULTS['max_iter'],
                              tol=RUN_DEFAULTS['tol'], calc_cost=RUN_DEFAULTS['calc_cost'],
                              logger_name=None, bin_id=None)


def test_extract_clusters():
    graph = _create_test_graph(nsamples=10)
    priors = np.zeros((10, 2))
    priors[:5, 0] = 1
    priors[5:, 1] = 1

    for nmf in [UnsupervisedSumoNMF(graph, nbins=RUN_DEFAULTS['n']),
                SupervisedSumoNMF(graph, nbins=RUN_DEFAULTS['n'], priors=priors)]:
        res = nmf.factorize(sparsity_penalty=10.0, k=2)

        assert res.labels is None
        res.extract_clusters(method="max_value")
        assert res.labels.shape[0] == 10
        assert all(res.labels < 2)

        # incorrect method
        with pytest.raises(ValueError):
            res.extract_clusters(method="method")


def test_calculate_avg_adjacency():
    a1 = np.array([[1, 0, 0.5], [0, 1, 1], [0.5, 1, 1]])
    a2 = np.array([[1, 0.5, np.nan], [0.5, 1, 1], [np.nan, 1, 1]])

    graph = MultiplexNet(adj_matrices=[a1, a2], node_labels=np.array(['sample_1', 'sample_2']))
    nmf = UnsupervisedSumoNMF(graph, nbins=RUN_DEFAULTS['n'])
    adj = nmf.calculate_avg_adjacency()
    assert np.allclose(adj, np.array([[1, 0.25, 0.5], [0.25, 1, 1], [0.5, 1, 1]]))


def test_create_sample_bins():
    s = UnsupervisedSumoNMF(graph=_create_test_graph(nsamples=12), nbins=RUN_DEFAULTS['n'])
    assert s.bin_size == round(12 * (1 - RUN_DEFAULTS['subsample']))

    bins = s.create_sample_bins()
    assert all([bin.size == s.bin_size for bin in bins])
    counter = Counter(x for xs in bins for x in set(xs))
    assert all([x in counter.keys() for x in range(12)])  # all samples represented

    s.nbins = None
    with pytest.raises(ValueError):
        s.create_sample_bins()
