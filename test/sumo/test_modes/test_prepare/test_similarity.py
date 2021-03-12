from numba.core.errors import TypingError
from sumo.modes.prepare import similarity
from sumo.utils import check_matrix_symmetry
from sumo.constants import SIMILARITY_METHODS
import numpy as np
import pytest


def _test_assumptions_no_numba(func):
    with pytest.raises(AssertionError):
        func(np.array([1]), np.array([1, 1]), missing=0.1)

    with pytest.raises(AttributeError):
        func([2], [3], missing=0.1)


def _test_assumptions_numba(func):
    with pytest.raises(AssertionError):
        func(np.array([1]), np.array([1, 1]), missing=0.1)

    with pytest.raises(TypingError):
        # running a function in nopython numba mode with incorrect attribute types raises numba specific error
        func([2], [3], missing=0.1)


def _test_threshold(func):
    assert np.isnan(func(np.array([np.nan, 1, 1]), np.array([1, np.nan, 2]), missing=0.5))
    assert np.isnan(func(np.array([np.nan, 1]), np.array([1, np.nan]), missing=0.1))


def test_euclidean_dist():
    assert np.allclose(similarity.euclidean_dist(np.array([1, 2]), np.array([1, 2]), missing=0.1), 0)
    assert np.allclose(similarity.euclidean_dist(np.arange(16), np.arange(16), missing=0.1), 0)

    v = np.array([1 if i != 0 else 2 for i in range(100)])
    assert np.allclose(similarity.euclidean_dist(v, np.ones(100), missing=0.1), 1 / 100)

    assert np.allclose(similarity.euclidean_dist(np.array([1, 1, 1]), np.array([1, 1, 2]), missing=0.1), 1 / 3)
    assert np.allclose(similarity.euclidean_dist(np.array([1, np.nan, 1]), np.array([1, np.nan, 2]), missing=0.1), 0.5)
    assert np.allclose(similarity.euclidean_dist(np.array([np.nan, 1, 1]), np.array([1, np.nan, 2]), missing=0.1), 1)

    _test_assumptions_numba(similarity.euclidean_dist)
    _test_threshold(similarity.euclidean_dist)


def test_cosine_sim():
    assert np.allclose(similarity.cosine_sim(np.array([1, 0, 0]), np.array([0, 1, 0]), missing=0.1), 0)
    assert np.allclose(similarity.cosine_sim(np.array([1, 1, 0]), np.array([1, 1, 0]), missing=0.1), 1)

    _test_assumptions_no_numba(similarity.cosine_sim)
    _test_threshold(similarity.cosine_sim)


def test_correlation():
    _test_assumptions_no_numba(similarity.correlation)
    with pytest.raises(AssertionError):
        similarity.correlation(np.array([1]), np.array([1, 1]), missing=0.1, method="method")

    _test_threshold(similarity.correlation)

    assert np.allclose(similarity.correlation(np.array([1, 2]), np.array([1, 2]), missing=0.1), 1)
    assert np.allclose(similarity.correlation(np.arange(16), np.arange(16), missing=0.1), 1)
    assert np.allclose(similarity.correlation(np.arange(16), np.arange(16), missing=0.1, method="spearman"), 1)

    assert np.allclose(similarity.correlation(np.array([1, 2, 3]), np.array([1, 2, np.nan]), missing=0.1), 1)
    assert np.allclose(
        similarity.correlation(np.array([1, 2, 3, np.nan]), np.array([1, 2, np.nan, np.nan]), missing=0.1), 1)


def test_feature_rbf_similarity():
    f = np.random.random((10, 20))

    a = similarity.feature_rbf_similarity(f)
    assert check_matrix_symmetry(a)
    assert np.all(np.diag(a) == 1)

    # incorrect hyperparameter
    with pytest.raises(ValueError):
        similarity.feature_rbf_similarity(f, alpha=0)

    # missing samples
    f2 = f.copy()
    f2[:2, :] = np.nan
    a = similarity.feature_rbf_similarity(f2)
    assert np.all(np.isnan(a[:2, :])) and np.all(np.isnan(a[:, :2]))

    # very similar samples but high n parameter
    f[:5, :] = f[5:, :]
    with pytest.raises(ValueError):
        similarity.feature_rbf_similarity(f)

    a = similarity.feature_rbf_similarity(f, n=0.2)
    assert check_matrix_symmetry(a)
    assert np.allclose(np.diag(a), 1)


def test_feature_to_adjacency():
    f = np.random.random((10, 20))

    # incorrect similarity method
    with pytest.raises(ValueError):
        similarity.feature_to_adjacency(f, method="method")

    for method in SIMILARITY_METHODS:
        a = similarity.feature_to_adjacency(f, method=method, n=0.1, alpha=0.5)
        assert check_matrix_symmetry(a)
        assert np.allclose(np.diag(a), 1)
        assert np.all(a >= 0)
