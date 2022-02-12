from sumo.modes.run.utils import svd_h_init, svd_si_init, svdEM
from sumo.utils import check_matrix_symmetry
import numpy as np
import pytest


def test_svdEM():
    a = np.random.random((20, 20))
    a[0, 4], a[3, 10] = np.nan, np.nan
    a = (a * a.T) / 2

    svdEM(a=a)
    for a in [np.array([]), np.array([[np.nan, np.nan], [np.nan, np.nan]])]:
        with pytest.raises(ValueError):
            svdEM(a=a)


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
