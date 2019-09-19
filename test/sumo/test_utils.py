from sumo import utils
import numpy as np
import os
import pytest


def test_check_matrix_symmetry():
    assert utils.check_matrix_symmetry(np.array([[1, 2], [2, 1]])) is True
    assert utils.check_matrix_symmetry(np.array([[1, 2], [3, 1]])) is False
    assert utils.check_matrix_symmetry(np.array([[1, 2, 3], [1, 2, 3]])) is False

    m = np.random.rand(5, 5)
    m = (m + m.T) / 2
    assert utils.check_matrix_symmetry(m) is True

    m[3, 2] = np.nan
    m[2, 3] = np.nan
    assert utils.check_matrix_symmetry(m) is True


def test_save_arrays_to_npz(tmpdir):
    fname = os.path.join(tmpdir, "test_data")
    fname_npz = os.path.join(tmpdir, "test_data_npz.npz")

    a1 = np.arange(16).reshape((4, 4))
    a2 = (a1 * a1.T) / 2

    utils.save_arrays_to_npz([a1, a2], fname_npz)
    utils.save_arrays_to_npz({"layer1": a1, "layer2": a2}, fname)
    with pytest.raises(ValueError):
        utils.save_arrays_to_npz([[1], [2]], fname_npz)

    assert os.path.exists(fname_npz)
    assert not os.path.exists(fname)
    assert os.path.exists(fname + ".npz")

    handle = np.load(fname_npz)
    assert all([x in getattr(handle, "files") for x in ["0", "1"]])
    assert np.array_equal(handle['0'], a1)
    assert np.array_equal(handle['1'], a2)

    handle = np.load(fname + ".npz")
    assert all([x in getattr(handle, "files") for x in ["layer1", "layer2"]])
    assert np.array_equal(handle['layer1'], a1)
    assert np.array_equal(handle['layer2'], a2)


def test_load_npz(tmpdir):
    fname = os.path.join(tmpdir, "test_data.npz")
    a1 = np.arange(16).reshape((4, 4))
    a2 = (a1 * a1.T) / 2
    np.savez(file=fname, a1=a1, a2=a2)

    with pytest.raises(FileNotFoundError):
        utils.load_npz(os.path.join(tmpdir, "not_a_file.npz"))
    files = utils.load_npz(fname)

    assert isinstance(files, dict)
    assert all([x in files.keys() for x in ["a1", "a2"]])
    assert np.array_equal(files['a1'], a1)
    assert np.array_equal(files['a2'], a2)


def test_extract_ncut():
    samples = 10
    a = np.random.random((samples, samples))
    a = (a * a.T) / 2

    labels = utils.extract_ncut(a, 2)
    assert labels.size == samples
    assert np.unique(labels).size == 2

    labels = utils.extract_ncut(a, 4)
    assert labels.size == samples
    assert np.unique(labels).size == 4

    with pytest.raises(AssertionError):
        a[0, 1], a[1, 0] = 2, 1
        utils.extract_ncut(a, 4)


def test_check_accuracy():
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    # incorrect method
    with pytest.raises(ValueError):
        utils.check_accuracy(labels, labels, method="method")

    # incorrect arrays
    with pytest.raises(ValueError):
        utils.check_accuracy(labels, labels[0:-1], method="method")

    ari = utils.check_accuracy(labels, labels, method="ARI")
    assert ari == 1.0
