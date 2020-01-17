from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
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


def test_check_categories():
    assert utils.check_categories(np.array([1, 2, 3])) == [1, 2, 3]
    assert utils.check_categories(np.array([1, 2, 1, 2])) == [1, 2]
    assert utils.check_categories(np.array([1, 2, 1, 2, np.nan, np.nan])) == [1, 2]


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


def test_is_standardized():
    f = np.random.random((20, 10))
    assert not utils.is_standardized(f, axis=0)[0]

    sc = StandardScaler()
    f = sc.fit_transform(f.T).T
    assert utils.is_standardized(f, axis=1)[0]
    assert utils.is_standardized(f.T, axis=0)[0]


def test_filter_features_and_samples():
    data_vals = np.random.random((10, 20))
    data = DataFrame(data_vals.T, columns=['sample_{}'.format(i) for i in range(data_vals.shape[0])],
                     index=['feature_{}'.format(i) for i in range(data_vals.shape[1])])

    filtered = utils.filter_features_and_samples(data)
    assert filtered.values.shape == (20, 10)

    # missing samples and features
    new_data = data.copy()
    new_data['sample_0'] = np.nan
    new_data.loc['feature_0'] = np.nan

    filtered = utils.filter_features_and_samples(new_data)
    assert 'feature_0' not in filtered.index
    assert 'sample_0' not in filtered.columns

    # missing values sample filtering
    new_data = data.copy()
    new_data['sample_0'][1:3] = np.nan

    filtered = utils.filter_features_and_samples(new_data)
    assert 'sample_0' in filtered.columns

    filtered = utils.filter_features_and_samples(new_data, drop_samples=0.05)
    assert 'sample_0' not in filtered.columns

    # missing values feature filtering
    new_data = data.copy()
    new_data.loc['feature_0'][1] = np.nan

    filtered = utils.filter_features_and_samples(new_data)
    assert 'feature_0' in filtered.index

    filtered = utils.filter_features_and_samples(new_data, drop_features=0.05)
    assert 'feature_0' not in filtered.index


def test_load_data_text(tmpdir):
    fname = os.path.join(tmpdir, "data.tsv")
    with pytest.raises(FileNotFoundError):
        utils.load_data_text(file_path=fname)

    # empty data
    empty_data = DataFrame()
    empty_data.to_csv(fname, sep="\t")
    with pytest.raises(ValueError):
        utils.load_data_text(file_path=fname)

    data_vals = np.random.random((10, 20))
    data = DataFrame(data_vals.T, columns=['sample_{}'.format(i) for i in range(data_vals.shape[0])],
                     index=['feature_{}'.format(i) for i in range(data_vals.shape[1])])

    # incorrectly formatted data
    data.to_csv(fname, sep=",")
    with pytest.raises(ValueError):
        utils.load_data_text(file_path=fname)

    # non-nummerical values
    data.to_csv(fname, sep="\t")
    with pytest.raises(ValueError):
        utils.load_data_text(file_path=fname)

    # tab delimited file
    utils.load_data_text(file_path=fname, sample_names=0, feature_names=0)

    # space delimited file
    data.to_csv(fname, sep=" ")
    utils.load_data_text(file_path=fname, sample_names=0, feature_names=0)

    # .gz file
    data.to_csv(fname + ".gz", sep=" ", compression='gzip')
    utils.load_data_text(file_path=fname, sample_names=0, feature_names=0)

    # .bz2 file
    data.to_csv(fname + ".bz2", sep=" ", compression='bz2')
    utils.load_data_text(file_path=fname, sample_names=0, feature_names=0)
