from matplotlib import rcParams
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from sklearn.metrics.cluster import entropy
from .constants import LOG_LEVELS, CLUSTER_METRICS
from sys import stdout
from typing import Union
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings


def docstring_formatter(*args, **kwargs):
    """ Decorator allowing for printing variable values in docstrings"""

    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return dec


def setup_logger(logger_name, level="INFO", log_file: str = None):
    """ Create and configure logging object """
    assert level in LOG_LEVELS

    formatter = logging.Formatter('%(message)s')
    if log_file:
        handler = logging.FileHandler(log_file, mode="w")
    else:
        handler = logging.StreamHandler(stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level))
    logger.addHandler(handler)
    return logger


def get_logger(logger_name: str = None):
    return logging.getLogger(logger_name if logger_name else 'main')


def check_matrix_symmetry(m: np.ndarray, tol=1e-8, equal_nan=True):
    """ Check symmetry of numpy array, after removal of missing samples"""
    # filter missing samples
    row_idx = np.array([i for i in range(m.shape[0]) if not np.all(np.isnan(m[i, :]))])
    col_idx = np.array([i for i in range(m.shape[1]) if not np.all(np.isnan(m[:, i]))])
    filtered = m[row_idx[:, None], col_idx]
    if filtered.shape[0] != filtered.shape[1]:
        return False
    return np.allclose(filtered, filtered.T, atol=tol, equal_nan=equal_nan)


def check_categories(a: np.ndarray):
    """ Check categories in data"""
    return list(np.unique(a[~np.isnan(a)]))


def load_npz(file_path: str):
    """ Load data from .npz file

    Args:
        file_path (str): path to .npz file

    Returns:
        dictionary with arrays as values and their indices used during saving to .npz file as keys

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("File {} does not exist".format(file_path))

    not_npz = False
    if not os.path.isfile(file_path):
        not_npz = True
    else:
        try:
            npz = np.load(file_path, allow_pickle=True)
        except OSError:
            not_npz = True

    if not_npz:
        raise ValueError("Failed to load file - file {} was saved incorrectly or is not an .npz file".format(file_path))

    return {i: npz[i] for i in npz.files}


def save_arrays_to_npz(data: Union[dict, list], file_path: str):
    """ Save numpy arrays to .npz file

    Args:
        data (dict/list): list of numpy arrays or dictionary with specified keywords for every array
        file_path (str): optional path to output file

    """
    arrays = list(data.values()) if isinstance(data, dict) else data
    if not all([isinstance(arrays[i], np.ndarray) for i in range(len(arrays))]):
        raise ValueError("Incorrect data arrays")

    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_path += ".npz" if ".npz" != file_path[-4:] else ""

    if isinstance(data, dict):
        np.savez(file=file_path, **data)
    else:
        args = {str(i): arrays[i] for i in range(len(arrays))}
        np.savez(file=file_path, **args)


def plot_heatmap_seaborn(a: np.ndarray, labels: np.ndarray = None, title: str = None, file_path: str = None):
    assert check_matrix_symmetry(a)

    plt.figure()
    ax = plt.axes()

    if labels is not None:
        rcParams.update({'figure.autolayout': True})
        assert a.shape[0] == labels.shape[0]
        a = pd.DataFrame(a, columns=labels, index=labels)
    p = sns.heatmap(a, vmin=np.nanpercentile(a, 1), vmax=np.nanpercentile(a, 99), cmap="coolwarm", ax=ax)

    if title:
        ax.set_title(title)

    if not file_path:
        plt.show()
    else:
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fig = p.get_figure()
        fig.savefig(file_path)


def plot_line(x: list, y: list, xlabel="x", ylabel="y", title="", file_path: str = None):
    """ Create line plot from vectors of x and y values """
    fig = plt.figure()

    plt.xticks(x)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if not file_path:
        plt.show()
    else:
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fig.savefig(file_path)
        plt.close()


def extract_ncut(a: np.ndarray, k: int):
    """ Select clusters using normalized cut based on graph similarity matrix

    Args:
        a (Numpy.ndarray): symmetric similarity matrix
        k (int): number of clusters

    Returns:
        one dimensional array containing clusters ids for every node
    """
    assert check_matrix_symmetry(a)

    d = np.diag(np.power(np.sum(a, axis=1) + np.spacing(1), -0.5))
    u, s, vh = np.linalg.svd(np.eye(a.shape[0]) - d @ a @ d)

    k = min(u.shape[1], k)
    v = u[:, u.shape[1] - k:]

    kmeans = KMeans(n_clusters=k).fit(v)
    return kmeans.labels_


def extract_max_value(h: np.ndarray):
    """ Select clusters based on maximum value in feature matrix H for every sample/row

    Args:
        h (Numpy.ndarray): feature matrix from optimization algorithm run of shape (n,k), where 'n' is a number of nodes
            and 'k' is a number of clusters

    Returns:
        one dimensional array containing clusters ids for every node

    """
    return np.argmax(h, axis=1)


def extract_spectral(h: np.ndarray, assign_labels: str = "kmeans", n_neighbors: int = 10, n_clusters: int = None):
    """ Select clusters using spectral clustering of feature matrix H

    Args:
        h (Numpy.ndarray): feature matrix from optimization algorithm run of shape (n,k), where 'n' is a number of \
            nodes and 'k' is a number of clusters
        assign_labels : {'kmeans', 'discretize'}, strategy to use to assign labels in the embedding space
        n_neighbors (int): number of neighbors to use when constructing the affinity matrix
        n_clusters (int): number of clusters, if not set use number of columns of 'a'

    Returns:
        one dimensional array containing clusters ids for every node

    """
    if not n_clusters:
        n_clusters = h.shape[1]
    if not n_neighbors:
        n_neighbors = 10
    args = {"n_clusters": n_clusters, "assign_labels": assign_labels, 'n_init': 100,
            "affinity": 'nearest_neighbors', 'n_neighbors': n_neighbors}

    with warnings.catch_warnings():
        warnings.filterwarnings('error', message="Graph is not fully connected")
        clustering = None

        while n_neighbors <= h.shape[0]:
            try:
                clustering = SpectralClustering(**args).fit(h)
                break
            except UserWarning:
                args['n_neighbors'] += 1

        if not clustering:
            clustering = SpectralClustering(**args).fit(h)

    return clustering.labels_


def purity(cl: np.ndarray, org: np.ndarray):
    """ Clustering accuracy measure representing percentage of total number of nodes classified correctly """
    assert cl.shape == org.shape

    acc = 0
    for label in np.unique(cl):
        labels = {}
        for node in range(len(org)):
            if cl[node] == label:
                if org[node] not in labels.keys():
                    labels[org[node]] = 0
                labels[org[node]] += 1
        acc += max(labels.values()) if labels.keys() else 0
    return acc / len(org)


def adjusted_rand_index(cl: np.ndarray, org: np.ndarray):
    """ Clustering accuracy measure calculated by considering all pairs of samples and counting pairs
        that are assigned in the same or different clusters in the predicted and true clusterings """
    assert cl.shape == org.shape

    return adjusted_rand_score(org, cl)  # (labels_true, labels_pred)


def normalized_mutual_information(cl: np.ndarray, org: np.ndarray):
    """ Clustering accuracy measure, which takes into account mutual information between two clusterings and entropy
        of each cluster """
    assert cl.shape == org.shape

    return mutual_info_score(org, cl) / (abs(entropy(cl) + entropy(org)) / 2)


@docstring_formatter(metrics=CLUSTER_METRICS)
def check_accuracy(cl: np.ndarray, org: np.ndarray, method="purity"):
    """ Check clustering accuracy

    Args:
        cl (Numpy.ndarray): one dimensional array containing computed clusters ids for every node
        org (Numpy.ndarray): one dimensional array containing true classes ids for every node
        method (str): accuracy assessment function from {metrics}

    """
    methods = {"NMI": normalized_mutual_information, "purity": purity, "ARI": adjusted_rand_index}
    assert all([m in methods.keys() for m in CLUSTER_METRICS])
    if method not in CLUSTER_METRICS:
        raise ValueError("Could not find selected method")

    if cl.shape != org.shape or cl.ndim != 1 or org.ndim != 1:
        raise ValueError("Incorrect dimensions of arrays")

    return methods[method](cl, org)


def is_standardized(a: np.ndarray, axis: int = 1, atol: float = 1e-8):
    """ Check if matrix values are standardized (have mean equal 0 and standard deviation equal 1)

    Args:
        a (Numpy.ndarray):feature matrix
        axis: either 0 (column-wise standardization) or 1 (row-wise standardization)
        atol (float): absolute tolerance

    Returns:

    """
    if axis not in [0, 1]:
        raise ValueError("Incorrect value of axis, expected either 0 or 1, got {} instead".format(axis))

    return np.allclose(np.nanmean(a, axis=axis), 0, atol=atol) and np.allclose(np.nanstd(a, axis=axis), 1, atol=atol)
