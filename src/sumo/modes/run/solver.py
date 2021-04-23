from math import sqrt
from numba import njit
from random import shuffle, seed
from sumo.constants import CLUSTER_METHODS
from sumo.network import MultiplexNet
from sumo.utils import extract_max_value, extract_spectral, get_logger, check_matrix_symmetry
import numpy as np


def svdEM(a: np.ndarray, tol=0.001, max_iter=100, logger_name: str = None):
    """ Approximate SVD on matrix with missing values in matrix using expectation-maximization algorithm

    Args:
        a (Numpy.ndarray): non-negative similarity matrix (n x n) with missing values
        tol (float): convergence tolerance threshold (default of 0.001)
        max_iter (int): maximum number of iterations (default of 100)
        logger_name (str): name of existing logger object, if not supplied new main logger is used

    Returns:
        a_hat (Numpy.ndarray): non-negative similarity matrix (n x n) with imputed values
    """
    logger = get_logger(logger_name)
    logger.info("#SVD-EM for imputation of missing values in average adjacency matrix")

    mu_hat_rows = np.nanmean(a, axis=0, keepdims=True)
    mu_hat_cols = np.nanmean(a, axis=1, keepdims=True)
    valid = np.isfinite(a)
    a_hat = (np.where(valid, a, mu_hat_rows) + np.where(valid, a, mu_hat_cols)) / 2
    mu_hat = (mu_hat_rows + mu_hat_cols) / 2
    logger.info("- Initialized {} missing sample-sample similarities (~{}%)".format(
        int(np.sum(~valid) / 2), round(np.sum(~valid) / np.size(a) * 100), 5))

    stop_iter = False
    step = 0
    v_prev = 0

    while not stop_iter:
        u, s_vec, v = np.linalg.svd(a_hat - mu_hat)
        # impute missing values
        a_hat[~valid] = (u @ np.diag(s_vec) @ v + mu_hat)[~valid]
        # update bias parameter
        mu_hat_rows = np.nanmean(a_hat, axis=0, keepdims=True)
        mu_hat_cols = np.nanmean(a_hat, axis=1, keepdims=True)
        mu_hat = (mu_hat_rows + mu_hat_cols) / 2
        # check convergence
        v = s_vec.sum()
        diff = (v - v_prev + np.spacing(1)) / (v_prev + np.spacing(1))
        logger.info(" - Iteration({}):\tRelative change in trace norm: {}".format(step, round(diff, 4)))
        if step >= max_iter or diff < tol:
            stop_iter = True
        step += 1
        v_prev = v

    return a_hat


def svd_si_init(ai: np.ndarray, k: int):
    """ Initialize S(i) values based on A(i) matrix SVD

    Args:
        ai (Numpy.ndarray): symmetric similarity matrix A(i) (n x n)
        k (int): rank of computed factor

    Returns:
        si (Numpy.ndarray): non-negative matrix S(i) (k x k)
    """
    if not check_matrix_symmetry(ai):
        raise ValueError("Non symmetric A(i) matrix")

    _, s_vec, _ = np.linalg.svd(ai)

    si = np.random.uniform(size=(k, k)) / 100
    si = (si + si.T) * 0.5  # symmetrize
    np.fill_diagonal(si, abs(s_vec[:k]))

    return si


def svd_h_init(a: np.ndarray, k: int):
    """ Initialize H matrix values based on A matrix SVD

        Args:
            a (Numpy.ndarray): symmetric similarity matrix A (n x n)
            k (int): rank of computed factor

        Returns:
            h (Numpy.ndarray): non-negative matrix H (n x k)
    """
    if not check_matrix_symmetry(a):
        raise ValueError("Non symmetric A(i) matrix")

    u, _, _ = np.linalg.svd(a)

    h = np.zeros((a.shape[0], k))
    for i in range(k):
        h[:, i] = u[:, i] - min(u[:, i]) + np.spacing(1)
    h = h / sqrt(np.sum(h ** 2))
    return h


class SumoNMFResults:
    """
    Wrapper class for SumoNMF factorization results
    """

    def __init__(self, graph: MultiplexNet, h: np.ndarray, s: list, objval: np.ndarray, steps: int,
                 sparsity_penalty: float, k: int, logger, sample_ids: np.ndarray):
        self.graph = graph
        self.h = h
        self.s = s
        self.delta_cost = objval
        self.final_cost = objval[-1, -1]
        self.steps = steps
        self.sparsity = sparsity_penalty
        self.k = k
        self.labels = None  # cluster labels for every node
        self.connectivity = None  # samples x samples with 1 if pair of samples is in the same cluster, 0 otherwise
        self.RE = np.sum(self.delta_cost[-1, :self.graph.layers])  # residual error
        self.logger = logger
        self.sample_ids = sample_ids

    def extract_clusters(self, method: str):
        """ Extract cluster labels using selected method

        Args:
            method (str): either "max_value" for extraction based on maximum value in each row of h matrix \
                or "spectral" for spectral clustering on h matrix values

        """
        if method not in CLUSTER_METHODS:
            raise ValueError(
                'Incorrect method of cluster extraction - supported methods: {}'.format(CLUSTER_METHODS))
        elif method == "max_value":
            # normalize H column-wise
            means = np.mean(self.h, axis=0)
            sds = np.std(self.h, axis=0)
            h = (self.h - means) / sds
            self.labels = extract_max_value(h)

            if np.unique(self.labels).size != self.h.shape[1]:
                self.logger.info('Number of clusters extracted from H matrix is different then expected (k)!')
        else:
            self.labels = extract_spectral(self.h)

        labels = np.zeros(self.h.shape)
        labels[np.arange(self.h.shape[0])[:, None], np.array([self.labels]).T] = 1
        self.connectivity = np.zeros((self.graph.nodes, self.graph.nodes))
        self.connectivity[self.sample_ids, self.sample_ids[:, None]] = labels @ labels.T


class SumoNMF:
    """
    Wrapper class for NMF algorithm.

    Args:
        | graph (MultiplexNet): network object, containing data about connections between nodes in each layer \
            in form of adjacency matrices
        | nbins (int): number of bins, to distribute samples into
        | bin_size (int): size of bin, if None set to number of samples

    """

    def __init__(self, graph: MultiplexNet, nbins: int, bin_size: int = None, rseed: int = None):

        if rseed is not None:
            np.random.seed(rseed)
            seed(rseed)

        if not isinstance(graph, MultiplexNet):
            raise ValueError("Unrecognized graph object")

        if bin_size is None:
            bin_size = graph.sample_names.size

        if nbins <= 0 or bin_size > graph.nodes:
            # This should never happen due to creation of SumoNMF objects in sumo 'run'
            raise ValueError("Incorrect number of bins or bin size")

        self.graph = graph

        # create average adjacency matrix
        self.avg_adj = np.zeros((self.graph.nodes, self.graph.nodes))
        connections = self.graph.connections
        connections[connections == 0] = np.nan

        for a in self.graph.adj_matrices:
            self.avg_adj = np.nansum(np.dstack((a, self.avg_adj)), 2)
        self.avg_adj = self.avg_adj / self.graph.connections

        # impute NaN values in average adjacency with SVD-EM algorithm
        if np.sum(np.isnan(self.avg_adj)) > 0:
            self.avg_adj = svdEM(self.avg_adj)

        # layer impact balancing parameters
        self.lambdas = [(1. / samples.shape[0]) ** 2 for samples in self.graph.samples]

        # create sample bins
        sample_ids = list(range(self.graph.nodes))
        shuffle(sample_ids)
        self.bins = [sample_ids[i::nbins] for i in range(nbins)]
        for i in range(len(self.bins)):
            ms = bin_size - len(self.bins[i])
            self.bins[i] = np.array(sorted(self.bins[i] + list(
                np.random.choice(list(set(sample_ids) - set(self.bins[i])), size=ms, replace=False))))  # TODO: add test

        self.logger = get_logger()

    def factorize(self, sparsity_penalty: float, k: int, max_iter: int = 500, tol: float = 1e-5, calc_cost: int = 1,
                  h_init: int = None, logger_name: str = None, bin_id: int = None):
        """ Run tri-factorization

        Args:
            sparsity_penalty (float): 'η' value, corresponding to sparsity penalty for H
            k (int): expected number of clusters
            max_iter (int): maximum number of iterations
            tol (float): if objective cost function value fluctuation is smaller than 'stop_val', \
                stop iterations before reaching max_iter
            calc_cost (int): number of steps between every calculation of objective cost function
            h_init (int): index of adjacency matrix to use for H matrix initialization or None for initialization \
                using average adjacency
            logger_name (str): name of existing logger object, if not supplied new main logger is used
            bin_id (int): id of sample bin created in SumoNMF constructor (default of None, means clustering \
                all samples instead of samples in given bin)

        Returns:
            h (Numpy.ndarray): result feature matrix / soft cluster indicator matrix
            s: list of result S matrices for each graph layer
        """
        self.logger = get_logger(logger_name)

        eps = np.spacing(1)  # epsilon

        if k > self.graph.nodes:
            raise ValueError("Expected number of clusters greater than number of nodes in graph - expected k << nodes!")

        if bin_id is None:
            sample_ids = np.arange(self.graph.nodes)
        else:
            sample_ids = self.bins[bin_id]

        # filter missing samples
        for layer in self.graph.adj_matrices:
            layer[np.isnan(layer)] = 0.

        a = np.array([self.graph.adj_matrices[i][sample_ids, sample_ids[:, None]] for i in range(self.graph.layers)])
        w = np.array([self.graph.w[i][sample_ids, sample_ids[:, None]] for i in range(self.graph.layers)])
        wa = np.array([w[i] * a[i] for i in range(self.graph.layers)])

        # randomize S matrices for each layer
        s = np.array([svd_si_init(self.graph.adj_matrices[i], k) for i in range(self.graph.layers)])

        # randomize feature matrix / soft cluster indicator matrix
        h = svd_h_init(self.graph.adj_matrices[h_init] if h_init is not None else self.avg_adj, k)[sample_ids, :]

        # objective function
        objval = np.zeros((max_iter + 1, self.graph.layers + 2))
        step = 0
        step_recorded = 0
        stop = np.inf
        before_val = np.inf
        best_cost = np.inf
        best_result = (0, h, s)  # step, H, S(i)

        while step <= max_iter and stop > tol:

            # update s matrices
            for i in range(len(s)):
                num = h.T @ wa[i] @ h
                den = h.T @ (w[i] * (h @ s[i] @ h.T)) @ h
                s[i] = s[i] * ((num + eps) / (den + eps))

            # update h
            num, den = np.zeros((sample_ids.size, k)), np.zeros((sample_ids.size, k))
            for i in range(len(s)):
                num = num + ((self.lambdas[i] * wa[i]) @ h @ s[i])
                den = den + (self.lambdas[i] * (w[i] * (h @ s[i] @ h.T)) @ h @ s[i] + 0.5 * sparsity_penalty * h)
            h = h * ((num + eps) / (den + eps))

            if step % calc_cost == 0 or step == max_iter:

                for i in range(len(s)):
                    objval[step_recorded, i] = self.lambdas[i] * np.sum(
                        w[i] * (np.nansum(np.dstack((a[i], -(h @ (s[i] @ h.T)))), 2)) ** 2)
                objval[step_recorded, self.graph.layers] = sparsity_penalty * np.sum(h ** 2)
                objval[step_recorded, - 1] = np.sum(objval[step_recorded, :])

                after_val = objval[step_recorded, - 1]
                stop = abs(after_val - before_val) / after_val

                if objval[step_recorded, -1] <= best_cost:
                    best_cost = objval[step_recorded, -1]
                    best_result = (step, h.copy(), s.copy())

                if step == 0:
                    self.logger.info("Initial ℒ/Δℒ: {}\t[{} + {}]".format(round(objval[step_recorded, -1], 6),
                                                                          round(np.sum(objval[step_recorded, :-2]), 6),
                                                                          round(objval[step_recorded, -2], 6)))
                else:
                    # objective function value decreases
                    self.logger.info("Step({}),\t ℒ: {}\t[{} + {}]".format(step, round(objval[step_recorded, -1], 6),
                                                                           round(np.sum(objval[step_recorded, :-2]), 6),
                                                                           round(objval[step_recorded, -2], 6)))
                before_val = after_val
                step_recorded += 1

            step += 1

        if stop < tol:
            # stop condition was achieved
            self.logger.info("Stop condition for iterations achieved (|Δℒ| < |{}|).".format(tol))
        else:
            # maximum iterations was reached
            self.logger.info("Maximum iterations ({}) reached".format(max_iter))

        objval = objval[:step_recorded, :]
        np.set_printoptions(threshold=np.inf)
        self.logger.info("#Best achieved results:")

        for i in range(len(best_result[2])):
            self.logger.debug("- Final S({}):\n{}".format(i, best_result[2][i]))
        self.logger.debug("- Final H:\n{}".format(best_result[1]))
        self.logger.info("- Final ℒ: {} ({}) [step: {}]".format(round(objval[-1, -1], 6),
                                                                round(np.sum(objval[-1, :-2]), 6),
                                                                best_result[0]))

        return SumoNMFResults(self.graph, h, s, objval, step - 1, sparsity_penalty, k, self.logger, sample_ids)
