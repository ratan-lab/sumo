from math import sqrt
from sumo.utils import get_logger, check_matrix_symmetry
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
    if a.shape[0] == 0 or np.isnan(a).sum() == a.size:
        raise ValueError("Incorrect values in a matrix!")

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
