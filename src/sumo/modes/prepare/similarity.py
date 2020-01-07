from math import sqrt
from scipy.stats import pearsonr, spearmanr
from sumo.constants import CORR_METHODS, VAR_TYPES
import numpy as np


def euclidean_dist(a: np.ndarray, b: np.ndarray, missing=0.1):
    """ Calculate euclidean distance between two vectors of continuous variables """
    assert a.shape == b.shape
    threshold = a.shape[0] * missing
    eps = np.spacing(1)
    vec = np.power(a - b, 2)
    common = ~np.isnan(vec)  # common / not missing values
    dist = (sqrt(np.sum(vec[common])) + eps) / (np.sum(common) + eps) if np.sum(common) >= threshold else np.nan
    assert np.isnan(dist) or (1 >= dist >= 0)
    return dist


def chi_squared_dist(a: np.ndarray, b: np.ndarray, missing=0.1):
    """ Calculate chi-squared distance between two vectors of categorical variables """
    assert a.shape == b.shape
    threshold = a.shape[0] * missing
    values = ~np.logical_or(np.isnan(b), np.isnan(a))  # find missing values in either of vectors
    avec = a[values]
    bvec = b[values]
    categories = np.unique(np.concatenate((avec, bvec)))
    if avec.size < threshold:
        return np.nan
    # create profiles of samples
    counts = np.zeros((2, categories.size))
    for i in range(categories.size):
        counts[0, i] = np.sum(avec == categories[i])
        counts[1, i] = np.sum(bvec == categories[i])
    profiles = counts / np.sum(counts, axis=1)[:, None]

    total = avec.size + bvec.size
    barycenter = np.sum(counts, axis=0) / total
    weights = np.sqrt(1 / barycenter)
    R = profiles - barycenter  # deviations from barycenter
    weightedR = R * weights

    cov = weightedR @ weightedR.T
    dist = (np.diag(cov) - cov) + (np.diag(cov) - cov).T
    assert 1 >= dist[0, 1] >= 0
    return dist[0, 1]


def agreement_dist(a: np.ndarray, b: np.ndarray, missing=0.1):
    """ Calculate agreement-based distance between two vectors of binary variables """
    assert a.shape == b.shape
    threshold = a.shape[0] * missing
    eps = np.spacing(1)
    # ignore uninformative positions and missing values in either of vectors
    na_values = ~np.logical_or(np.isnan(b), np.isnan(a))
    informative = ~np.logical_and((a == np.zeros(a.shape[0])), (b == np.zeros(b.shape[0])))
    indices = np.logical_and(informative, na_values)
    avec = a[indices]
    bvec = b[indices]
    if avec.size < threshold:
        return np.nan
    dist = 1 - (list(avec == bvec).count(True) + eps) / (avec.size + eps)
    assert 1 >= dist >= 0
    return dist


def binomial_dist(a: np.ndarray, b: np.ndarray, missing=0.1):
    """ Calculate agreement-based distance between two vectors of binary variables using binomial distance"""
    assert a.shape == b.shape
    threshold = a.shape[0] * missing
    na_values = ~np.logical_or(np.isnan(b), np.isnan(a))
    indices = na_values
    avec = a[indices]
    bvec = b[indices]
    if avec.size < threshold:
        return np.nan
    categories = np.unique(np.concatenate((avec, bvec), axis=0))
    # if categories.size == 1:
    #     return 0
    # assert categories.size == 2
    dist = 0
    for i in categories:
        ai = np.sum(avec == i)
        bi = np.sum(bvec == i)
        ni = ai + bi
        logni = np.log(ni)
        t1 = ai * (np.log(ai) - logni) if ai > 0 else 0
        t2 = bi * (np.log(bi) - logni) if bi > 0 else 0
        dist += (t1 * t2 - ni * np.log(1 / 2)) / ni

    # assert 1 >= dist >= 0

    return dist


def pearson_corr(a: np.ndarray, b: np.ndarray):
    pcor = pearsonr(a, b)
    return pcor if pcor[0] >= 0 else (0, np.nan)


def cosine_sim(a: np.ndarray, b: np.ndarray):
    from scipy.spatial.distance import cosine
    sim = 1 - cosine(a, b)
    return sim if sim >= 0 else 0


def corr(a: np.ndarray, b: np.ndarray, method="pearson", missing=0.1):
    """ Calculate similarity between two vectors"""
    assert a.shape == b.shape
    assert method in CORR_METHODS
    threshold = int(a.shape[0] * missing)
    values = ~np.logical_or(np.isnan(b), np.isnan(a))  # find missing values in either of vectors
    avec = a[values]
    bvec = b[values]

    if method == "cosine":
        return cosine_sim(avec, bvec) if avec.size >= threshold else np.nan
    else:
        cor_func = pearson_corr if method == "pearson" else spearmanr
        return cor_func(avec, bvec)[0] if avec.size >= threshold else np.nan


def feature_corr_similarity(f: np.ndarray, missing: float = 0.1, method="pearson"):
    """ Generate similarity matrix from genomic assay based on correlation between samples

        Args:
            f (Numpy.ndarray): Feature matrix (n x k, where 'n' - samples, 'k' - measurements)
            missing (float): acceptable fraction of values for assessment of distance/similarity between two samples \
            (default of 0.1, means that up to 90 % of missing values is acceptable)
            method (str): either "pearson" or "spearman"
        Returns:
            sim (Numpy.ndarray): symmetric matrix describing similarity between samples (n x n)

    """
    assert method in CORR_METHODS
    # filter missing samples
    samples = np.array([i for i in range(f.shape[0]) if not np.all(np.isnan(f[i, :]))])

    # euclidean distance matrix
    sim = np.zeros((f.shape[0], f.shape[0]))
    sim[:] = np.nan

    for i in [sample for sample in range(f.shape[0]) if sample in samples]:
        for j in [sample for sample in range(i, f.shape[0]) if sample in samples]:
            sim[i, j] = corr(f[i, :], f[j, :], method=method, missing=missing)
            if i != j:
                sim[j, i] = sim[i, j]

    vals = sim[~np.isnan(sim)]
    vals[vals < 0] = 0
    sim[~np.isnan(sim)] = vals
    return sim


def feature_to_adjacency(f: np.ndarray, variable_type: str, n: float = 0.1, missing: float = 0.1, alpha: float = 0.5):
    """ Generate similarity matrix from genomic assay, partially based on:

        Wang, B., Mezlini, A. M., Demir, F., Fiume, M., Tu, Z., Brudno, M., … Goldenberg, A. (2014).
        Similarity network fusion for aggregating data types on a genomic scale.
        Nature Methods, 11(3), 333–337.

    Args:
        f (Numpy.ndarray): Feature matrix (n x k, where 'n' - samples, 'k' - measurements)
        variable_type (str): either 'continuous', 'categorical' or 'binary', value indicating how to calculate distance \
            between two samples ('continuous' - using Euclidean distance, 'categorical' - chi-squared distance, \
            'binary' - agreement-based measure)
        n (float): fraction of nearest neighbours to use for samples similarity calculation
        missing (float): acceptable fraction of values for assessment of distance/similarity between two samples \
            (default of 0.1, means that up to 90 % of missing values is acceptable)
        alpha (float): hyperparameter

    Returns:
        w (Numpy.ndarray): symmetric matrix describing similarity between samples (n x n)

    """
    if alpha <= 0:
        raise ValueError("Incorrect value of hyperparameter")

    if variable_type not in VAR_TYPES:
        raise ValueError("Incorrect value of 'variable_type'")
    elif variable_type == 'continuous':
        dist_calc = euclidean_dist
    elif variable_type == 'categorical':
        dist_calc = chi_squared_dist
    elif variable_type == 'binomial':
        dist_calc = binomial_dist
    else:
        dist_calc = agreement_dist
    # NOTE: for every method distance is calculated only over features that are not missing for every pair of samples

    # filter missing samples
    samples = np.array([i for i in range(f.shape[0]) if not np.all(np.isnan(f[i, :]))])
    k = int(round(samples.shape[0] * n, 0))

    # euclidean distance matrix
    dist = np.zeros((f.shape[0], f.shape[0]))
    dist[:] = np.nan

    for i in [sample for sample in range(f.shape[0]) if sample in samples]:
        for j in [sample for sample in range(i, f.shape[0]) if sample in samples]:
            if i != j:
                dist[i, j] = dist_calc(f[i, :], f[j, :], missing=missing)
                dist[j, i] = dist[i, j]
            else:
                dist[i, j] = 0

    # weighted edges matrix
    w = np.zeros((f.shape[0], f.shape[0]))
    w[:] = np.nan

    sample_dist = dist[samples[:, None], samples]  # select only available samples
    sample_w = np.zeros((samples.shape[0], samples.shape[0]))

    for i in range(samples.shape[0]):
        for j in range(i, samples.shape[0]):
            num = -1 * (sample_dist[i, j] ** 2)
            neighbours_i = np.sort(sample_dist[i, :])[1:k + 1]
            neighbours_j = np.sort(sample_dist[j, :])[1:k + 1]
            # np.nan means that it is not possible to calculate distance between this sample and any other in layer
            ni_mean = np.nanmean(neighbours_i) if neighbours_i[0] != np.nan else np.nan
            nj_mean = np.nanmean(neighbours_j) if neighbours_j[0] != np.nan else np.nan
            den = alpha * ni_mean * nj_mean
            if np.allclose(den, 0):
                raise ValueError(
                    "Average distance between sample and its all nearest neighbours is 0, please increase" +
                    " fraction of nearest neighbours used for similarity calculation ('-k' parameter)")
            if i != j:
                sample_w[i, j] = np.exp(num / den)
                sample_w[j, i] = sample_w[i, j]
            else:
                sample_w[i, j] = 1.

            w[samples[:, None], samples] = sample_w  # put missing samples back in

    return w
