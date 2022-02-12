from operator import add
from random import shuffle
from sumo.modes.run.solver import SumoSolver, SumoNMFResults
from sumo.modes.run.utils import svd_si_init
from sumo.network import MultiplexNet
from sumo.utils import get_logger
import numpy as np


class SupervisedSumoNMF(SumoSolver):
    """ Supervised SUMO solver (A(i)=HS(i)H^T formulation) """

    def __init__(self, graph: MultiplexNet, priors: np.ndarray, nbins: int, bin_size: int = None, rseed: int = None):

        super(SupervisedSumoNMF, self).__init__(graph=graph, nbins=nbins, bin_size=bin_size, rseed=rseed)

        self.priors = priors

        # create average adjacency matrix
        self.avg_adj = self.calculate_avg_adjacency()

        # create sample bins
        self.bins = self.create_sample_bins()

        # layer impact balancing parameters
        self.lambdas = [(1. / samples.shape[0]) ** 2 for samples in self.graph.samples]

    def _get_bins(self, label=None) -> list:
        """ Returns list of bins, each in form of a list containing sample indices, randomly sampled from
        set of samples of given prior label. Maintains the ratio of labelled and unlabelled samples in the dataset. """
        # separate samples with given prior label between bins
        # (this make sure that all samples will be classified at least once)
        if label is not None:
            sample_ids = list(np.where(self.priors[:, label] == 1)[0])
        else:
            sample_ids = list(np.where(np.sum(self.priors, axis=1) == 0)[0])

        prop = round(len(sample_ids) * (self.bin_size / self.graph.nodes))
        shuffle(sample_ids)

        bins_label = [sample_ids[i::self.nbins] for i in range(self.nbins)]
        for i in range(self.nbins):
            # add random samples with given prior label (without doubles, while maintaining prior ratio in dataset)
            ms = prop - len(bins_label[i])  # number of samples to add
            remaining = list(set(sample_ids) - set(bins_label[i]))
            if ms < 0:
                # this should not happen due to allowed ranges set for '-n' and '-subsample' parameters
                raise ValueError("Prior label ratio in bins is disturbed!")
            bins_label[i] = bins_label[i] + list(np.random.choice(remaining, size=ms, replace=False))

        return bins_label

    def _get_bins_unlabelled(self) -> list:
        """ Wrapper around bins generation for unlabelled samples """
        return self._get_bins(label=None)

    def _get_bins_labelled(self) -> list:
        """ Wrapper around bins generation for all samples, with known prior labels"""
        bins = [[] for x in range(self.nbins)]
        for label in range(self.priors.shape[1]):
            bins = list(map(add, bins, self._get_bins(label=label)))
        return bins

    def create_sample_bins(self) -> list:
        """  Separate samples randomly into bins of set size, while making sure that each sample is allocated
        in at least one bin and each prior label in represented equally.

        Returns: list of arrays containing indices of samples allocated to the bin

        """
        if any([x is None for x in [self.graph, self.nbins, self.bin_size, self.priors]]):
            raise ValueError("Solver parameters not set!")
        return [np.array(x) for x in map(add, self._get_bins_unlabelled(), self._get_bins_labelled())]

    def factorize(self, sparsity_penalty: float, k: int, max_iter: int = 500, tol: float = 1e-5, calc_cost: int = 1,
                  logger_name: str = None, bin_id: int = None) -> SumoNMFResults:
        """ Run tri-factorization

        Args:
            sparsity_penalty (float): 'η' value, corresponding to sparsity penalty for H
            k (int): expected number of clusters
            max_iter (int): maximum number of iterations
            tol (float): if objective cost function value fluctuation is smaller than 'stop_val', \
                stop iterations before reaching max_iter
            calc_cost (int): number of steps between every calculation of objective cost function
            logger_name (str): name of existing logger object, if not supplied new main logger is used
            bin_id (int): id of sample bin created in SumoNMF constructor (default of None, means clustering \
                all samples instead of samples in given bin)

        Returns:
            SumoNMFResults object (with result feature matrix / soft cluster indicator matrix (H array),
            and the list of result S matrices for each graph layer)
        """
        self.logger = get_logger(logger_name)

        eps = np.spacing(1)  # epsilon

        if k > self.graph.nodes:
            raise ValueError("Expected number of clusters greater than number of nodes in graph - expected k << nodes!")

        if self.priors.shape[1] > k:
            raise ValueError(
                "Number of unique labels found in the labels file higher then expected number of clusters (k)!")

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

        # initialize H0
        h0 = np.zeros((sample_ids.shape[0], k))
        h0[:, :self.priors.shape[1]] = self.priors[sample_ids, :]

        # initialize H based on H0
        h = np.random.uniform(size=h0.shape)
        h[np.any(h0, axis=1), :] = h0[np.any(h0, axis=1), :]

        # initialize D & V
        v = np.zeros((h.shape[0], h.shape[0]))
        np.fill_diagonal(v, np.any(h0, axis=1) * 1)
        d = v.copy()

        # objective function
        objval = np.zeros((max_iter + 1, self.graph.layers + 2))
        step = 0
        step_recorded = 0
        stop = np.inf
        before_val = np.inf
        best_cost = np.inf
        best_result = (0, h, s)  # step, H, S(i)

        while step <= max_iter and stop > tol:

            # update d
            num = v @ h @ h0.T
            den = v @ d @ h0 @ h0.T
            d = d * ((num + eps) / (den + eps))

            # update s matrices
            for i in range(len(s)):
                num = h.T @ wa[i] @ h
                den = h.T @ (w[i] * (h @ s[i] @ h.T)) @ h
                s[i] = s[i] * ((num + eps) / (den + eps))

            # update h
            num, den = np.zeros((sample_ids.size, k)), np.zeros((sample_ids.size, k))
            for i in range(len(s)):
                num = num + ((self.lambdas[i] * wa[i]) @ h @ s[i])
                den = den + (self.lambdas[i] * (w[i] * (h @ s[i] @ h.T)) @ h @ s[i])
            h = h * (((2 * num + (sparsity_penalty ** 2) * (v @ d @ h0)) + eps) / (
                    (2 * den + (sparsity_penalty ** 2) * (v @ h)) + eps))

            if step % calc_cost == 0 or step == max_iter:

                for i in range(len(s)):
                    objval[step_recorded, i] = self.lambdas[i] * np.sum(
                        w[i] * (np.nansum(np.dstack((a[i], -(h @ (s[i] @ h.T)))), 2)) ** 2)
                objval[step_recorded, self.graph.layers] = sparsity_penalty * np.sum((v @ (h - d @ h0)) ** 2)
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

        return SumoNMFResults(graph=self.graph, h=h, s=s, objval=objval, steps=step - 1, logger=self.logger,
                              sample_ids=sample_ids, d=d, h0=h0)
