from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sumo.constants import RUN_ARGS, CLUSTER_METHODS, LOG_LEVELS
from sumo.modes.mode import SumoMode
from sumo.modes.run.solver import SumoNMF
from sumo.network import MultiplexNet
from sumo.utils import extract_ncut, load_npz, save_arrays_to_npz, setup_logger, docstring_formatter, \
    plot_heatmap_seaborn
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os

_sumo_run = None


@docstring_formatter(cluster_methods=CLUSTER_METHODS, log_levels=LOG_LEVELS)
class SumoRun(SumoMode):
    """
    Sumo mode for factorization of multiplex network to identify molecular subtypes. Constructor args are set in \
    'run' subparser.

    Args:
        | infile (str): input .npz file containing adjacency matrices for every network layer and sample names \
            (file created by running program with mode "prepare") - consecutive adjacency arrays in file are indexed \
            in following way: "0", "1" ... and index of sample name vector is "samples"
        | k (int): number of clusters
        | outdir (str) path to save output files
        | sparsity (list): list of sparsity penalty values for H matrix (sumo will try different values and select \
            the best results
        | n (int): number of repetitions
        | method (str): method of cluster extraction, selected from {cluster_methods}
        | max_iter (int): maximum number of iterations for factorization
        | tol (float): if objective cost function value fluctuation (|Δℒ|) is smaller than this value, stop iterations \
            before reaching max_iter
        | calc_cost (int): number of steps between every calculation of objective cost function
        | logfile (str): path to save log file, if set to None stdout is used
        | log (str): sets the logging level from {log_levels}
        | h_init (int): index of adjacency matrix to use for H matrix initialization, if set to None average adjacency \
            matrix is used
        | t (int): number of threads

    """

    def __init__(self, **kwargs):
        """ Creates instance od SumoPrepare class

        Args:
            **kwargs: keyword arguments, have to contain all of arguments detailed in class description otherwise
                AttributeError is raised
        """
        super().__init__(**kwargs)

        if not all([hasattr(self, arg) for arg in RUN_ARGS]):
            # this should never happened due to object creation with parse_args in sumo/__init__.py
            raise AttributeError("Cannot create SumoRun object")

        self.graph = None
        self.nmf = None

        # check positional arguments
        if not os.path.exists(self.infile):
            raise FileNotFoundError("Input file not found")
        if self.n < 1:
            raise ValueError("Incorrect value of 'n'")
        if self.t < 1:
            raise ValueError("Incorrect number of threads")

        self.logger = setup_logger("main", self.log, self.logfile)

        if os.path.exists(self.outdir):
            if not os.path.isdir(self.outdir):
                raise NotADirectoryError("{} already exists and is not a directory!".format(self.outdir))
            self.logger.warning("Directory '{}' already exist, some files may be overwritten.".format(self.outdir))
        else:
            os.mkdir(self.outdir)
            self.logger.info("Directory '{}' created".format(self.outdir))

        if len(self.k) > 2 or (len(self.k) == 2 and self.k[0] > self.k[1]):
            raise ValueError("Incorrect range of k values")
        elif len(self.k) == 2:
            self.k = list(range(self.k[0], self.k[1] + 1))

        self.plot_dir = os.path.join(self.outdir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def run(self):
        """ Cluster multiplex network using non-negative matrix tri-factorization """
        self.logger.info("Number of clusters ('k'): {}".format(self.k))
        self.logger.info("Sparsity values ('eta'): {}".format(self.sparsity))

        # load file
        self.logger.info("#Loading file: {}".format(self.infile))
        data = load_npz(self.infile)

        if "samples" not in data.keys():
            raise ValueError("Sample name vector not found, incorrect structure of input file.")
        sample_names = data["samples"]

        adj_matrices = []
        for i in range(len(data.keys())):
            if str(i) in data.keys():
                a = data[str(i)]
                if a.shape[0] != sample_names.shape[0]:
                    raise ValueError(
                        "Number of samples in adjacency matrix and sample name vector does not correspond, " +
                        "incorrect structure of input file.")
                adj_matrices.append(a)
            else:
                break

        self.logger.info("Number of found graph layers: {}".format(len(adj_matrices)))
        if len(adj_matrices) == 0:
            raise ValueError("No adjacency matrices found in input file")

        if self.h_init is not None:
            if self.h_init >= len(adj_matrices) or self.h_init < 0:
                raise ValueError("Incorrect value of h_init")

        # create multilayer graph
        self.graph = MultiplexNet(adj_matrices=adj_matrices, node_labels=sample_names)

        # create solver
        self.nmf = SumoNMF(graph=self.graph)

        global _sumo_run
        _sumo_run = self  # this solves multiprocessing issue with pickling

        # run factorization for every (eta, k)
        cophenet_list = []
        for k in self.k:
            self.logger.debug("#K:{}".format(k))

            if self.t == 1:
                results = [_run_factorization(sparsity=sparsity, k=k, sumo_run=_sumo_run) for sparsity in self.sparsity]
                sparsity_order = self.sparsity
            else:
                self.logger.debug("#{} processes to run".format(len(self.sparsity)))
                pool = mp.Pool(self.t)

                results = []
                sparsity_order = []
                iproc = 1
                for res in pool.imap_unordered(run_thread_wrapper, zip(self.sparsity, [k] * len(self.sparsity))):
                    self.logger.debug("- process {} finished".format(iproc))
                    results.append(res[0])
                    sparsity_order.append(res[1])
                    iproc += 1
                # TODO: implement more comprehensive multiprocessing

            # select best result
            best_result = sorted(results, reverse=True)[0]
            best_eta = None

            quality_output = []
            for (result, sparsity) in zip(results, sparsity_order):
                self.logger.info("Clustering quality (eta={}): {}".format(sparsity, result[0]))
                quality_output.append(np.array([sparsity, result[0]]))
                if result[1] == best_result[1]:
                    best_eta = sparsity

            # summarize results
            assert best_eta is not None
            self.logger.info("#Selected eta: {}".format(best_eta))
            out_arrays = load_npz(best_result[1])
            out_arrays["summary"] = np.array(quality_output)
            cophenet_list.append(out_arrays["cophenet"][0])

            # save to output file
            outfile = os.path.join(self.outdir, "k{}".format(k), "sumo_results.npz")
            self.logger.info("#Results (k = {}) saved to {}".format(k, outfile))
            save_arrays_to_npz(out_arrays, outfile)

            plot_heatmap_seaborn(out_arrays['consensus'], labels=out_arrays['clusters'][:, 0],
                                 title="Consensus matrix (K = {})".format(k),
                                 file_path=os.path.join(self.plot_dir, "consensus_k{}.png".format(k)))
            # TODO: change sample order

        if len(cophenet_list) > 1:
            plt.figure()
            axes = plt.gca()
            axes.set_ylim([0, 1.2])
            plt.plot(self.k, cophenet_list)
            plt.xlabel("K")
            plt.ylabel("cophenetic correlation coefficient")
            plt.title("Cluster stability for different K values")
            plot_path = os.path.join(self.plot_dir, "cophenet.png")
            plt.savefig(plot_path)
            self.logger.info("#Cophentic correlation coefficient plot for different K values has " +
                             "been saved to {}".format(plot_path))


def run_thread_wrapper(args: tuple):
    global _sumo_run
    # this solves multiprocessing issue with pickling
    assert len(args) == 2
    return _run_factorization(sparsity=args[0], k=args[1], sumo_run=_sumo_run), args[0]


def _run_factorization(sparsity: float, k: int, sumo_run: SumoRun):
    """ Run factorization for set sparsity and number of clusters

    Args:
        sparsity (float): value of sparsity penalty
        k (int): number of clusters
        sumo_run: SumoRun object

    Returns:
        quality (float): assessed quality of cluster structure
        outfile (str): path to .npz output file with results of factorization

    """
    k_dir = os.path.join(sumo_run.outdir, "k{}".format(k))
    os.makedirs(k_dir, exist_ok=True)
    log_file = os.path.join(k_dir, "eta_{}.log".format(sparsity))
    outfile = os.path.join(k_dir, "eta_{}.npz".format(sparsity))
    eta_logger = setup_logger("eta{}_logger".format(sparsity), level=sumo_run.log, log_file=log_file)

    # run factorization N times
    results = []
    for repeat in range(sumo_run.n):
        eta_logger.info(
            "#Runing NMF algorithm with sparsity {} (N={})".format(sparsity, repeat + 1))
        opt_args = {
            "sparsity_penalty": sparsity,
            "k": k,
            "max_iter": sumo_run.max_iter,
            "tol": sumo_run.tol,
            "calc_cost": sumo_run.calc_cost,
            "h_init": sumo_run.h_init,
            "logger_name": "eta{}_logger".format(sparsity)
        }

        result = sumo_run.nmf.factorize(**opt_args)
        # extract computed clusters
        eta_logger.info("#Using {} for cluster labels extraction".format(sumo_run.method))
        result.extract_clusters(method=sumo_run.method)
        results.append(result)

    # consensus graph
    assert len(results) > 0
    eta_logger.info("#Creating consensus graph")

    REs = []  # residual errors
    for run_idx in range(sumo_run.n):
        REs.append(results[run_idx].RE)
    minRE, maxRE = min(REs), max(REs)

    consensus = np.zeros((sumo_run.graph.nodes, sumo_run.graph.nodes))
    eps = np.spacing(1)  # epsilon
    weights = []
    for run_idx in range(sumo_run.n):
        weight = (maxRE - results[run_idx].RE + eps) / (maxRE - minRE + eps)
        weights.append(weight)
        consensus += results[run_idx].connectivity * weight

    consensus = consensus / np.sum(weights)
    org_con = consensus.copy()
    consensus[consensus < 0.5] = 0

    # calculate cophenetic correlation coefficient
    dist = pdist(org_con, metric="correlation")
    ccc = cophenet(linkage(dist, method="complete", metric="correlation"), dist)[0]

    eta_logger.info("#Extracting final clustering result, using normalized cut")
    consensus_labels = extract_ncut(consensus, k=k)

    cluster_array = np.empty((sumo_run.graph.sample_names.shape[0], 2), dtype=np.object)
    # TODO add column with confidence value when investigating soft clustering
    cluster_array[:, 0] = sumo_run.graph.sample_names
    cluster_array[:, 1] = consensus_labels

    clusters_dict = {num: sumo_run.graph.sample_names[list(np.where(consensus_labels == num)[0])] for num in
                     np.unique(consensus_labels)}
    for cluster_idx in sorted(clusters_dict.keys()):
        eta_logger.info("Cluster {} ({} samples): \n{}".format(cluster_idx, len(clusters_dict[cluster_idx]),
                                                               clusters_dict[cluster_idx]))

    # calculate quality of clustering for given sparsity
    quality = sumo_run.graph.get_clustering_quality(labels=cluster_array[:, 1])
    # create output file
    out_arrays = {
        "clusters": cluster_array,
        "consensus": consensus,
        "unfiltered_consensus": consensus,
        "quality": np.array(quality),
        "cophenet": np.array([ccc])
    }

    if sumo_run.log == "DEBUG":
        for i in range(len(results)):
            out_arrays["cost{}".format(i)] = results[i].delta_cost[-1, :]
            out_arrays["h{}".format(i)] = results[i].h
            for si in range(len(results[i].s)):
                out_arrays["s{}{}".format(si, i)] = results[i].s[si]

    save_arrays_to_npz(data=out_arrays,
                       file_path=outfile)  # TODO: add detailed output files description in documentation
    eta_logger.info("#Output file {} created".format(outfile))

    return quality, outfile
