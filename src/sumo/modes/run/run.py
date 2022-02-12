from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sumo.constants import RUN_ARGS, CLUSTER_METHODS, LOG_LEVELS
from sumo.modes.mode import SumoMode
from sumo.modes.run.solvers.unsupervised_sumo import UnsupervisedSumoNMF
from sumo.modes.run.solvers.supervised_sumo import SupervisedSumoNMF
from sumo.network import MultiplexNet
from sumo.utils import extract_ncut, load_npz, save_arrays_to_npz, setup_logger, docstring_formatter, \
    plot_heatmap_seaborn, plot_metric, close_logger
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import shutil

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
        | sparsity (list): list of sparsity penalty values for H matrix (if multiple values sumo will try all \
            and select the best results
        | n (int): number of repetitions
        | method (str): method of cluster extraction, selected from {cluster_methods}
        | max_iter (int): maximum number of iterations for factorization
        | tol (float): if objective cost function value fluctuation is smaller than this value, stop iterations \
            before reaching max_iter
        | subsample (float): fraction of samples randomly removed from each run, cannot be greater then 0.5
        | calc_cost (int): number of steps between every calculation of objective cost function
        | logfile (str): path to save log file, if set to None stdout is used
        | log (str): sets the logging level from {log_levels}
        | h_init (int): index of adjacency matrix to use for H matrix initialization, if set to None average adjacency \
            matrix is used
        | t (int): number of threads
        | rep (int): number of times consensus matrix is created for the purpose of assessing clustering quality

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
        self.priors = None

        # check arguments
        if not os.path.exists(self.infile):
            raise FileNotFoundError("Input file not found")
        if self.labels is not None and not os.path.exists(self.labels):
            raise FileNotFoundError("Labels file not found")
        if self.n < 1:
            raise ValueError("Incorrect value of 'n' parameter")
        if self.t < 1:
            raise ValueError("Incorrect number of threads")
        if self.subsample > 0.5 or self.subsample < 0:
            # do not allow for removal of more then 50% of samples in each run
            raise ValueError("Incorrect value of 'subsample' parameter")
        if self.rep < 1:
            # number of times additional consensus matrix will be created
            raise ValueError("Incorrect value of 'rep' parameter")
        if self.seed is not None and self.seed < 0:
            raise ValueError("Seed value cannot be negative")
        self.runs_per_con = max(round(self.n * 0.8), 1)  # number of runs per consensus matrix creation

        self.logger = setup_logger("main", self.log, self.logfile)

        if os.path.exists(self.outdir):
            if not os.path.isdir(self.outdir):
                raise NotADirectoryError("{} already exists and is not a directory!".format(self.outdir))
            self.logger.warning("Directory '{}' already exist and will be overwritten.".format(self.outdir))
            shutil.rmtree(self.outdir)

        else:
            self.logger.info("Creating directory '{}'".format(self.outdir))
        os.mkdir(self.outdir)

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

        self.logger.info("#Number of found graph layers: {}".format(len(adj_matrices)))
        if len(adj_matrices) == 0:
            raise ValueError("No adjacency matrices found in input file")

        if self.h_init is not None:
            if self.h_init >= len(adj_matrices) or self.h_init < 0:
                raise ValueError("Incorrect value of h_init")
            if self.labels is not None:
                self.logger.warning("#Ignoring set h_init value (supervised mode enabled with '-labels' parameter)")

        # read prior sample labels (if set with -labels param)
        if self.labels is not None:
            self.logger.info("#Loading sample labels file: {}".format(self.labels))
            known_labels = pd.read_csv(self.labels, sep="\t")
            if not all([coln in known_labels.columns for coln in ['sample', 'label']]):
                raise ValueError("Incorrect structure of the labels file")
            nsample_labels = known_labels.shape[0]
            known_labels = known_labels[known_labels['sample'].isin(sample_names)]
            if known_labels.shape[0] != nsample_labels:
                self.logger.warning("- samples not available in the {} file: {}".format(self.infile, nsample_labels -
                                                                                        known_labels.shape[0]))
            labels_groups = np.unique(known_labels['label'].values)
            self.logger.info(
                "#Found {} unique sample labels for {}/{} samples".format(labels_groups.size, known_labels.shape[0],
                                                                          sample_names.size))

            self.priors = np.zeros((sample_names.size, labels_groups.size))
            for label_idx in range(labels_groups.size):
                label = labels_groups[label_idx]
                selected_samples = known_labels[known_labels['label'] == label]['sample'].tolist()
                self.priors[[sample in selected_samples for sample in sample_names], label_idx] = 1

        # create multilayer graph
        self.graph = MultiplexNet(adj_matrices=adj_matrices, node_labels=sample_names)
        n_sub_samples = round(sample_names.size * self.subsample)
        self.logger.info(
            "#Number of samples randomly removed in each run: {} out of {}".format(n_sub_samples, sample_names.size))

        # create solver
        if self.priors is None:
            self.nmf = UnsupervisedSumoNMF(graph=self.graph, nbins=self.n, bin_size=self.graph.nodes - n_sub_samples,
                                           rseed=self.seed)
        else:
            self.nmf = SupervisedSumoNMF(graph=self.graph, priors=self.priors, nbins=self.n,
                                         bin_size=self.graph.nodes - n_sub_samples, rseed=self.seed)

        global _sumo_run
        _sumo_run = self  # this solves multiprocessing issue with pickling

        # run factorization for every (eta, k)
        cophenet_list = []
        pac_list = []
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
                self.logger.info("#Clustering quality (eta={}): {}".format(sparsity, result[0]))
                quality_output.append(np.array([sparsity, result[0]]))
                if result[1] == best_result[1]:
                    best_eta = sparsity

            # summarize results
            assert best_eta is not None
            self.logger.info("Selected eta: {}".format(best_eta))
            out_arrays = load_npz(best_result[1])

            cophenet_list.append(out_arrays["cophenet"])
            pac_list.append(out_arrays["pac"])

            # create text file with cluster labels
            clusters = out_arrays['clusters']
            with open(os.path.join(self.outdir, "k{}".format(k), "clusters.tsv"), 'w') as cl_file:
                cl_file.write("sample\tlabel\n")
                for row_idx in range(clusters.shape[0]):
                    cl_file.write("{}\t{}\n".format(clusters[row_idx, 0], clusters[row_idx, 1]))

            # create symlink to the selected best result
            summary_outfile = os.path.join(self.outdir, "k{}".format(k), "sumo_results.npz")
            if os.path.lexists(summary_outfile):
                # overwriting symlink
                os.remove(summary_outfile)

            workdir = os.getcwd()
            os.chdir(os.path.dirname(best_result[1]))
            os.symlink(os.path.basename(best_result[1]), os.path.basename(summary_outfile))
            os.chdir(workdir)
            assert os.getcwd() == workdir

            self.logger.info("Results (k = {}) saved to {}".format(k, summary_outfile))

            plot_heatmap_seaborn(out_arrays['consensus'], labels=np.arange(self.graph.nodes),
                                 title="Consensus matrix (K = {})".format(k),
                                 file_path=os.path.join(self.plot_dir, "consensus_k{}.png".format(k)))
            # TODO: change sample order

        if len(cophenet_list) > 1 and len(pac_list) > 1:
            cophenet_plot_path = os.path.join(self.plot_dir, "cophenet.png")
            plot_metric(x=self.k, y=cophenet_list, xlabel="K", ylabel="cophenetic correlation coefficient",
                        title="Cluster stability for different K values", file_path=cophenet_plot_path, color="red")
            self.logger.info("#Cophentic correlation coefficient plot for different K values has " +
                             "been saved to {}".format(cophenet_plot_path))

            pac_plot_path = os.path.join(self.plot_dir, "pac.png")
            plot_metric(x=self.k, y=pac_list, xlabel="K", ylabel="PAC",
                        title="Proportion of ambiguous clusterings for different K values", file_path=pac_plot_path,
                        color="blue")
            self.logger.info("#Proportion of ambiguous clusterings plot for different K values has " +
                             "been saved to {}".format(pac_plot_path))


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

    if sumo_run.seed is not None:
        np.random.seed(sumo_run.seed)

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
            "logger_name": "eta{}_logger".format(sparsity),
            "bin_id": repeat
        }
        if sumo_run.priors is None:
            # unsupervised classification
            opt_args["h_init"] = sumo_run.h_init

        result = sumo_run.nmf.factorize(**opt_args)
        # extract computed clusters
        eta_logger.info("#Using {} for cluster labels extraction".format(sumo_run.method))
        result.extract_clusters(method=sumo_run.method)
        results.append(result)

    # consensus graph
    assert len(results) > 0

    all_REs = []  # residual errors
    for run_idx in range(sumo_run.n):
        all_REs.append(results[run_idx].RE)

    out_arrays = {'pac': np.array([]), 'cophenet': np.array([])}
    minRE, maxRE = min(all_REs), max(all_REs)

    for rep in range(sumo_run.rep):
        run_indices = list(np.random.choice(range(len(results)), sumo_run.runs_per_con, replace=False))

        consensus = np.zeros((sumo_run.graph.nodes, sumo_run.graph.nodes))
        weights = np.empty((sumo_run.graph.nodes, sumo_run.graph.nodes))
        weights[:] = np.nan

        all_equal = np.allclose(minRE, maxRE)

        for run_idx in run_indices:
            weight = np.empty((sumo_run.graph.nodes, sumo_run.graph.nodes))
            weight[:] = np.nan
            sample_ids = results[run_idx].sample_ids
            if all_equal:
                weight[sample_ids, sample_ids[:, None]] = 1.
            else:
                weight[sample_ids, sample_ids[:, None]] = (maxRE - results[run_idx].RE) / (maxRE - minRE)

            weights = np.nansum(np.stack((weights, weight)), axis=0)
            consensus_run = np.nanprod(np.stack((results[run_idx].connectivity, weight)), axis=0)
            consensus = np.nansum(np.stack((consensus, consensus_run)), axis=0)

        eta_logger.info("#Creating consensus graphs [{} out of {}]".format(rep + 1, sumo_run.rep))
        assert not np.any(np.isnan(consensus))
        consensus = consensus / weights

        if sumo_run.log == "DEBUG":
            out_arrays.update({'pac_consensus_{}'.format(rep): consensus,
                               'runs_{}'.format(rep): np.array(run_indices)})

        org_con = consensus.copy()
        consensus[consensus < 0.5] = 0

        # calculate cophenetic correlation coefficient
        dist = pdist(org_con, metric="correlation")
        if np.any(np.isnan(dist)):
            ccc = np.nan
            sumo_run.logger.warning("Cannot calculate cophenetic correlation coefficient! Please inspect values in " +
                                    "your consensus matrix.")
        else:
            ccc = cophenet(linkage(dist, method="complete", metric="correlation"), dist)[0]

        # calculate proportion of ambiguous clustering
        den = (sumo_run.graph.nodes ** 2) - sumo_run.graph.nodes
        num = org_con[(org_con > 0.1) & (org_con < 0.9)].size
        pac = num * (1. / den)

        out_arrays.update({'pac': np.append(out_arrays['pac'], pac),
                           'cophenet': np.append(out_arrays['cophenet'], ccc)})

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
    conf_array = np.empty((9, 2), dtype=object)
    conf_array[:, 0] = ['method', 'n', 'max_iter', 'tol', 'subsample', 'calc_cost', 'h_init', 'seed', 'sparsity']
    conf_array[:, 1] = [sumo_run.method, sumo_run.n, sumo_run.max_iter, sumo_run.tol, sumo_run.subsample,
                        sumo_run.calc_cost, np.nan if sumo_run.h_init is None else sumo_run.h_init,
                        np.nan if sumo_run.seed is None else sumo_run.seed, sparsity]
    out_arrays.update({
        "clusters": cluster_array,
        "consensus": consensus,
        "unfiltered_consensus": org_con,
        "quality": np.array(quality),
        "samples": sumo_run.graph.sample_names,
        "config": conf_array
    })
    if sumo_run.priors is not None:
        out_arrays["priors"] = sumo_run.priors

    steps_reached = [results[i].steps for i in range(len(results))]
    maxiter_proc = round((sum([step == sumo_run.max_iter for step in steps_reached]) / len(steps_reached)) * 100, 3)
    eta_logger.info("#Reached maximum number of iterations in {}% of runs".format(maxiter_proc))
    if maxiter_proc >= 90:
        eta_logger.warning("Consider increasing -max_iter and deacreasing -tol to achieve better accuracy")
    out_arrays['steps'] = np.array([steps_reached])

    if sumo_run.log == "DEBUG":
        for i in range(len(results)):
            out_arrays["cost{}".format(i)] = results[i].delta_cost[-1, :]
            out_arrays["h{}".format(i)] = results[i].h
            out_arrays["samples{}".format(i)] = results[i].graph.sample_names[results[i].sample_ids]
            for si in range(len(results[i].s)):
                out_arrays["s{}{}".format(si, i)] = results[i].s[si]

        steps_plot_path = os.path.join(sumo_run.plot_dir, "steps_k{}.png".format(k))
        plot_metric(x=list(range(1, len(steps_reached) + 1)), y=[np.array(x) for x in steps_reached],
                    xlabel="factorization repetition", ylabel="number of iterations/steps",
                    title="Number of iterations reached by solver",
                    file_path=steps_plot_path, color="green", allow_omit_xticks=True)
        eta_logger.info("#Number of iterations reached for each factorization repetition has " +
                        "been saved to {}".format(steps_plot_path))

    save_arrays_to_npz(data=out_arrays, file_path=outfile)
    eta_logger.info("#Output file {} created".format(outfile))
    close_logger(eta_logger)

    return quality, outfile
