from collections import Counter
from sumo.constants import PREPARE_ARGS, SUPPORTED_EXT, SIMILARITY_METHODS, LOG_LEVELS
from sumo.modes.mode import SumoMode
from sumo.modes.prepare.similarity import feature_to_adjacency
from sumo.utils import plot_heatmap_seaborn, save_arrays_to_npz, setup_logger, docstring_formatter, check_categories, \
    is_standardized, load_data_text
import numpy as np
import os
import pathlib


@docstring_formatter(sim_methods=SIMILARITY_METHODS, log_levels=LOG_LEVELS, supported=SUPPORTED_EXT)
class SumoPrepare(SumoMode):
    """ Sumo mode for data pre-processing and creation of multiplex network files. Constructor args are set \
    in 'prepare' subparser.

    Args:
        | infiles (list): comma-delimited list of paths to input files, containing standardized feature matrices, \
            with samples in columns and features in rows (supported types of files: {supported})
        | outfile (str): path to output .npz file
        | method (list): comma-separated list of methods for every layer (available methods: {sim_methods})
        | k (float): fraction of nearest neighbours to use for sample similarity calculation using Euclidean distance \
            similarity
        | alpha (float): hypherparameter of RBF similarity kernel, for Euclidean distance similarity
        | missing (list): acceptable fraction of available (not missing) values for assessment of distance/similarity \
            between pairs of samples, either one value or different values for every layer
        | atol (float): if input files have continuous values, sumo checks if data is standardized feature-wise, \
            meaning all features should have mean close to zero, with standard deviation around one; use this \
            parameter to set tolerance of standardization checks
        | sn (int): index of row with sample names for .txt input files
        | fn (int): index of column with feature names for .txt input files
        | df (float): if percentage of missing values for feature exceeds this value, remove feature
        | ds (float): if percentage of missing values for sample (that remains after feature dropping) exceeds \
            this value, remove sample
        | logfile (str): path to save log file, if set to None stdout is used
        | log (str): sets the logging level from {log_levels}
        | plot (str): path to save adjacency matrix heatmap(s), if set None plots are displayed on screen

    """

    def __init__(self, **kwargs):
        """ Creates instance od SumoPrepare class

        Args:
            **kwargs: keyword arguments, have to contain all of arguments detailed in class description otherwise
                AttributeError is raised
        """
        super().__init__(**kwargs)

        if not all([hasattr(self, arg) for arg in PREPARE_ARGS]):
            # this should never happened due to object creation with parse_args in sumo/__init__.py
            raise AttributeError("Cannot create SumoPrepare object, missing constructor arguments")

        # check positional arguments
        if not all([os.path.exists(fname) for fname in self.infiles]):
            raise FileNotFoundError("Input file not found")

        self.ftypes = []
        for fname in self.infiles:
            suff = pathlib.Path(fname).suffix if pathlib.Path(fname).suffix not in ['.gz', '.bz2'] else ''.join(
                pathlib.Path(fname).suffixes[-2:])
            if suff not in SUPPORTED_EXT:
                raise ValueError("Unrecognized input file type '{}'".format(suff))
            self.ftypes.append(suff if suff != "" else ".txt")

        if not all([method in SIMILARITY_METHODS for method in self.method]):
            raise ValueError("Unrecognized similarity method")

        self.plot_base = None
        if self.plot:
            basename = os.path.basename(self.plot)
            dirname = os.path.dirname(self.plot)
            self.plot_base = os.path.join(dirname, basename.split('.png')[0])

        self.logger = setup_logger("main", self.log, self.logfile)

    def load_all_data(self):
        """ Load all of input files

        Returns:
            list of tuples, every containing file name (str) and filtered feature matrix (pandas.DataFrame))

        """
        layers, layers_fnames = [], []
        # TODO: add multiprocessing
        for i in range(len(self.infiles)):
            fname = self.infiles[i]
            self.logger.info(
                "#Loading file: {} {}".format(fname, "(text file)" if pathlib.Path(fname).suffix == "" else ""))

            # load text file
            data = load_data_text(file_path=fname, sample_names=self.sn, feature_names=self.fn, drop_features=self.df,
                                  drop_samples=self.ds)
            layers.append(data)
            layers_fnames.append(fname)

        return list(zip(layers_fnames, layers))

    def run(self):
        """ Generate similarity matrices for samples based on biological data """
        # load_data
        layers = self.load_all_data()  # list of tuples (file_name, feature_matrix)

        # check variable types
        if len(self.method) == 1:
            self.method = [self.method[0]] * len(layers)
        elif len(layers) != len(self.method):
            raise ValueError("Number of matrices extracted from input files and number of similarity methods " +
                             "does not correspond")

        # check missing value parameter
        if len(self.missing) == 1:
            self.logger.info("#Setting all 'missing' parameters to {}".format(self.missing[0]))
            self.missing = [self.missing[0]] * len(layers)
        elif len(layers) != len(self.missing):
            raise ValueError("Number of matrices extracted from input files and number of given missing parameters " +
                             "does not correspond")

        # extract sample names
        all_samples = set()
        for layer_data in layers:
            all_samples = all_samples.union({name for name in layer_data[1].columns})
        self.logger.info("#Total number of unique samples: {}".format(len(all_samples)))

        out_arrays = {}
        adj_matrices = []

        # create adjacency matrices
        for i in range(len(layers)):
            self.logger.info("#Layer: {}".format(i))
            layer_data = layers[i][1]

            # add missing samples layer
            samples = {name for name in layer_data.columns}
            for name in all_samples - samples:
                layer_data[name] = np.nan

            # sort data frame by sample names
            layer_data.sort_index(axis=1, inplace=True)

            # extract feature matrices
            f = layer_data.values.T
            self.logger.info("Feature matrix: ({} samples x {} features)".format(f.shape[0], f.shape[1]))

            # check if feature matrix values are correct
            ncat = check_categories(f)
            if ncat != [0, 1]:
                standardized = is_standardized(f, axis=0, atol=self.atol)
                if not standardized[0]:
                    raise ValueError("Incorrect values in feature matrix. Mean of features in " +
                                     "({},{}) ".format(round(standardized[1][0], 3), round(standardized[1][1], 3)) +
                                     "range. Standard deviation of features in " +
                                     "({}, {}) ".format(round(standardized[2][0], 3), round(standardized[2][1], 3)) +
                                     "range. Please, supply either binary dataset " +
                                     "(0 or 1 feature values) or continuous values standardized feature-wise. " +
                                     "Alternatively for almost standardized continuous data, " +
                                     "increase '-atol' parameter value (currently {}).".format(self.atol))
                else:
                    self.logger.debug("Data is correctly standardized")
            else:
                self.logger.debug("Found two unique categories in data: [0, 1]")
                if self.method[i] != 'cosine':
                    self.logger.info("Using '{}' similarity for [0, 1] data. ".format(self.method[i]) +
                                     "Suggested better measure: cosine similarity.")

            # create adjacency matrix
            a = feature_to_adjacency(f, missing=self.missing[i], method=self.method[i], n=self.k, alpha=self.alpha)
            self.logger.info('Adjacency matrix {} created [similarity method: {}]'.format(a.shape, self.method[i]))

            # plot adjacency matrix
            plot_path = self.plot_base + "_" + str(i) + ".png" if self.plot else self.plot_base
            plot_heatmap_seaborn(a, title="Layer {} (source:{})".format(i, layers[i][0]), file_path=plot_path)
            if self.plot:
                self.logger.info("Adjacency matrix plot saved to {}".format(plot_path))

            # add matrices to output arrays
            out_arrays[str(i)] = a
            adj_matrices.append(a)
            out_arrays["f" + str(i)] = f

        # check if there are samples not accessible in any layer
        missing_samples = []
        for a in adj_matrices:
            missing_samples += [i for i in range(a.shape[1]) if np.all(np.isnan(a[:, i]))]

        samples_to_drop = [sample for sample, val in Counter(missing_samples).items() if val == len(adj_matrices)]
        if samples_to_drop:
            # drop inaccessible samples
            self.logger.info("Found samples inaccessible in every layer of graph. " +
                             "Try changing '-missing' parameter or inspect your data ")
            sample_names = np.array(sorted(list(all_samples)))[np.array(samples_to_drop)]
            self.logger.info("Dropped samples: {}".format(list(sample_names)))
            updated_out_arrays = {}
            selector = np.array([x for x in range(len(all_samples)) if x not in samples_to_drop])
            for i in range(len(out_arrays.keys())):
                if str(i) not in out_arrays.keys():
                    break
                updated_out_arrays[str(i)] = out_arrays[str(i)][selector[:, None], selector]
                updated_out_arrays["f" + str(i)] = out_arrays["f" + str(i)][selector, :]

            # create output file
            updated_out_arrays["samples"] = np.array(sorted(list(all_samples)))[selector]
            save_arrays_to_npz(data=updated_out_arrays, file_path=self.outfile)

        else:
            # create output file
            out_arrays["samples"] = np.array(sorted(list(all_samples)))
            save_arrays_to_npz(data=out_arrays, file_path=self.outfile)

        self.logger.info("#Output file {} created".format(self.outfile))
