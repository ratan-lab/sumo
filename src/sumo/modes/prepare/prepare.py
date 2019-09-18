from collections import Counter
from pandas import DataFrame, isna, read_csv
from sumo.constants import PREPARE_ARGS, SUPPORTED_EXT, VAR_TYPES, SIMILARITY_METHODS, LOG_LEVELS
from sumo.modes.mode import SumoMode
from sumo.modes.prepare.similarity import feature_corr_similarity, feature_to_adjacency
from sumo.utils import load_npz, plot_heatmap, save_arrays_to_npz, setup_logger, get_logger, docstring_formatter
import numpy as np
import os
import pathlib


def filter_features_and_samples(data: DataFrame, drop_features: float = 0.1, drop_samples: float = 0.1):
    """ Filter data frame features and samples

    Args:
        data (pandas.DataFrame): data frame (with samples in columns and features in rows)
        drop_features (float): if percentage of missing values for feature exceeds this value, remove this feature
        drop_samples (float): if percentage of missing values for sample (that remains after feature dropping) \
            exceeds this value, remove this sample

    Returns:
        filtered data frame
    """
    logger = get_logger()
    # check arguments
    if drop_features < 0 or drop_features >= 1:
        raise ValueError("Incorrect value od 'drop_feature', expected value in range [0,1)")
    if drop_samples < 0 or drop_samples >= 1:
        raise ValueError("Incorrect value od 'drop_samples', expected value in range [0,1)")

    # drop features if needed
    nans = isna(data).values
    remove_feature = []
    for i in range(nans.shape[0]):
        if list(nans[i, :]).count(True) / nans.shape[1] > drop_features:
            remove_feature.append(i)
    data.drop(data.index[remove_feature], axis=0, inplace=True)

    # drop samples if needed
    nans = isna(data).values
    remove_sample = []
    for i in range(nans.shape[1]):
        if list(nans[:, i]).count(True) / nans.shape[0] > drop_samples:
            remove_sample.append(i)
    data.drop(data.columns[remove_sample], axis=1, inplace=True)

    logger.info("Number of dropped rows/features: {}".format(len(remove_feature)))
    logger.info("Number of dropped columns/samples: {}".format(len(remove_sample)))
    logger.info("Data shape: {}".format(data.values.shape))

    return data


def load_data_txt(file_path: str, sample_names: int = None, feature_names: int = None, drop_features: float = 0.1,
                  drop_samples: float = 0.1):
    """ Loads data from tab delimited .txt file (with samples in columns and features in rows) into pandas.DataFrame

    Args:
        file_path (str): path to the tab delimited .txt file
        sample_names (int): index of row with sample names
        feature_names (int): index of column with feature names
        drop_features (float): if percentage of missing values for feature exceeds this value, remove this feature
        drop_samples (float): if percentage of missing values for sample (that remains after feature dropping) \
            exceeds this value, remove this sample

    Returns:
        data (pandas.DataFrame): data frame loaded from file, with missing values removed

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Data file not found")

    data = read_csv(file_path, sep="\t", header=sample_names, index_col=feature_names)
    if data.empty or data.values.shape == (1, 1):
        raise ValueError('File cannot be read correctly, file is not tab delimited or is corrupted')
    elif data.values.dtype == np.object:
        raise ValueError("File contains some non-numerical values other than 'NA'")

    return filter_features_and_samples(data, drop_features=drop_features, drop_samples=drop_samples)


def load_data_npz(file_path: str, sample_idx: str = None, drop_features: float = 0.1, drop_samples: float = 0.1):
    """ Loads data from .npz file into pandas.DataFrame

    Args:
        file_path (str): path to the .npz file
        sample_idx (str): key of array containing custom sample names in every .npz file \
            (if not supplied use column indices)
        drop_features (float): if percentage of missing values for feature exceeds this value, remove this feature
        drop_samples (float): if percentage of missing values for sample (that remains after feature dropping) \
            exceeds this value, remove this sample

    Returns:
        data_frames (list): data frames loaded from file, with missing values removed

    """

    data = load_npz(file_path)

    if len(data.keys()) == 0:
        raise ValueError("File {} is empty".format(file_path))

    logger = get_logger()
    # extract sample names
    if sample_idx:
        # sample names supplied
        if sample_idx not in data.keys():
            raise ValueError("Sample names vector '{}' not found in {} file".format(sample_idx, file_path))

        sample_names = data[sample_idx]
        array_idx = [k for k in data.keys() if k != sample_idx]

    else:
        # sample names not supplied
        array_idx = list(data.keys())
        sample_names = np.arange(data[array_idx[0]].shape[1])
        logger.warning("Sample names not supplied with '-names' parameter. Using default sample names, can " +
                       "cause mistakes when there are missing samples!")

    try:
        if not all([data[idx].shape[1] == sample_names.shape[0] for idx in array_idx]):
            raise ValueError("Length of vector with sample names does not corresponds to " +
                             "shapes of other arrays in {} file.".format(file_path))
    except IndexError:
        raise AttributeError(
            "One dimensional array found in input file, use '-s' option to supply sample names or remove array")

    data_frames = []
    for idx in array_idx:
        df = filter_features_and_samples(DataFrame(data[idx], columns=list(sample_names)), drop_features=drop_features,
                                         drop_samples=drop_samples)
        data_frames.append(df)

    return data_frames


@docstring_formatter(var_types=VAR_TYPES, sim_methods=SIMILARITY_METHODS, log_levels=LOG_LEVELS)
class SumoPrepare(SumoMode):
    """ Sumo mode for data pre-processing and creation of multiplex network files. Constructor args are set \
    in 'prepare' subparser.

    Args:
        | infiles (list): list of paths to input .npz or .txt files (all input files should be structured in following \
            way: consecutive samples in columns, consecutive features in rows)
        | vars (list): either one variable type from {var_types} for every data matrix or list of variable types for \
            each of them
        | outfile (str): path to output .npz file
        | method (str): method of sample-sample similarity calculation selected from {sim_methods}
        | k (float): fraction of nearest neighbours to use for sample similarity calculation using with RBF method
        | alpha (float): hypherparameter of RBF similarity kernel
        | missing (float): acceptable fraction of available (not missing) values for assessment of distance/similarity \
            between pairs of samples
        | names (str): optional key of array containing custom sample names in every .npz file (if not set ids of \
            samples are used, which can cause problems when layers have missing samples)
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

        self.ftypes = [pathlib.Path(fname).suffix if pathlib.Path(fname).suffix != "" else ".txt" for fname in
                       self.infiles]
        if not all([ftype in SUPPORTED_EXT for ftype in self.ftypes]):
            raise ValueError("Unrecognized input file type")

        if not all([var in VAR_TYPES for var in self.vars]):
            raise ValueError("Unrecognized variable type")

        if len(self.vars) > 1 and len(self.infiles) != len(self.vars):
            raise ValueError("Number of input files and variable types does not correspond")

        self.plot_base = None
        if self.plot:
            basename = os.path.basename(self.plot)
            dirname = os.path.dirname(self.plot)
            self.plot_base = os.path.join(dirname, basename.split('.png')[0])

        self.logger = setup_logger("main", self.log, self.logfile)

    def load_all_data(self):
        """ Load all of .txt and .npz input files

        Returns:
            list of tuples, every containing file name (str) and filtered feature matrix (pandas.DataFrame))

        """
        layers, layers_fnames = [], []
        # TODO: add multiprocessing
        for i in range(len(self.infiles)):
            fname = self.infiles[i]
            self.logger.info(
                "#Loading file: {} {}".format(fname, "(text file)" if pathlib.Path(fname).suffix == "" else ""))

            if self.ftypes[i] == ".txt":
                # load .txt file
                data = load_data_txt(file_path=fname, sample_names=self.sn, feature_names=self.fn,
                                     drop_features=self.df, drop_samples=self.ds)
                layers.append(data)
                layers_fnames.append(fname)

            else:
                # load .npz file (can contain multiple matrices)
                data = load_data_npz(file_path=fname, sample_idx=self.names, drop_features=self.df,
                                     drop_samples=self.ds)
                self.logger.info("#Found {} matrices in {} file".format(len(data), fname))
                layers += data
                layers_fnames += [fname] * len(data)

        return list(zip(layers_fnames, layers))

    def run(self):
        """ Generate similarity matrices for samples based on biological data """
        # load_data
        layers = self.load_all_data()  # list of tuples (file_name, feature_matrix)

        # check variable types
        if len(self.vars) == 1:
            self.logger.info("#Setting all variable types to {}".format(self.vars[0]))
            self.vars = [self.vars[0]] * len(layers)
        elif len(layers) != len(self.vars):
            raise ValueError("Number of matrices extracted from input files and number of variable types " +
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
            self.logger.info("Feature matrix shape: {}".format(f.shape))

            # create adjacency matrix
            if self.method == "rbf":
                a = feature_to_adjacency(f, variable_type=self.vars[i], n=self.k, missing=self.missing,
                                         alpha=self.alpha)
            else:
                a = feature_corr_similarity(f, missing=self.missing, method=self.method)
            self.logger.info('Adjacency matrix {} created [variable_type={}]'.format(a.shape, self.vars[i]))

            # plot adjacency matrix
            plot_path = self.plot_base + "_" + str(i) + ".png" if self.plot else self.plot_base
            plot_heatmap(a, log_scale=True, color_bar=True, title="Layer {} (source:{})".format(i, layers[i][0]),
                         file_path=plot_path)
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
