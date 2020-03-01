from lightgbm import LGBMClassifier
from pathlib import Path
from sumo.constants import PREDICT_ARGS, SUPPORTED_EXT, LOG_LEVELS
from sumo.modes.mode import SumoMode
from sumo.utils import setup_logger, load_npz, load_data_text, docstring_formatter
import os
import numpy as np
import pandas as pd
import pickle


@docstring_formatter(supported=SUPPORTED_EXT, log_levels=LOG_LEVELS)
class SumoPredict(SumoMode):
    """
    Sumo mode for classifying new samples into clusters found by sumo. Constructor args are set in 'predict' subparser.

    Args:
        | infiles (list): comma-delimited list of paths to input files, containing standardized feature matrices, \
            with columns corresponding to new samples we want to classify into clusters found by sumo (supported \
            types of files: {supported})
        | classifier (str): classifier file created by sumo 'interpret
        | output_prefix (str): prefix of output files - sumo will create two output files (1) .tsv file, containing \
            table with predicted probability for each class for each sample and (2) .labels.tsv file with table \
            containing predicted cluster labels for every sample
        | logfile (str): path to save log file, if set to None stdout is used
        | log (str): sets the logging level from {log_levels}
        | sn (int): index of row with sample names for .txt input files
        | fn (int): index of column with feature names for .txt input files
        | df (float): if percentage of missing values for feature exceeds this value, remove feature
        | ds (float): if percentage of missing values for sample (that remains after feature dropping) exceeds \
            this value, remove sample

    """

    def __init__(self, **kwargs):
        """ Creates instance od SumoPredict class

        Args:
            **kwargs: keyword arguments, have to contain all of arguments detailed in class description otherwise
                AttributeError is raised
        """
        super().__init__(**kwargs)

        if not all([hasattr(self, arg) for arg in PREDICT_ARGS]):
            # this should never happened due to object creation with parse_args in sumo/__init__.py
            raise AttributeError("Cannot create SumoPredict object")

        self.logger = setup_logger("main", level=self.log, log_file=self.logfile)

        # check positional arguments
        if not all([os.path.exists(fname) for fname in self.infiles]):
            raise FileNotFoundError("Input file not found")
        if not os.path.exists(self.classifier):
            raise FileNotFoundError("Classifier file {} not found".format(self.classifier))
        if Path(self.classifier).suffix != ".pickle":
            raise ValueError("Incorrect classifier, expected .pickle file")
        if os.path.exists(self.output_prefix):
            self.logger.warning("File '{}' already exist and will be overwritten.".format(self.output_prefix))

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

    def run(self):
        """ Classify new samples into existing clusters """

        # read classifier
        with open(self.classifier, "rb") as pickle_handle:
            classifier = pickle.load(pickle_handle)
            model = classifier['model']
            feature_names = classifier['features']
            if not isinstance(model, LGBMClassifier) and feature_names.size == model.n_features_:
                # this should not happen if .pickle file is created by sumo interpret
                raise ValueError("Incorrect classifier file")

        # read all the features matrices
        data = pd.DataFrame()
        for i in range(len(self.infiles)):
            fname = self.infiles[i]
            self.logger.info(
                "# Loading file: {} {}".format(fname, "(text file)" if Path(fname).suffix == "" else ""))
            # load text file
            newdata = load_data_text(file_path=fname, sample_names=self.sn, feature_names=self.fn,
                                     drop_features=self.df, drop_samples=self.ds)
            data = pd.concat([data, newdata], sort=False)

        self.logger.info("# Found {} unique samples and {} features".format(data.shape[1], data.shape[0]))
        feature_indicator = np.array([name in feature_names for name in data.index])  # TODO: can we speed it up?

        self.logger.info("# {}% of features from input files will be used for sample classification".format(
            (np.sum(feature_indicator) / data.shape[0]) * 100))

        # drop features that are not included in classifier file (were not used to build a classifier)
        data = data[feature_indicator]

        # fill not found features with NA's
        empty_features = pd.DataFrame(columns=data.columns, index=[x for x in feature_names if x not in data.index])
        self.logger.info(
            "# {} out of {} classifier features were not found in input files".format(empty_features.shape[0],
                                                                                      feature_names.size))
        if not empty_features.empty:
            data = pd.concat([data, empty_features])

        # arrange features in order dictated by classifier
        data = data.loc[list(feature_names), :]
        assert np.all(data.index == feature_names)

        # predict sample labels
        self.logger.info("# Number of classes: {}".format(model.n_classes_))
        res = model.predict_proba(data.T)
        predicted_classes = pd.DataFrame(res, columns=model.classes_, index=data.columns)
        predicted_classes.index.name = "sample"
        assert np.allclose(predicted_classes.sum(axis=1).values, 1)
        labels = predicted_classes.idxmax(axis=1).to_frame(name="label")

        # create output files
        predicted_classes.columns = ["GROUP_{}".format(x) for x in predicted_classes.columns]
        predicted_classes.to_csv("{}.tsv".format(self.output_prefix), sep="\t")
        self.logger.debug("Predicted probability for each class:\n{}".format(predicted_classes))

        labels.to_csv("{}.labels.tsv".format(self.output_prefix),
                      sep="\t")  # TODO now in case of equality, returns first occurence
        self.logger.debug("Sample labels:\n{}".format(labels))

        self.logger.info(
            "# Output files {}.tsv and {}.labels.tsv saved.".format(self.output_prefix, self.output_prefix))
