from sumo.constants import EVALUATE_ARGS, CLUSTER_METRICS, LOG_LEVELS
from sumo.modes.mode import SumoMode
from sumo.utils import setup_logger, check_accuracy, docstring_formatter
import pandas as pd
import os


@docstring_formatter(metrics=CLUSTER_METRICS, log_levels=LOG_LEVELS)
class SumoEvaluate(SumoMode):
    """
    Sumo mode for evaluating accuracy of clustering. Constructor args are set in 'evaluate' subparser.

    Args:
        | infile (str): input .tsv file containing sample names in 'sample' and clustering labels in 'label' column \
            (clusters.tsv file created by running sumo with mode 'run')
        | labels (str): .tsv of the same structure as input file
        | metric (str): one of metrics ({metrics}) for accuracy evaluation, if set to None all metrics are calculated
        | logfile (str): path to save log file, if set to None stdout is used
        | log (str): sets the logging level from {log_levels}

    """

    def __init__(self, **kwargs):
        """ Creates instance od SumoEvaluate class

        Args:
            **kwargs: keyword arguments, have to contain all of arguments detailed in class description otherwise
                AttributeError is raised
        """
        super().__init__(**kwargs)

        if not all([hasattr(self, arg) for arg in EVALUATE_ARGS]):
            # this should never happened due to object creation with parse_args in sumo/__init__.py
            raise AttributeError("Cannot create SumoEvaluate object")

        if not os.path.exists(self.infile):
            raise FileNotFoundError("Input file not found")
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError("Labels file not found")

        self.logger = setup_logger("main", level=self.log, log_file=self.logfile)
        self.common_samples = None
        self.data = None
        self.labels = None

    def load_tsv(self, fname: str):
        """ Load .tsv file"""
        data = pd.read_csv(fname, delimiter=r'\s+')
        if data.empty or data.values.shape == (1, 1) or data.values.shape[1] <= 1:
            raise ValueError('File {} is not tab delimited or have incorrect structure'.format(fname))
        if not all([label in data.columns for label in ['sample', 'label']]):
            raise ValueError('Incorrect file header ({})'.format(fname))
        return data

    def run(self):
        """ Evaluate clustering results, given set of labels """

        data = self.load_tsv(self.infile)
        self.logger.info("#Loading input file: {} [{} x {}]".format(self.infile, data.shape[0], data.shape[1]))

        labels = self.load_tsv(self.labels_file)
        self.logger.info(
            "#Loading labels file: {} [{} x {}]".format(self.labels_file, labels.shape[0], labels.shape[1]))

        self.common_samples = list(set(data['sample']) & set(labels['sample']))
        if len(self.common_samples) == 0:
            raise ValueError("Sample labels in both files does not correspond.")
        elif len(self.common_samples) != data.shape[0]:
            all_samples = list(set(data['sample']) | set(labels['sample']))
            self.logger.warning(
                "Found {} common labels [{} unique labels supplied in both files]".format(len(self.common_samples),
                                                                                          len(all_samples)))

        # TODO: assert sample and label in colnames
        data_common = data.loc[data['sample'].isin(self.common_samples)]
        self.data = data_common.sort_values(by=['sample'])
        labels_common = labels.loc[labels['sample'].isin(self.common_samples)].sort_values(by=['sample'])
        self.labels = labels_common.sort_values(by=['sample'])

        assert list(self.data['sample']) == list(self.labels['sample'])
        self._evaluate(metric=self.metric)

    def _evaluate(self, metric: str = None):
        """Evaluate clustering accuracy"""
        assert self.data is not None and self.labels is not None
        methods = [metric] if metric else CLUSTER_METRICS
        for method in methods:
            self.logger.info(
                "{}:\t{}".format(method, round(check_accuracy(self.data['label'].values, self.labels['label'].values,
                                                              method=method), 5)))
