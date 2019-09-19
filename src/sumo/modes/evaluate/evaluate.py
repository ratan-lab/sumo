from sumo.constants import EVALUATE_ARGS, CLUSTER_METRICS
from sumo.modes.mode import SumoMode
from sumo.utils import setup_logger, load_npz, check_accuracy, docstring_formatter
import numpy as np
import os


@docstring_formatter(metrics=CLUSTER_METRICS)
class SumoEvaluate(SumoMode):
    """
    Sumo mode for evaluating accuracy of clustering. Constructor args are set in 'evaluate' subparser.

    Args:
        | infile (str): input .npz file containing array indexed as 'clusters', with sample names in first column and \
            clustering labels in second column (file created by running sumo with mode 'run')
        | labels (str): .npy file containing array with sample names in first column and true labels in second column \
        | metric (str): one of metrics ({metrics}) for accuracy evaluation, if set to None all metrics are calculated \
        | logfile (str): path to save log file, if set to None stdout is used

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
            raise AttributeError("Cannot create SumoRun object")

        if not os.path.exists(self.infile):
            raise FileNotFoundError("Input file not found")
        if not os.path.exists(self.labels):
            raise FileNotFoundError("Labels file not found")

        self.logger = setup_logger("main", log_file=self.logfile)

        self.true_labels = None
        self.cluster_labels = None
        self.common_samples = None

    def run(self):
        """ Evaluate clustering results, given set of labels """

        self.logger.info("#Loading file: {}".format(self.infile))
        data = load_npz(self.infile)
        if 'clusters' not in data.keys():
            raise ValueError("Incorrect structure of input file")
        clusters_array = data['clusters']

        if not self.npz:
            labels_array = np.load(self.labels)
            if isinstance(labels_array, np.lib.npyio.NpzFile):
                raise ValueError("Attempting to use .npz label file without '-npz' option supplied")
        else:
            label_data = load_npz(self.labels)
            if self.npz not in label_data.keys():
                raise ValueError("Incorrect structure of label file")
            labels_array = label_data[self.npz]

        if len(labels_array.shape) == 1 or labels_array.shape[1] < 2:
            raise ValueError("Incorrect structure of label file")

        self.common_samples = list(set(clusters_array[:, 0]) & set(labels_array[:, 0]))
        if len(self.common_samples) == 0:
            raise ValueError(
                "Labels from {} file does not correspond to cluster labels in {} file.".format(self.labels,
                                                                                               self.infile))

        if len(self.common_samples) != clusters_array.shape[0]:
            all_samples = list(set(clusters_array[:, 0]) | set(labels_array[:, 0]))
            self.logger.warning(
                "Found {} common labels [{} unique labels supplied in both files]".format(len(self.common_samples),
                                                                                          len(all_samples)))

        true_labels = []
        cluster_labels = []
        for label in self.common_samples:
            cluster_labels.append(clusters_array[np.where(clusters_array[:, 0] == label), 1][0][0])
            true_labels.append(labels_array[np.where(labels_array[:, 0] == label), 1][0][0])

        self.true_labels = np.array(true_labels)
        self.cluster_labels = np.array(cluster_labels)

        self._evaluate(metric=self.metric)

    def _evaluate(self, metric: str = None):
        """Evaluate clustering accuracy"""
        assert self.cluster_labels is not None and self.true_labels is not None
        methods = [metric] if metric else CLUSTER_METRICS
        for method in methods:
            self.logger.info("{}:\t{}".format(method, round(check_accuracy(self.cluster_labels, self.true_labels,
                                                                           method=method), 5)))
