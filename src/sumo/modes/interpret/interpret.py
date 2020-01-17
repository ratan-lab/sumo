from sumo.constants import INTERPRET_ARGS, SUPPORTED_EXT
from sumo.modes.mode import SumoMode
from sumo.utils import setup_logger, load_npz, load_data_text, docstring_formatter
import os
import numpy as np
import pandas as pd
import pathlib
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, cv, Dataset
from hyperopt import hp, tpe, fmin, STATUS_OK
import shap


@docstring_formatter(supported=SUPPORTED_EXT)
class SumoInterpret(SumoMode):
    """
    Sumo mode for interpreting clustering results. Constructor args are set in 'interpret' subparser.

    Args:
        | sumo_results (str): path to sumo_results.npz (created by running program with mode "run")
        | infiles (list): comma-delimited list of paths to input files, containing standardized feature matrices, \
            with samples in columns and features in rows(supported types of files: {supported})
        | outfile (str): output file from this analysis, containing matrix (features x clusters), \
            where the value in each cell is the importance of the feature in that cluster
        | do_cv (bool): use cross-validation to find the best model (default of False)
        | sn (int): index of row with sample names for .txt input files
        | fn (int): index of column with feature names for .txt input files
        | df (float): if percentage of missing values for feature exceeds this value, remove feature
        | ds (float): if percentage of missing values for sample (that remains after feature dropping) exceeds \
            this value, remove sample
        | logfile (str): path to save log file, if set to None stdout is used
    """

    def __init__(self, **kwargs):
        """ Creates instance od SumoInterpret class

        Args:
            **kwargs: keyword arguments, have to contain all of arguments detailed in class description otherwise
                AttributeError is raised
        """
        super().__init__(**kwargs)

        if not all([hasattr(self, arg) for arg in INTERPRET_ARGS]):
            # this should never happened due to object creation with parse_args in sumo/__init__.py
            raise AttributeError("Cannot create SumoInterpret object")
        assert hasattr(self, 'do_cv')

        self.logger = setup_logger("main", log_file=self.logfile)

        # check positional arguments
        if not os.path.exists(self.sumo_results):
            raise FileNotFoundError("Sumo results file not found")
        if not all([os.path.exists(fname) for fname in self.infiles]):
            raise FileNotFoundError("Input file not found")
        if os.path.exists(self.outfile):
            self.logger.warning("File '{}' already exist and will be overwritten.".format(self.outfile))

    def run(self):
        """ Find features that drive clusters separation """

        # read in the output of the classifier
        sumo = load_npz(self.sumo_results)
        if 'clusters' not in sumo.keys():
            raise ValueError('Clusters array not found in sumo_results file.')
        clusters = sumo['clusters']
        samples = list(clusters[:, 0])
        labels = np.array(list(clusters[:, 1]))

        # read all the features and arrange them in the same order as the output of
        # the classifier, order specified in 'samples'
        features = pd.DataFrame(columns=samples)
        for i in range(len(self.infiles)):
            fname = self.infiles[i]
            self.logger.info(
                "#Loading file: {} {}".format(fname, "(text file)" if pathlib.Path(fname).suffix == "" else ""))
            # load text file
            data = load_data_text(file_path=fname, sample_names=self.sn, feature_names=self.fn, drop_features=self.df,
                                  drop_samples=self.ds)
            data = data.reindex(columns=samples)
            features = pd.concat([features, data], sort=False)

        self.logger.info("#All found features: {}".format(features.shape[0]))
        features = features.T
        X = features.to_numpy(copy=True)

        # find the features that drive this classification
        model = self.create_classifier(x=X, y=labels)
        model.fit(X, labels)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X, y=labels)

        # create output file
        df = pd.DataFrame(0, index=features.columns.values, columns=range(len(shap_values)))
        for i in range(len(shap_values)):
            shap_cat_values = np.abs(shap_values[i])
            mean_shap = np.sum(shap_cat_values, axis=0)
            df.iloc[:, [i]] = mean_shap
        df.to_csv(self.outfile, sep='\t', index_label=False)

    def create_classifier(self, x: np.ndarray, y: np.ndarray):
        """ Create a gradient boosting method classifier

        Args:
            x (Numpy.ndarray): input feature matrix
            y (Numpy.ndarray): one dimensional array of labels in classification

        Returns:
            LGBM classifier
        """

        # divide this data into training and test data
        train_index = np.random.choice(range(0, y.shape[0]), int(0.8 * y.shape[0]), replace=False)
        test_index = list(set(range(0, y.shape[0])) - set(train_index))
        y_train = y[train_index]
        x_train = x[train_index]
        y_test = y[test_index]
        x_test = x[test_index]

        # lets use a LGBM classifier
        model = LGBMClassifier()
        model.fit(x_train, y_train)
        predictions = model.predict_proba(x_test)
        auc = roc_auc_score(y_test, predictions, multi_class='ovr')
        self.logger.info('The baseline score on the test set is {:.4f}.'.format(auc))

        if self.do_cv:
            train_set = Dataset(x_train, label=y_train)

            def objective(params, n_folds=10):
                params['num_leaves'] = int(params['num_leaves'])
                params['verbose'] = -1

                # Perform n_fold cross validation with hyperparameters
                # Use early stopping and evalute based on ROC AUC
                cv_results = cv(params, train_set, nfold=n_folds,
                                num_boost_round=10000,
                                early_stopping_rounds=100, metrics='auc',
                                seed=50)

                # Extract the best score
                best_score = max(cv_results['auc-mean'])

                # minimize loss
                loss = 1 - best_score

                return {'loss': loss, 'params': params, 'status': STATUS_OK}

            space = {'num_leaves': hp.quniform('num_leaves', 30, 150, 1)}
            best_params = fmin(fn=objective, space=space, algo=tpe.suggest,
                               max_evals=50, rstate=np.random.RandomState(50))
            self.logger.info(best_params)
            return LGBMClassifier(**best_params)

        return model
