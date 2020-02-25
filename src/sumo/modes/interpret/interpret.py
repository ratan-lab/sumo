from sumo.constants import INTERPRET_ARGS, SUPPORTED_EXT, LOG_LEVELS
from sumo.modes.mode import SumoMode
from sumo.utils import setup_logger, load_npz, load_data_text, docstring_formatter
import os
import numpy as np
import pandas as pd
import pathlib
from timeit import default_timer as timer
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, cv, Dataset
from hyperopt import STATUS_OK, hp, fmin, tpe, Trials
import shap


@docstring_formatter(supported=SUPPORTED_EXT, log_levels=LOG_LEVELS)
class SumoInterpret(SumoMode):
    """
    Sumo mode for interpreting clustering results. Constructor args are set in 'interpret' subparser.

    Args:
        | sumo_results (str): path to sumo_results.npz (created by running program with mode "run")
        | infiles (list): comma-delimited list of paths to input files, containing standardized feature matrices, \
            with samples in columns and features in rows(supported types of files: {supported})
        | outfile (str): output file from this analysis, containing matrix (features x clusters), \
            where the value in each cell is the importance of the feature in that cluster
        | max_iter (int): maximum number of iterations, while searching through hyperparameter space
        | n_folds (int): number of folds for model cross validation
        | t (int): number of threads
        | seed (int): random state
        | sn (int): index of row with sample names for .txt input files
        | fn (int): index of column with feature names for .txt input files
        | df (float): if percentage of missing values for feature exceeds this value, remove feature
        | ds (float): if percentage of missing values for sample (that remains after feature dropping) exceeds \
            this value, remove sample
        | logfile (str): path to save log file, if set to None stdout is used
        | log (str): sets the logging level from {log_levels}

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

        self.logger = setup_logger("main", level=self.log, log_file=self.logfile)
        self.iteration = 0

        # check positional arguments
        if not os.path.exists(self.sumo_results):
            raise FileNotFoundError("Sumo results file not found")
        if not all([os.path.exists(fname) for fname in self.infiles]):
            raise FileNotFoundError("Input file not found")
        if os.path.exists(self.outfile):
            self.logger.warning("File '{}' already exist and will be overwritten.".format(self.outfile))

        if self.t <= 0:
            raise ValueError("Incorrect number of threads set with parameter '-t'")

    def run(self):
        """ Find features that support clusters separation """

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
        df = pd.DataFrame(0, index=features.columns.values,
                          columns=["group_{}".format(x) for x in range(len(shap_values))])
        for i in range(len(shap_values)):
            shap_cat_values = np.abs(shap_values[i])
            mean_shap = np.sum(shap_cat_values, axis=0)
            df.iloc[:, [i]] = mean_shap
        df.to_csv(self.outfile, sep='\t', index_label="feature")

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
        model = LGBMClassifier(n_jobs=self.t)
        model.fit(x_train, y_train)
        predictions = model.predict_proba(x_test)
        is_binary = True if np.unique(y_test).shape[0] == 2 else False  # special 'binary' case

        space = {
            'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt',
                                                          'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                                         {'boosting_type': 'goss', 'subsample': 1.0}]),
            'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
            'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
        }

        if is_binary:
            predictions = np.argmax(predictions, axis=1)
        else:
            space['class_weight'] = hp.choice('class_weight', [None, 'balanced'])
        auc = roc_auc_score(y_test, predictions, multi_class='ovr')
        self.logger.debug('The baseline score on the test set is {:.4f}.'.format(auc))

        def objective(params, n_folds=self.n_folds):
            self.iteration += 1

            subsample = params['boosting_type'].get('subsample', 1.0)
            params['boosting_type'] = params['boosting_type']['boosting_type']
            params['subsample'] = subsample
            params['verbose'] = -1
            for p in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
                params[p] = int(params[p])

            params['histogram_pool_size'] = 1024
            # NOTE: Above parameter is introduced to reduce memory consumption
            self.logger.debug("Parameters: {}".format(params))

            start = timer()
            train_set = Dataset(x_train, label=y_train)

            # Perform n_folds cross validation
            cv_results = cv(params, train_set, num_boost_round=10000,
                            nfold=n_folds, early_stopping_rounds=100,
                            metrics='auc', seed=self.seed)
            run_time = timer() - start

            # Loss must be minimized
            best_score = np.max(cv_results['auc-mean'])
            loss = 1 - best_score

            # Boosting rounds that returned the highest cv score
            n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

            return {'loss': loss, 'params': params, 'iteration': self.iteration,
                    'estimators': n_estimators,
                    'train_time': run_time, 'status': STATUS_OK}

        bayes_trials = Trials()

        # find best parameters
        _ = fmin(fn=objective, space=space, algo=tpe.suggest,
                 max_evals=self.max_iter, trials=bayes_trials,
                 rstate=np.random.RandomState(self.seed))
        assert len(list(bayes_trials)) == self.max_iter

        bayes_trials_results = sorted(bayes_trials.results, key=lambda z: z['loss'])
        best_bayes_params = bayes_trials_results[0]['params']
        best_bayes_estimators = bayes_trials_results[0]['estimators']
        best = LGBMClassifier(n_estimators=best_bayes_estimators,
                              random_state=self.seed, **best_bayes_params,
                              n_jobs=self.t)

        best.fit(x_train, y_train)
        predictions = best.predict_proba(x_test)
        if is_binary:
            predictions = np.argmax(predictions, axis=1)
        auc = roc_auc_score(y_test, predictions, multi_class='ovr')
        self.logger.info('The score on the test set after CV is {:.4f}.'.format(auc))

        return best
