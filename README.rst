========================================
sumo: subtyping tool for multi-omic data
========================================

|badge1| |badge2| |badge3| |badge4|

.. |badge1| image:: https://travis-ci.org/ratan-lab/sumo.svg?branch=master
    :target: https://travis-ci.org/ratan-lab/sumo
.. |badge2| image:: https://img.shields.io/github/license/ratan-lab/sumo
    :alt: GitHub
.. |badge3| image:: https://readthedocs.org/projects/python-sumo/badge/?version=latest
    :target: https://python-sumo.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |badge4| image:: https://img.shields.io/pypi/v/python-sumo
    :alt: PyPI

.. |sumo| raw:: html

    <img src="https://raw.githubusercontent.com/ratan-lab/sumo/development/doc/_images/sumo.png" height="200px">

|sumo|

.. inclusion-start-marker-do-not-remove

.. long-description-start-marker-do-not-remove

.. short-description-start-marker-do-not-remove

**sumo** is a command-line tool to identify molecular subtypes in multi-omics datasets. It implements a novel nonnegative matrix factorization (NMF) algorithm to identify groups of samples that share molecular signatures, and provides additional modules to evaluate such assignments and identify features that drive the classification.

.. short-description-end-marker-do-not-remove

To see how **sumo** performs the joint factorization of patient-similarity networks and through integration of multi-omic data
identifies significantly different molecular subtypes of LGG
read our `publication in CellReportsMethods <https://www.sciencedirect.com/science/article/pii/S2667237521002290>`_:

    Sienkiewicz, K., Chen, J., Chatrath, A., Lawson, J. T., Sheffield, N. C., Zhang, L., & Ratan, A. (2022). Detecting molecular subtypes from multi-omics datasets using SUMO. In Cell Reports Methods (Vol. 2, Issue 1, p. 100152). Elsevier BV. https://doi.org/10.1016/j.crmeth.2021.100152

For practical details about **sumo** analysis pipeline, examples of downstream analysis and troubleshooting
please refer to our published `STAR Protocol <https://www.sciencedirect.com/science/article/pii/S2666166721008169>`_ and/or package documentation available at https://python-sumo.readthedocs.io.


Installation
------------
You can install **sumo** from PyPI, by executing command below. Please note that we require **python 3.6+**.

.. code:: sh

    pip install python-sumo

(March 2021): We have noted an installation issue with the *llvmlite* package (required for one of **sumo** dependencies). To avoid errors with installation, **upgrade pip to a >19.0 version**.

License
-------

`MIT <LICENSE>`__

Usage
-----

**sumo** consists of four subroutines. A typical workflow includes running *prepare* mode for preparation of similarity matrices from feature matrices, followed by factorization of produced multiplex network (mode *run*). Third mode *evaluate* can be used for comparison of created cluster labels against biologically significant labels. A fourth mode *interpret* can be used to detect the importance of each feature in driving the classification.

(February 2022) As of SUMO v0.3, a semi-supervised classification of samples is now supported. This allows the inclusion of "a priori" knowledge about labels of fraction of samples to improve the factorization results. The supervised version of solver is automatically enabled in *sumo run*, when the '-labels' parameter is used.

prepare
^^^^^^^
Generates similarity matrices for samples based on biological data and saves them into multiplex network files.

::

    usage: sumo prepare [-h] [-method METHOD] [-k K] [-alpha ALPHA] [-missing MISSING]
                        [-atol ATOL] [-sn SN] [-fn FN] [-df DF] [-ds DS] [-logfile LOGFILE]
                        [-log {DEBUG,INFO,WARNING}] [-plot PLOT]
                        infile1,infile2,... outfile.npz

    positional arguments:
      infile1,infile2,...   comma-delimited list of paths to input files, containing
                            standardized feature matrices, with samples in columns and
                            features in rows (supported types of files: ['.txt', '.txt.gz',
                            '.txt.bz2', '.tsv', '.tsv.gz', '.tsv.bz2'])
      outfile.npz           path to output .npz file

    optional arguments:
      -h, --help            show this help message and exit
      -method METHOD        either one method of sample-sample similarity calculation, or
                            comma-separated list of methods for every layer (available
                            methods: ['euclidean', 'cosine', 'pearson', 'spearman'], default
                            of euclidean)
      -k K                  fraction of nearest neighbours to use for sample similarity
                            calculation using Euclidean distance similarity (default of 0.1)
      -alpha ALPHA          hypherparameter of RBF similarity kernel, for Euclidean distance
                            similarity (default of 0.5)
      -missing MISSING      acceptable fraction of available values for assessment of
                            distance/similarity between pairs of samples - either one value
                            or comma-delimited list for every layer (default of [0.1])
      -atol ATOL            if input files have continuous values, sumo checks if data is
                            standardized feature-wise, meaning all features should have mean
                            close to zero, with standard deviation around one; use this
                            parameter to set tolerance of standardization checks (default of
                            0.01)
      -sn SN                index of row with sample names for input files (default of 0)
      -fn FN                index of column with feature names for input files (default of 0)
      -df DF                if percentage of missing values for feature exceeds this value,
                            remove feature (default of 0.1)
      -ds DS                if percentage of missing values for sample (that remains after
                            feature dropping) exceeds this value, remove sample (default of
                            0.1)
      -logfile LOGFILE      path to save log file, by default stdout is used
      -log {DEBUG,INFO,WARNING}
                            sets the logging level (default of INFO)
      -plot PLOT            path to save adjacency matrix heatmap(s), by default plots are
                            displayed on screen

**Example**

.. code:: sh

    sumo prepare -plot plot.png methylation.txt,expression.txt prepared.data.npz

run
^^^
Cluster multiplex network using non-negative matrix tri-factorization to identify molecular subtypes.

::

    usage: sumo run [-h] [-sparsity SPARSITY] [-labels labels.tsv] [-n N]
                    [-method {max_value,spectral}] [-max_iter MAX_ITER] [-tol TOL]
                    [-subsample SUBSAMPLE] [-calc_cost CALC_COST] [-logfile LOGFILE]
                    [-log {DEBUG,INFO,WARNING}] [-h_init H_INIT] [-t T] [-rep REP]
                    [-seed SEED]
                    infile.npz k outdir

    positional arguments:
      infile.npz            input .npz file containing adjacency matrices for every network
                            layer and sample names (file created by running program with mode
                            "run") - consecutive adjacency arrays in file are indexed in
                            following way: "0", "1" ... and index of sample name vector is
                            "samples"
      k                     either one value describing number of clusters or coma-delimited
                            range of values to check (sumo will suggest cluster structure
                            based on cophenetic correlation coefficient)
      outdir                path to save output files

    optional arguments:
      -h, --help            show this help message and exit
      -sparsity SPARSITY    either one value or coma-delimited list of sparsity penalty
                            values for H matrix (sumo will try different values and select
                            the best results; default of [0.1])
      -labels labels.tsv    optional path to .tsv file containing some of known sample labels
                            to be included as prior knowledge during the factorization
                            (inclusion of this parameter enables the 'supervised' mode of
                            sumo), the file should contain sample names in 'sample' and labels
                            in 'label' column
      -n N                  number of repetitions (default of 60)
      -method {max_value,spectral}
                            method of cluster extraction (default of "max_value")
      -max_iter MAX_ITER    maximum number of iterations for factorization (default of 500)
      -tol TOL              if objective cost function value fluctuation (|Δℒ|) is smaller
                            than this value, stop iterations before reaching max_iter
                            (default of 1e-05)
      -subsample SUBSAMPLE  fraction of samples randomly removed from each run, cannot be
                            greater then 0.5 (default of 0.05)
      -calc_cost CALC_COST  number of steps between every calculation of objective cost
                            function (default of 20)
      -logfile LOGFILE      path to save log file (by default printed to stdout)
      -log {DEBUG,INFO,WARNING}
                            set the logging level (default of INFO)
      -h_init H_INIT        index of adjacency matrix to use for H matrix initialization (by
                            default using average adjacency), only for unsupervised
                            classification (when no "-labels" are set)
      -t T                  number of threads (default of 1)
      -rep REP              number of times consensus matrix is created for the purpose of
                            assessing clustering quality (default of 5)
      -seed SEED            random state (none by default)

**Example**

.. code:: sh

    sumo run -t 8 prepared.data.npz 2,5 results_dir

evaluate
^^^^^^^^
Evaluate clustering results, given set of labels.

::

    usage: sumo evaluate [-h] [-metric {NMI,purity,ARI}] [-logfile LOGFILE]
                         [-log {DEBUG,INFO,WARNING}]
                         infile.tsv labels


    positional arguments:
      infile.tsv            input .tsv file containing sample names in 'sample' and
                            clustering labels in 'label' column (clusters.tsv file created by
                            running sumo with mode 'run')
      labels                .tsv of the same structure as input file

    optional arguments:
      -h, --help            show this help message and exit
      -metric {NMI,purity,ARI}
                            metric for accuracy evaluation (by default all metrics are
                            calculated)
      -logfile LOGFILE      path to save log file (by default printed to stdout)
      -log {DEBUG,INFO,WARNING}
                            sets the logging level (default of INFO)


**Example**

.. code:: sh

    sumo evaluate results_dir/k3/clusters.tsv labels.tsv

interpret
^^^^^^^^^
Find features that support clusters separation.

::

    usage: sumo interpret [-h] [-logfile LOGFILE] [-log {DEBUG,INFO,WARNING}] [-hits HITS]
                          [-max_iter MAX_ITER] [-n_folds N_FOLDS] [-t T] [-seed SEED]
                          [-sn SN] [-fn FN] [-df DF] [-ds DS]
                          sumo_results.npz infile1,infile2,... output_prefix

    positional arguments:
      sumo_results.npz      path to sumo_results.npz (created by running program with mode
                            "run")
      infile1,infile2,...   comma-delimited list of paths to input files, containing
                            standardized feature matrices, with samples in columns and
                            features in rows(supported types of files: ['.txt', '.txt.gz',
                            '.txt.bz2', '.tsv', '.tsv.gz', '.tsv.bz2'])
      output_prefix         prefix of output files - sumo will create two output files (1)
                            .tsv file containing matrix (features x clusters), where the
                            value in each cell is the importance of the feature in that
                            cluster; (2) .hits.tsv file containing features of most
                            importance

    optional arguments:
      -h, --help            show this help message and exit
      -logfile LOGFILE      path to save log file (by default printed to stdout)
      -log {DEBUG,INFO,WARNING}
                            sets the logging level (default of INFO)
      -hits HITS            sets number of most important features for every cluster, that
                            are logged in .hits.tsv file
      -max_iter MAX_ITER    maximum number of iterations, while searching through
                            hyperparameter space
      -n_folds N_FOLDS      number of folds for model cross validation (default of 5)
      -t T                  number of threads (default of 1)
      -seed SEED            random state (default of 1)
      -sn SN                index of row with sample names for input files (default of 0)
      -fn FN                index of column with feature names for input files (default of 0)
      -df DF                if percentage of missing values for feature exceeds this value,
                            remove feature (default of 0.1)
      -ds DS                if percentage of missing values for sample (that remains after
                            feature dropping) exceeds this value, remove sample (default of
                            0.1)

**Example**

.. code:: sh

    sumo interpret -t 8 results_dir/k3/sumo_results.npz methylation.txt,expression.txt interpret_results

.. inclusion-end-marker-do-not-remove

Please refer to documentation for `example usage cases and suggestions for data preprocessing <https://python-sumo.readthedocs.io/en/latest/example.html>`_.

.. long-description-end-marker-do-not-remove
