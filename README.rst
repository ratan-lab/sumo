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

.. inclusion-start-marker-do-not-remove

.. short-description-start-marker-do-not-remove

**sumo** is a command-line tool to identify molecular subtypes in multi-omics datasets. It implements a novel nonnegative matrix factorization (NMF) algorithm to identify groups of samples that share molecular signatures, and provides tools to evaluate such assignments.

.. short-description-end-marker-do-not-remove

Installation
------------
You can install **sumo** from PyPI, by executing command below. Please note that we require python 3.6+.

.. code:: sh

    pip install python-sumo

Dependencies
------------

-  python 3.6+
-  python libraries:

   -  `NumPy <https://www.numpy.org>`__
   -  `pandas <https://pandas.pydata.org>`__
   -  `SciPy <https://www.scipy.org>`__
   -  `scikit-learn <https://scikit-learn.org>`__
   -  `Matplotlib <https://matplotlib.org>`__
   -  `Seaborn <https://seaborn.pydata.org>`__

Optional requirements
^^^^^^^^^^^^^^^^^^^^^

-  `pytest <http://pytest.org>`__ (for running the test suite)
-  `Sphinx <http://sphinx-doc.org>`__ (for generating documentation)

Documentation
-------------
The official documentation is available at https://python-sumo.readthedocs.io

License
-------

`MIT <LICENSE>`__


Usage
-----

Typical workflow includes running *prepare* mode for preparation of similarity
matrices from feature matrices, followed by factorization of produced multiplex network (mode *run*).
Third mode *evaluate* can be used for comparison of created cluster labels against biologically significant labels.

prepare
^^^^^^^
Generates similarity matrices for samples based on biological data and saves them into multiplex network files.

::

    Usage:
        sumo prepare [-h] [-method {rbf,pearson,spearman}] [-k K]
                     [-alpha ALPHA] [-missing MISSING] [-names NAMES] [-sn SN]
                     [-fn FN] [-df DF] [-ds DS] [-logfile LOGFILE]
                     [-log {DEBUG,INFO,WARNING}] [-plot PLOT]
                     infile1,infile2,... var1,var2,... outfile.npz

    Positional arguments:
        infile1,infile2,...   comma-delimited list of paths to input .npz or .txt
                              files (all input files should be structured in
                              following way: consecutive samples in columns,
                              consecutive features in rows")
        var1(,var2,...)       either one variable type for every data matrix in
                              input file(s) or comma-delimited list of variable
                              types ['continuous', 'binary', 'categorical']
        outfile.npz           path to output .npz file

    Optional arguments:
        -h, --help            show this help message and exit
        -method {rbf,pearson,spearman}
                              method of sample-sample similarity calculation
                              (default of "rbf")
        -k K                  fraction of nearest neighbours to use for sample
                              similarity calculation using RBF method (default of 0.1)
        -alpha ALPHA          hypherparameter of RBF similarity kernel (default of 0.5)
        -missing MISSING      acceptable fraction of available values for assessment
                              of distance/similarity between pairs of samples (default of 0.1)
        -names NAMES          optional key of array containing custom sample names
                              in every .npz file (if not set ids of samples are used,
                              which can cause problems when layers have missing samples)
        -sn SN                index of row with sample names for .txt input files
                              (default of 0)
        -fn FN                index of column with feature names for .txt input files
                              (default of 0)
        -df DF                if percentage of missing values for feature exceeds
                              this value, remove feature (default of 0.1)
        -ds DS                if percentage of missing values for sample (that
                              remains after feature dropping) exceeds this value,
                              remove sample (default of 0.1)
        -logfile LOGFILE      path to save log file, by default stdout is used
        -log {DEBUG,INFO,WARNING}
                              Sets the logging level (default of INFO)
        -plot PLOT            path to save adjacency matrix heatmap(s),
                              by default plots are displayed on screen

**Example**

.. code:: sh

    sumo prepare -plot plot.png methylation.txt,expression.txt continuous prepared.data.npz

run
^^^
Cluster multiplex network using non-negative matrix tri-factorization to identify molecular subtypes.

::

    Usage:
        sumo run [-h] [-sparsity SPARSITY] [-n N]
                 [-method {max_value,spectral}] [-max_iter MAX_ITER] [-tol TOL]
                 [-calc_cost CALC_COST] [-logfile LOGFILE]
                 [-log {DEBUG,INFO,WARNING}] [-h_init H_INIT] [-t T]
                 infile.npz k outdir

    Positional arguments:
        infile.npz            input .npz file containing adjacency matrices for
                              every network layer and sample names (file created by
                              running program with mode "run") - consecutive
                              adjacency arrays in file are indexed in following way:
                              "0", "1" ... and index of sample name vector is "samples"
        k                     either one value describing number of clusters or
                              coma-delimited range of values to check (sumo will
                              suggest cluster structure based on cophenetic
                              correlation coefficient)
        outdir                path to save output files

    Optional arguments:
        -h, --help            show this help message and exit
        -sparsity SPARSITY    either one value or coma-delimited list of sparsity
                              penalty values for H matrix (sumo will try different
                              values and select the best results; default of
                              [0.0001, 0.001, 0.01, 0.1, 1, 10.0, 100.0])
        -n N                  number of repetitions (default of 50)
        -method {max_value,spectral}
                              method of cluster extraction (default of "max_value")
        -max_iter MAX_ITER    maximum number of iterations for factorization
                              (default of 500)
        -tol TOL              if objective cost function value fluctuation (|Δℒ|) is
                              smaller than this value, stop iterations before
                              reaching max_iter (default of 1e-05)
        -calc_cost CALC_COST  number of steps between every calculation of objective
                              cost function (default of 20)
        -logfile LOGFILE      path to save log file (by default printed to stdout)
        -log {DEBUG,INFO,WARNING}
                              Set the logging level (default of INFO)
        -h_init H_INIT        index of adjacency matrix to use for H matrix
                              initialization (by default using average adjacency)
        -t T                  number of threads (default of 1)

**Example**

.. code:: sh

    sumo run -t 10 prepared.data.npz 2,5 results_dir

evaluate
^^^^^^^^
Evaluate clustering results, given set of labels.

::

    Usage:
        sumo evaluate [-h] [-npz NPZ] [-metric {NMI,purity,ARI}]
                      [-logfile LOGFILE]
                      infile.npz labels


    Positional arguments:
        infile.npz            input .npz file containing array indexed as
                              'clusters', with sample names in first column and
                              clustering labels in second column (file created by
                              running sumo with mode 'run')
        labels                either .npy file containing array with sample names in
                              first column and labels in second column or .npz
                              file (requires using '-npz' option)

    Optional arguments:
        -h, --help            show this help message and exit
        -npz NPZ              key of array containing labels in .npz file
        -metric {NMI,purity,ARI}
                              metric for accuracy evaluation (by default all metrics
                              are calculated)
        -logfile LOGFILE      path to save log file (by default printed to stdout)

**Example**

.. code:: sh

    sumo evaluate -npz subtypes results_dir/k3/sumo_results.npz labels.npz

.. inclusion-end-marker-do-not-remove

.. Please refer to documentation for more detailed description of a method,
.. example usage cases and suggestions for data pre-preparation.
