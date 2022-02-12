.. :changelog:

History
=======
0.3.0 (2021-??-??)
------------------
* Updated class structure, to allow for addition of new solvers.
* Implemented a *supervised solver* for sumo, which allows to include "a priori" knowledge about labels of fraction of samples to improve the factorization results. This solver is automatically enabled when the '-labels' parameter is used.
* Fixed error that prevented using sumo *interpret* with newer hyperopt versions (>0.2.5)

0.2.7 (2021-07-02)
------------------
* Add random seed parameter for sumo *run*.
* Add more arrays in sumo_results.npz files:
    - 'steps' array, with number of iterations/steps reached in each repetition of the factorization;
    - 'config' array with simulation parameters (including sparsity).
* Add warning in eta*.log files if more then 90% of factorization repetitions finished in reaching set maximum number of iterations.
* Update plotting function and add 'steps' plot, produced when using the -DEBUG flag,
* Remove incorrect assertion about Euclidean Distance being bound to [0,1] range.
* Add entry point to run sumo directly from the repository (run.py).
* Updated the function checking is feature matrix is standardized in sumo *prepare*. Now reporting a range of feature means and standard deviations.

0.2.6 (2021-03-12)
------------------
* Updated REs scaling for consensus matrix creation.
* Add sample identifiers to sumo *run* result files.
* Updated documentation to include the detailed description of arrays in each .npz result file and an example of integration of somatic mutation data into SUMO workflow.
* Improved execution speed of sumo *prepare* by updating the filtering of loaded datasets and incorporating numba for euclidean distance calculation.
* Improved execution speed of sumo *run* by updating the resampling during the factorization.

0.2.5 (2020-06-11)
------------------
* Added Dockerfile.
* Improved clustering quality assessment by better utilization of consensus clustering in sumo *run*:
    - introduced clustering different random of subsets of samples in each run of factorization (fraction of samples removed in each run can be set with '-subsample' parameter);
    - increased default number of runs and introduced creation of multiple consensus matrices based on subsets of runs;
    - results .npz file now contains multiple PAC and CCC values (which are calculated for each consensus matrix);
    - updated plotting of PAC and CCC curves to show error bars.
* Updated scikit-learn version requirement.

0.2.4 (2020-03-06)
------------------
* Sumo *interpret* now creates two output files:
    - .tsv file containing matrix (features x clusters), where the value in each cell is the importance of the feature in that cluster;
    - .hits.tsv file containing features of most importance (number of top hits can be set with '-hits' parameter).
* Fixed training dataset in *interpret* to contain 80% of every unique class label.

0.2.3 (2020-02-25)
------------------
* Handle NaN values of cophenetic correlation coefficient.
* Update vignette.
* Fix issue resulting in not closing log files in *run*.
* If output directory of *run* already exists, remove it instead of overwriting.
* Change error information for data not meeting standardization thresholds in *prepare*.
* Add column-wise normalization of H matrix in *run* before cluster extraction using max_value method.

0.2.1 & 0.2.2 (2020-01-21)
--------------------------
* Adressed PyPI issues with long description.

0.2.0 (2020-01-21)
------------------
* Added sumo icon.
* Implemented new mode of sumo *interpret*, for finding features that drive clusters separation.
* Added example test case and data preprocessing suggestions in documentation.

[*prepare*]

* Improved plotting of heatmaps.
* Changed acceptable data types to binary [0, 1] data and continuous data which is standardized feature wise.
* Added cosine similarity measure, recommended for sparse data.
* Removed variable type argument, categorical and binary distance measure. Similarity measures now can be set for every layer separately with *-method* parameter. Available measures include pearson and spearman correlation, (new) cosine similarity and euclidean distance with RBF kernel (before used for continuous variables). The last measure remain the default.
* Updated support for .txt (space) and .tsv (tab delimited files).
* Added support for compressed files (.txt.gz, .txt.bz2, .tsv.gz, .tsv.bz2).
* Removed support for .npz files.

[*run*]

* Fixed typo resulting in adding two identical consensus matrices into .npz file.
* Changed default sparsity from a range of values to one (0.1), as *sumo* results are very stable towards changes in sparsity parameter (unless the number of clusters is raised to very high values).
* Improved plotting of consensus matrices.
* Added proportion of ambiguous clusterings (PAC) plot for improved number of clusters selection.
* Changed sumo_results.npz file into symbolic link to selected eta result file.
* Added text file with cluster labels in every k directory.

[*evaluate*]

* Change mode to support .tsv files as both input and labels file.

0.1.2 (2019-09-20)
------------------
* Fixed numerical issue in *feature_to_adjacency*.
* Added more tests.

0.1.1 (2019-09-16)
------------------
* Fixed minor issues with documentation and README.

0.1.0 (2019-09-16)
------------------
* First release.