.. :changelog:

History
=======
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