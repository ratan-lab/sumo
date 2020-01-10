.. :changelog:

History
=======
0.2.0 (????-??-??)
------------------
* Added sumo icon.

[*prepare*]

* Improved plotting of heatmaps.
* Changed acceptable data types to binary [0, 1] data and continuous data which is standardized feature wise.
* Added cosine similarity measure, recommended for sparse data.
* Removed variable type argument, categorical and binary distance measure. Similarity measures now can be set for every layer separately with *-method* parameter. Available measures include pearson and spearman correlation, (new) cosine similarity and euclidean distance with RBF kernel (before used for continuous variables). The last measure remain the default.
* Updated tests accordingly.

[*run*]

* Fixed typo resulting in adding two identical consensus matrices into .npz file.
* Changed default sparsity from a range of values to one (0.1), as *sumo* results are very stable towards changes in sparsity parameter (unless the number of clusters is raised to very high values).
* Improved plotting of consensus matrices
* Added proportion of ambiguous clusterings (PAC) plot for improved number of clusters selection.

[*evaluate*]

* Fixed error preventing loading of pickled object arrays stored in label files, while using *-npz* flag.

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