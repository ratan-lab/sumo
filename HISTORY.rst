.. :changelog:

History
=======
0.1.3 (????-??-??)
------------------
* Fixed typo resulting in adding two identical consensus matrices into .npz file in *run* mode.
* Added sumo icon.
* Updated calculation of similarity between two samples for binary datasets. Now uninformative positions are ignored.
* Fixed error preventing loading of pickled object arrays stored in label files in *evaluate* mode, while using *-npz* flag.
* Changed default sparsity in *run* mode from a range of values to one (0.1), as *sumo* results are very stable towards changes in sparsity parameter (unless the number of clusters is raised to very high values).
* Changed color palette for the *run* mode heatmaps.
* Update chi-squared and agreement distance implementation and '-missing' parameter in *prepare* mode.
* Unify variable names to categorical, continuous and binary.

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