*************
Example usage
*************

In this example, we will use SUMO to stratify patients diagnosed with Acute Myeloid Leukemia (LAML) into sub-groups based on gene-expression, micro-RNA expression, and methylation datasets. We will describe steps that should be taken for pre-processing of these data, use SUMO to detect the various sub-groups of patients, and identify the features e.g., genes and methylation probes, that drive each of those subgroups.

We will use `gene expression <https://gdc.xenahubs.net/download/TCGA-LAML.htseq_fpkm.tsv.gz>`_, `methylation <https://gdc.xenahubs.net/download/TCGA-LAML.methylation27.tsv.gz>`_ and `miRNA expression <https://gdc.xenahubs.net/download/TCGA-LAML.mirna.tsv.gz>`_ data from UCSC XENA browser in this vignette.

==================
Data preprocessing
==================

While preprocessing data for SUMO analysis we strongly suggest the following steps:

 1. **Filtering your data** - SUMO handles missing values to some extent, however removing features and samples with a large fraction of missing values (>10%) has been shown to improve the classification. You can additionally choose to remove features that are likely to be noise e.g., low expressed genes in all samples.

 2. **Data normalization/transformation** - We advise the use of a log transform or a variant stabilizing transform when using count data as input. This step is omitted in the code below, as the downloaded data is already log normalized. When using methylation data, we prefer the use of M-values over beta values.  

 3. **Data standardization** - Lastly, each feature should be standardized before being input to SUMO.

Here is code in python that we can use to perform the preprocessing for the RNA-seq and the miRNA-sequence datasets

.. code-block:: python3

    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    
    def preprocess_logfpkm(inputfile, outputfile):
        """Filter the input log2(fpkm+1) values and write the output"""

        # read the log2(fpkm+1) values
        norm_fpkm = pd.read_csv(inputfile, sep='\t', header=0, index_col=0)
        print("Read %s" % inputfile)
    
        # remove genes where less then two samples have FPKM higher then zero
        norm_fpkm = norm_fpkm[np.sum(norm_fpkm > 1, axis=1) > 1]

        # standardize the norm_fpkm matrix
        scaler = preprocessing.StandardScaler()
        scaled_fpkm = scaler.fit_transform(norm_fpkm.T)
        scaled_fpkm = scaled_fpkm.T
        scaled_fpkm = pd.DataFrame(scaled_fpkm, index=list(norm_fpkm.index), columns=list(norm_fpkm.columns))
            # write the file
            scaled_fpkm.to_csv(outputfile, sep='\t', index=True, na_rep="NA")
            print("Wrote %s" % outputfile)
        
    preprocess_logfpkm("TCGA-LAML.htseq_fpkm.tsv.gz", "TCGA-LAML.htseq_fpkm.flt.tsv.gz")
    preprocess_logfpkm("TCGA-LAML.mirna.tsv.gz", "TCGA-LAML.mirna.flt.tsv.gz")


Here is the python code to perform preprocessing of the methylation dataset.

.. code-block:: python3

    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    
    # read in the beta values
    beta = pd.read_csv("TCGA-LAML.methylation27.tsv.gz", sep='\t', header=0, index_col=0)
    
    # remove rows where we do not have information from any sample
    beta = beta.dropna(axis=0, how='all')
    
    # convert each beta value to the corresponding M values
    def convert(B):
        eps = np.spacing(1)
        return(np.log2((B + eps)/(1. - B + eps)))
        
    M = beta.applymap(convert)
    print("Converted to M values")
    
    scaler = preprocessing.StandardScaler()
    scaled_M = scaler.fit_transform(M.T)
    scaled_M = scaled_M.T
    scaled_M = pd.DataFrame(scaled_M, index=list(M.index), columns=list(M.columns))
    print("Standardization complete")
    
    scaled_M.to_csv("TCGA-LAML.methylation27.flt.tsv.gz", sep='\t', index=True, na_rep="NA")


============
Running SUMO
============

.. |modes| raw:: html

    <img src="https://raw.githubusercontent.com/ratan-lab/sumo/development/doc/_images/modes.png" height="200px">

|modes|

SUMO provides four modes allowing for molecular subtyping of multi-omic data (*prepare* and *run*), as well as comprehensive analysis that includes identification of molecular features driving classification (*interpret*) and comparison with existing subtype classifications (*evaluate*).

------------
sumo prepare
------------

In this mode, SUMO calculates the pairwise similarity between the samples using each of the input omic datatypes (in this case gene expression, methylation and miRNA expression).

::

    sumo prepare -plot LAML.png TCGA-LAML.htseq_fpkm.flt.tsv.gz,TCGA-LAML.methylation27.flt.tsv.gz,TCGA-LAML.mirna.flt.tsv.gz prepared.LAML.npz

The above creates 'prepared.LAML.npz' file that contains the pairwise similarities organized as adjacency matrices, and three .png files with plots of the adjacency matrices for each omic datatype.

--------
sumo run
--------

In this mode, SUMO applies symmetric non-negative matrix tri-factorization on the similarity matrices to identify the clusters of samples. Estimating the best number of clusters remains a challenging problem, but we recommend that the user supply a range of values to use with SUMO. 

::

    sumo run prepared.LAML.npz 2,4 LAML

When the above command is run, SUMO creates an output directory named 'LAML'. In that directory, SUMO creates a sub-directory for each *k* (the number of clusters) that contains the factorization results in the form of .npz files, and a 'clusters.tsv' file with sample labels. A 'plots' sub-directory is also created, where we provide several plots that can assist in selection of the best number of subtypes in the dataset. A stable clustering result is characterized by a high value of cophenetic correlation coefficient (plotted in LAML/plots/cophenet.png) and low proportion of ambiguous clusterings (plotted in LAML/plots/pac.png).

The complete directory structure generated after running the above command is shown below.

::

    LAML
    ├── k2
    │   ├── clusters.tsv
    │   ├── eta_0.1.log
    │   ├── eta_0.1.npz
    │   └── sumo_results.npz -> eta_0.1.npz
    ├── k3
    │   ├── clusters.tsv
    │   ├── eta_0.1.log
    │   ├── eta_0.1.npz
    │   └── sumo_results.npz -> eta_0.1.npz
    ├── k4
    │   ├── clusters.tsv
    │   ├── eta_0.1.log
    │   ├── eta_0.1.npz
    │   └── sumo_results.npz -> eta_0.1.npz
    └── plots
        ├── consensus_k2.png
        ├── consensus_k3.png
        ├── consensus_k4.png
        ├── cophenet.png
        └── pac.png


--------------
sumo interpret
--------------

Use SUMO *interpret* mode to investigate which features drive obtained clustering results.

::

    sumo interpret LAML/k4/sumo_results.npz TCGA-LAML.htseq_fpkm.flt.tsv.gz,TCGA-LAML.methylation27.flt.tsv.gz,TCGA-LAML.mirna.flt.tsv.gz features.tsv


The above command generates a file "features.tsv" which reports the importance of each feature in driving each cluster. Briefly, we train a LightGBM model (https://github.com/microsoft/LightGBM) based on the clusters identified by SUMO, and the results from this mode are the SHAP (SHapley Additive exPlanations) feature importance deduced using that model.

For example, the results in features.tsv shows that the following top 10 features drive the various clusters:

+----------------------------+-----------------------------+
| Group 0                    | Group 1                     |
+====================+=======+====================+========+
| cg27497900         | 30.21 | cg21299958         | 36.01  |
+--------------------+-------+--------------------+--------+
| cg05934874         | 21.77 | cg16907075         | 13.5   |
+--------------------+-------+--------------------+--------+
| hsa-mir-574        | 17.65 | cg14142521         | 12.92  |
+--------------------+-------+--------------------+--------+
| hsa-mir-450a-1     | 12.71 | ENSG00000135404.10 | 11.87  |
+--------------------+-------+--------------------+--------+
| ENSG00000173599.12 | 7.795 | ENSG00000185875.11 | 9.78   |
+--------------------+-------+--------------------+--------+
| cg23705973         | 6.13  | cg24995240         | 4.21   |
+--------------------+-------+--------------------+--------+
| cg26450541         | 4.56  | ENSG00000114942.12 | 3.97   |
+--------------------+-------+--------------------+--------+
| ENSG00000173482.15 | 4.5   | ENSG00000271270.4  | 3.59   |
+--------------------+-------+--------------------+--------+
| hsa-mir-450b       | 2.76  | ENSG00000113272.12 | 3.51   |
+--------------------+-------+--------------------+--------+
| ENSG00000154122.11 | 1.96  | cg06540636         | 3.175  |
+--------------------+-------+--------------------+--------+


+----------------------------+-----------------------------+
| Group 2                    | Group 3                     |
+====================+=======+====================+========+
| hsa-mir-199a-2     | 24.34 | cg14178895         | 18.385 |
+--------------------+-------+--------------------+--------+
| ENSG00000153786.11 | 14.95 | cg00617305         | 12.89  |
+--------------------+-------+--------------------+--------+
| hsa-let-7e         | 10.81 | ENSG00000269845.1  | 11.81  |
+--------------------+-------+--------------------+--------+
| ENSG00000229816.1  | 9.43  | ENSG00000196705.7  | 11.61  |
+--------------------+-------+--------------------+--------+
| ENSG00000281016.1  | 8.96  | cg09891761         | 11.535 |
+--------------------+-------+--------------------+--------+
| ENSG00000177731.14 | 7.75  | ENSG00000160229.10 | 7.88   |
+--------------------+-------+--------------------+--------+
| cg18959422         | 7.57  | ENSG00000255730.3  | 4.84   |
+--------------------+-------+--------------------+--------+
| ENSG00000206841.1  | 4.71  | ENSG00000269399.2  | 2.88   |
+--------------------+-------+--------------------+--------+
| hsa-mir-128-2      | 4.63  | hsa-mir-4473       | 2.735  |
+--------------------+-------+--------------------+--------+
| hsa-mir-106a       | 3.645 | ENSG00000270876.1  | 2.68   |
+--------------------+-------+--------------------+--------+