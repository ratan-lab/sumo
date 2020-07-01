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

    sumo interpret LAML/k4/sumo_results.npz TCGA-LAML.htseq_fpkm.flt.tsv.gz,TCGA-LAML.methylation27.flt.tsv.gz,TCGA-LAML.mirna.flt.tsv.gz LAML_features


The above command generates a file two files "LAML_features.tsv" and "LAML_features.hits.tsv" which report the importance of each feature in supporting cluster separation. Briefly, we train a LightGBM model (https://github.com/microsoft/LightGBM) based on the clusters identified by SUMO, and the results from this mode are the SHAP (SHapley Additive exPlanations) feature importance deduced using that model.

For example, here the results shows that the following top 10 features support various clusters:



+----------------------------------+----------------------------------+
| Group 1                          | Group 2                          |
+====================+=============+====================+=============+
| hsa-mir-22         | 75.3233574  | cg18979819         | 52.90086054 |
+--------------------+-------------+--------------------+-------------+
| cg25076881         | 54.68371529 | cg14325649         | 45.66930331 |
+--------------------+-------------+--------------------+-------------+
| ENSG00000170584.9  | 29.78108163 | hsa-mir-199a-2     | 40.69787094 |
+--------------------+-------------+--------------------+-------------+
| cg14027234         | 22.89081537 | cg23705973         | 40.62359106 |
+--------------------+-------------+--------------------+-------------+
| cg04109382         | 20.48871007 | cg19346899         | 34.96037943 |
+--------------------+-------------+--------------------+-------------+
| cg01110312         | 17.64408184 | ENSG00000148672.8  | 30.14950394 |
+--------------------+-------------+--------------------+-------------+
| cg25645748         | 16.85100694 | cg20340596         | 25.47572752 |
+--------------------+-------------+--------------------+-------------+
| ENSG00000213468.3  | 16.76245821 | cg14576628         | 22.37044065 |
+--------------------+-------------+--------------------+-------------+
| ENSG00000131778.16 | 15.58228512 | hsa-mir-574        | 20.81683191 |
+--------------------+-------------+--------------------+-------------+
| ENSG00000005238.18 | 12.45771289 | hsa-mir-193a       | 19.25008652 |
+--------------------+-------------+--------------------+-------------+

+----------------------------------+----------------------------------+
| Group 1                          | Group 2                          |
+====================+=============+====================+=============+
| hsa-mir-26a-1      | 62.81218414 | ENSG00000281162.1  | 89.15823387 |
+--------------------+-------------+--------------------+-------------+
| cg10957584         | 40.31222349 | ENSG00000139318.7  | 51.8247784  |
+--------------------+-------------+--------------------+-------------+
| hsa-mir-146a       | 39.65185698 | ENSG00000198585.10 | 41.81821387 |
+--------------------+-------------+--------------------+-------------+
| hsa-mir-9-1        | 31.29804593 | hsa-mir-181c       | 27.85737717 |
+--------------------+-------------+--------------------+-------------+
| hsa-mir-26a-2      | 25.79902879 | hsa-mir-335        | 26.22691472 |
+--------------------+-------------+--------------------+-------------+
| cg07656391         | 23.58458233 | hsa-mir-6503       | 23.9870722  |
+--------------------+-------------+--------------------+-------------+
| hsa-mir-139        | 23.49122814 | cg03387497         | 21.45016846 |
+--------------------+-------------+--------------------+-------------+
| cg00795268         | 23.33766879 | hsa-mir-3913-1     | 18.55877291 |
+--------------------+-------------+--------------------+-------------+
| cg18403361         | 23.31470381 | ENSG00000101160.12 | 17.26709816 |
+--------------------+-------------+--------------------+-------------+
| cg16108132         | 21.16833145 | cg21402071         | 17.20317828 |
+--------------------+-------------+--------------------+-------------+