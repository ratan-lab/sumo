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
        fpkm = pd.read_csv(inputfile, sep='\t', header=0, index_col=0)
        print("Read %s" % inputfile)
    
        # convert to fpkm
        fpkm = np.power(2, fpkm) - 1
    
        # remove genes where the FPKM rowsum is less than or equal to 1
        fpkm = fpkm[fpkm.sum(axis=1, skipna=True) > 1]
        fpkm = fpkm + 1
    
        # convert back to log2 space 
        fpkm = fpkm.apply(np.log2)
        
        # standardize the fpkm matrix
        scaler = preprocessing.StandardScaler()
        scaled_fpkm = scaler.fit_transform(fpkm.T)
        scaled_fpkm = scaled_fpkm.T
        scaled_fpkm = pd.DataFrame(scaled_fpkm, index=list(fpkm.index), columns=list(fpkm.columns))
    
        # write the file
        scaled_fpkm.to_csv(outputfile, sep='\t', index=True)
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
        return(np.log2(B/(1. - B)))
        
    M = beta.applymap(convert)
    print("Converted to M values")
    
    scaler = preprocessing.StandardScaler()
    scaled_M = scaler.fit_transform(M.T)
    scaled_M = scaled_M.T
    scaled_M = pd.DataFrame(scaled_M, index=list(M.index), columns=list(M.columns))
    print("Standardization complete")
    
    scaled_M.to_csv("TCGA-LAML.methylation27.flt.tsv.gz", sep='\t', index=True)


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

+--------------------+--------------------+--------------------+--------------------+
|       Group 0      |       Group 1      |       Group 2      |       Group 3      |
+====================+====================+====================+====================+
|     cg27497900     |     cg21299958     |   hsa-mir-199a-2   |     cg14178895     |
+--------------------+--------------------+--------------------+--------------------+
|     cg05934874     |     cg16907075     | ENSG00000153786.11 |     cg00617305     |
+--------------------+--------------------+--------------------+--------------------+
|     hsa-mir-574    |     cg14142521     |     hsa-let-7e     |  ENSG00000269845.1 |
+--------------------+--------------------+--------------------+--------------------+
|   hsa-mir-450a-1   | ENSG00000135404.10 |  ENSG00000229816.1 |  ENSG00000196705.7 |
+--------------------+--------------------+--------------------+--------------------+
| ENSG00000173599.12 | ENSG00000185875.11 |  ENSG00000281016.1 |     cg09891761     |
+--------------------+--------------------+--------------------+--------------------+
|     cg23705973     |     cg24995240     | ENSG00000177731.14 | ENSG00000160229.10 |
+--------------------+--------------------+--------------------+--------------------+
|     cg26450541     | ENSG00000114942.12 |     cg18959422     |  ENSG00000255730.3 |
+--------------------+--------------------+--------------------+--------------------+
| ENSG00000173482.15 |  ENSG00000271270.4 |  ENSG00000206841.1 |  ENSG00000269399.2 |
+--------------------+--------------------+--------------------+--------------------+
|    hsa-mir-450b    | ENSG00000113272.12 |    hsa-mir-128-2   |    hsa-mir-4473    |
+--------------------+--------------------+--------------------+--------------------+
| ENSG00000154122.11 |     cg06540636     |    hsa-mir-106a    |  ENSG00000270876.1 |
+--------------------+--------------------+--------------------+--------------------+