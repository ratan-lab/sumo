******************
Example usage case
******************

In this example case we want to identify groups of samples and molecular signatures that they are sharing among
LGG patients, based on multi-omic data from GDC TCGA.
You can download `gene expression <https://gdc.xenahubs.net/download/TCGA-LGG.htseq_fpkm.tsv.gz>`_,
`methylation <https://gdc.xenahubs.net/download/TCGA-LGG.methylation450.tsv.gz>`_
and `miRNA <https://gdc.xenahubs.net/download/TCGA-LGG.mirna.tsv.gz>`_ expression data from UCSC XENA browser.

==================
data preprocessing
==================

While preprocessing data for *sumo* analysis we strongly suggest following steps below in order to receive the most
meaningful results:

 1. **Filtering your data** - *sumo* handles missing values in feature matrices up to some extent, however removing
features and samples with high amount of missing values improves quality of classification. Depending on used data,
removing outliers may also be advisable.

 2. **Data normalization / transformation** - in case of using counts data, log normalization should be performed
(this step is omitted in code below, as data is already normalized). When using methylation data in form of beta values,
we strongly recommend transforming values into M-values.

 3. **Data standardization** - sumo required standardization of each feature matrices to provide classification that
isn't biased towards any of the data types over another


.. code-block:: R

    dir()
    #[1] "TCGA-LGG.htseq_fpkm.tsv.gz"     "TCGA-LGG.methylation450.tsv.gz" "TCGA-LGG.mirna.tsv.gz"

    #####################
    ## GENE EXPRESSION ##
    #####################

    ## load data
    exp <- read.table("TCGA-LGG.htseq_fpkm.tsv.gz", header=T, check.names = F, row.names = 1)

    exp[1:3,1:3]
    #                       TCGA-S9-A6WQ-01A TCGA-S9-A7IS-01A TCGA-DU-8162-01A
    # ENSG00000242268.2        0.17573358        0.7922957        0.1981496
    # ENSG00000270112.3        0.04593638        0.4337997        0.2958724
    # ENSG00000167578.15       2.83294358        2.3981568        2.5938472

    dim(exp)
    #[1] 60483   529

    ## standardization
    exp <- t(scale(t(as.matrix(exp))))

    # save preprocessed file
    gz <- gzfile("TCGA-LGG.htseq_fpkm.flt.tsv.gz", "w")
    write.table(exp, file=gz)
    close(gz)

    #################
    ## METHYLATION ##
    #################

    ## load data
    met <- read.table("TCGA-LGG.methylation450.tsv", header=T, check.names = F, row.names = 1)

    dim(met)

    ## transform beta-values in M-values
    beta <- as.matrix(met)
    Mvals <- log2(beta/(1-beta))
    met <- Mvals

    ## standardization
    met <- t(scale(t(as.matrix(met))))

    # save preprocessed file
    gz <- gzfile("TCGA-LGG.methylation450.flt.tsv.gz", "w")
    write.table(met, file=gz)
    close(gz)

    ######################
    ## miRNA EXPRESSION ##
    ######################

    ## load data
    mirna <- read.table("TCGA-LGG.mirna.tsv.gz", header=T, check.names = F, row.names = 1)

    ## standardization
    mirna <- t(scale(t(as.matrix(mirna))))

    # save preprocessed file
    gz <- gzfile("TCGA-LGG.mirna.flt.tsv.gz", "w")
    write.table(mirna, file=gz)
    close(gz)`


============
running sumo
============

.. |modes| raw:: html

    <img src="https://raw.githubusercontent.com/ratan-lab/sumo/development/doc/_images/modes.png" height="200px">

|modes|

*sumo* tool provides four modes allowing for molecular subtyping of multi-omic data (*prepare* and *run*),
as well as comprehensive analysis including identification of molecular signatures driving classification (*interpret*)
and comparison with existing subtype data (*evaluate*).

------------
sumo prepare
------------

First step in sumo analysis is a creation of multiplex network. In this case our network will have three layers,
each corresponding to intra-layer similarities between samples in one of data types (gene expression, methylation
and miRNA expression).

::

    sumo prepare -plot LGG.png TCGA-LGG.htseq_fpkm.flt.tsv.gz,TCGA-LGG.methylation450.flt.tsv.gz,TCGA-LGG.mirna.flt.tsv.gz prepared.LGG.npz

Above commands creates 'prepared.LGG.npz' file containing our network and three .png files with plots of adjacency
matrices for each network layer.

--------
sumo run
--------

To identify molecular subtypes in multi-omic supply network file created in last step to sumo *run*. Sumo factorizes
network using symmetric non-negative matrix tri-factorization. You can run sumo by supplying set number of subtypes
to find in data or the range of values to check (in this case we test number of clusters, k = {2, 3, 4}).

::

    sumo run prepared.LGG.npz 2,4 LGG

Sumo creates directory for every k value containing factorization results in form of .npz files and 'clusters.tsv' file
with sample labels. In 'plots' directory you can find plots helpful in selecting accurate number of subtypes in your data.
Stable clustering results is characterized by high value of cophenetic correlation coefficient and low proportion of
ambiguous clusterings.

Output directory structure for above command is shown below.

::

    LGG
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

Use sumo *interpret* to investigate which features drive obtained clustering results.

::

    sumo interpret LGG/k4/sumo_results.npz TCGA-LGG.htseq_fpkm.flt.tsv.gz,TCGA-LGG.methylation450.flt.tsv.gz,TCGA-LGG.mirna.flt.tsv.gz

