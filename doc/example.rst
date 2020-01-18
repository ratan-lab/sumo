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

Data should be prepared for *sumo* analysis using pipeline described below:

.. |preprocessing| raw:: html

    <img src="https://raw.githubusercontent.com/ratan-lab/sumo/development/doc/_images/preprocessing.png" height="200px">

|preprocessing|

.. 1. filtering by missing data (removing features and samples with high amount of missing values) & outliers
.. 2. normalization (data from xena already normalized) => log normalization or converting to M-value
.. 3. standardization (the most important step)

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
    met <- read.table("TCGA-LGG.methylation450.tsv.gz", header=T, check.names = F, row.names = 1)

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

    mirna[1:3,1:3]
    #               TCGA-S9-A6WQ-01A TCGA-S9-A7IS-01A TCGA-S9-A7IQ-01A
    # hsa-let-7a-1         14.38229         13.73226         12.34947
    # hsa-let-7a-2         14.38425         13.72343         12.36370
    # hsa-let-7a-3         14.38227         13.73397         12.36738

    dim(mirna)
    #[1] 1881  530

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

------------
sumo prepare
------------


--------
sumo run
--------

.. PAC, CCC, directory tree

--------------
sumo interpret
--------------


-------------
sumo evaluate
-------------
.. Comparing results of clustering with pre-existing

