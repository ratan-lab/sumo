*************
Example usage
*************

In this example, we will use SUMO to stratify patients diagnosed with Acute Myeloid Leukemia (LAML) into sub-groups based on gene-expression, micro-RNA expression, and methylation datasets. We will describe steps that should be taken for pre-processing of these data, use SUMO to detect the various sub-groups of patients, and identify the features e.g., genes and methylation probes, that drive each of those subgroups.

We will use `gene expression <https://gdc.xenahubs.net/download/TCGA-LAML.htseq_fpkm.tsv.gz>`_, `methylation <https://gdc.xenahubs.net/download/TCGA-LAML.methylation27.tsv.gz>`_ and `miRNA expression <https://gdc.xenahubs.net/download/TCGA-LAML.mirna.tsv.gz>`_ data from UCSC XENA browser in this python vignette.

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

In this mode, SUMO calculates the pairwise similarity between the samples for each separate input file containing a feature matrix with omic data (in this case gene expression, methylation and miRNA expression).

::

    sumo prepare -plot LAML.png TCGA-LAML.htseq_fpkm.flt.tsv.gz,TCGA-LAML.methylation27.flt.tsv.gz,TCGA-LAML.mirna.flt.tsv.gz prepared.LAML.npz

The above command creates a multiplex network file 'prepared.LAML.npz' containing:

* the pairwise similarities organized as network adjacency matrices in order of input files (arrays: '0', '1', '2')
* input feature matrices (arrays: 'f0', 'f1', 'f2')
* list of sample identifiers in order corresponding to rows/columns of adjacency matrices ('sample' array)

Thanks to the -plot flag SUMO also creates three .png files with plots of the adjacency matrices for each omic datatype.

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

To make subtyping results more robust SUMO uses a resampling-based approach in conjunction with consensus clustering. In this mode, the factorization is repeated multiple times (set by -n flag), with a fraction of samples (set by -subsample flag) removed from each run. Next, we use random subsets of runs to create multiple (set by -rep flag) weighted consensus matrices, that are utilized for the robust assessment of the factorization results and derivation of final clustering labels.

As presented in the directory structure above SUMO creates an .npz result file for each (k, eta) pair, where k is a set number of clusters and eta is a factorization sparsity value (can be modified by -sparsity flag). Each such file contains:

* calculated clustering stability metrics for each consensus matrix: the proportion of ambiguous clusterings and cophenetic correlation coefficient ('pac' and 'cophenet' arrays respectively)
* quality metric assessing the within-cluster similarities based on final sample labels, used for sparsity parameter selection ('quality' array)
* selected consensus matrix ('unfiltered_consensus' array) and its copy used for final sample label assignment after the noise filtering ('consensus' array)
* final sample labels ('clusters' array)
* number of iterations/steps reached in each solver run ('steps' array)
* simulation parameters ('config' array)

Adding -log DEBUG flag when running SUMO 'run' mode, results in additional arrays (saved in .npz files) and plots displaying the number of iterations reached by sumo for each "k" (saved to 'plots' directory).

Following additional arrays are added to each .npz file:

* every weighted consensus matrix used for the calculation of stability metrics (arrays: 'pac_consensus_0', 'pac_consensus_1'...)
* indices of solver runs used to create each consensus matrix (arrays: 'runs_0', 'runs_1', ...)
* results of each factorization run:
    * final cost function value (array 'costi' for run 'i')
    * final H matrix (array 'hi' for run 'i')
    * final S matrix for each data type (array 'sij' for data type 'i' and run 'j')
    * indices of fraction of samples used in the factorization run (arrays: 'samples0', 'samples1', ...)

--------------
sumo interpret
--------------

Use SUMO *interpret* mode to investigate which features drive obtained clustering results.

::

    sumo interpret LAML/k4/sumo_results.npz TCGA-LAML.htseq_fpkm.flt.tsv.gz,TCGA-LAML.methylation27.flt.tsv.gz,TCGA-LAML.mirna.flt.tsv.gz LAML_features


The above command generates a file two files "LAML_features.tsv" and "LAML_features.hits.tsv" which report the importance of each feature in supporting cluster separation. Briefly, we train a LightGBM model (https://github.com/microsoft/LightGBM) based on the clusters identified by SUMO, and the results from this mode are the SHAP (SHapley Additive exPlanations) feature importance deduced using that model.

For example, here the results shows that the following top 10 features support various clusters:

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

\

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

============================================
Including somatic mutations in SUMO analysis
============================================

Calculation of similarity distances is challenging for sparse data types such as somatic mutation. Feature selection e.g., limiting to the genes that are known to play a role in the disease, or feature transformation, e.g. mapping to known pathways instead of individual genes can reduce sparsity and improve distance calculations. Another approach is to convert the somatic data into a continuous data matrix appropriate for SUMO.

One way to efficiently convert somatic mutation data into a continuous matrix is to not only consider the gene's self-characteristic (in this case a mutation presence/absence) but also its influence on the regulatory/interaction network.

In this example, we use a random walk with restart on the somatic mutation data from LAML patients (see the previous example) on a protein-protein interaction network, creating a continuous matrix that can be included in the SUMO data integration workflow.

In the below R vignettes we use `MC3 public somatic mutation data <https://gdc.cancer.gov/about-data/publications/mc3-2017>`_. The file was subsetted to contain only LAML samples.

.. code-block:: R

    library(biomaRt)
    library(igraph)
    library(STRINGdb)
    library(annotables)
    library(RandomWalkRestartMH)
    library(tidyverse)
    library(ComplexHeatmap)

    # Load the somatic mutation data
    mutations <- read_tsv("mc3.v0.2.8.PUBLIC.LAML.maf.gz", guess_max=1000000)
    patient_ids <- sapply(mutations$Tumor_Sample_Barcode,
                      function(x){spx = strsplit(x, '-')[[1]]; paste(spx[1], spx[2], spx[3],sep="-")})
    mutations$patient_id <- patient_ids

Fetch the protein-protein interactions for H. Sapiens from the `STRING data base <https://string-db.org>`_.

.. code-block:: R

    string_db <- STRINGdb$new( version="10", species=9606, score_threshold=400, input_directory="")
    db <- string_db$get_graph()

Create the gene-peptide identifier mapping with BioMart.

.. code-block:: R

    mart <- useMart(biomart = "ensembl", dataset = "hsapiens_gene_ensembl")
    mapping <- getBM(attributes=c("ensembl_gene_id", "ensembl_transcript_id", "ensembl_peptide_id"), mart=mart) %>%
               mutate(ensembl_peptide_id = paste0("9606.", ensembl_peptide_id)) %>%
               filter(ensembl_peptide_id %in% names(V(db)))

Here we apply an exemplary filtering of the mutation types and possible artifacts based on the `MC3 filters <https://www.synapse.org/#!Synapse:syn7214402/wiki/406007>`_.

.. code-block:: R

    classes_toignore <- c("Intron", "Silent", "5'Flank", "3'UTR", "5'UTR", "3'Flank", "IGR", "RNA")
    mutations <- mutations %>%
        filter(!Variant_Classification %in% classes_toignore) %>% # subset the mutation types
        filter(FILTER != "oxog,wga") # filter out possible artifacts

Create one layer protein-protein interaction (PPI) network.

.. code-block:: R

    PPI_MultiplexObject <- create.multiplex(db,Layers_Name=c("PPI"))
    # To apply the Random Walk with Restart (RWR) on this monoplex network,
    # we need to compute the adjacency matrix of the network and normalize it by column.
    AdjMatrix_PPI <- compute.adjacency.matrix(PPI_MultiplexObject)
    AdjMatrixNorm_PPI <- normalize.multiplex.adjacency(AdjMatrix_PPI)

Apply the Random Walk with Restart (RWR) for each patient based on PPI network with seeds set according to somatic mutation data.

.. code-block:: R

    score_patient <- function(tbl) {
        seedgenes <- tbl$ensembl_peptide_id
        RWR_PPI_Results <- Random.Walk.Restart.Multiplex(AdjMatrixNorm_PPI, PPI_MultiplexObject,seedgenes)
        seeddf <- tibble(NodeNames = seedgenes, Score = 1/length(seedgenes))
        outdf <- RWR_PPI_Results$RWRM_Results %>% as_tibble()
        outdf <- bind_rows(outdf, seeddf)
        return(outdf)
    }

    scores <- mutations %>%
        select(patient_id, Gene) %>%
        distinct() %>%
        left_join(mapping, by=c("Gene"="ensembl_gene_id")) %>%
        select(patient_id, ensembl_peptide_id) %>%
        na.omit() %>%
        group_by(patient_id) %>%
        nest() %>%
        mutate(df = map(data, score_patient)) %>%
        select(-data) %>%
        unnest(cols=c(df)) %>%
        ungroup()

Save the result.

.. code-block:: R

    df <- scores %>% spread(NodeNames, Score, fill=0)
    mat <- data.matrix(df %>% select(-patient_id))
    rownames(mat) <- df$patient_id

    write.table(t(mat), file="mutation_scores.tsv", quote=F, sep="\t")

We now have a continuous matrix that can be used as input to *sumo prepare*. The choice of the interaction network can be important, and tissue-specific networks such as those from `HumanBase <https://hb.flatironinstitute.org/>`_ lead to an improvement in results.
