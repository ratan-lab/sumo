# command_line
__version__ = "0.1.0"
SUMO_COMMANDS = ["prepare", "run", "evaluate"]

# prepare
CORR_METHODS = ["pearson", "spearman"]
SIMILARITY_METHODS = ["rbf"] + CORR_METHODS
SUPPORTED_EXT = [".txt", ".npz"]
PREPARE_ARGS = ["infiles", "vars", "outfile", "method", "k", "alpha", "missing", "names", "sn", "fn", "df", "ds",
                "logfile", "log", "plot"]
VAR_TYPES = ["continuous", "binary", "categorical"]

# run
RUN_ARGS = ["infile", "k", "outdir", "sparsity", "n", "method", "max_iter", "tol", "calc_cost", "logfile", "log",
            "h_init", "t"]
CLUSTER_METHODS = ["max_value", "spectral"]
SPARSITY_RANGE = [1e-04, 1e-03, 1e-02, 1e-01, 1, 1e01, 1e02]

# evaluate args
EVALUATE_ARGS = ["infile", "labels", "metric", "logfile", "npz"]

# utils
LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING']
CLUSTER_METRICS = ['NMI', 'purity', 'ARI']
