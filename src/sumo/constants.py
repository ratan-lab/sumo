# command_line
__version__ = "0.1.1"
SUMO_COMMANDS = ["prepare", "run", "evaluate"]

# prepare
CORR_METHODS = ["pearson", "spearman"]
SIMILARITY_METHODS = ["rbf"] + CORR_METHODS
SUPPORTED_EXT = [".txt", ".npz"]
VAR_TYPES = ["continuous", "binary", "categorical"]

PREPARE_ARGS = ["infiles", "vars", "outfile", "method", "k", "alpha", "missing", "names", "sn", "fn", "df", "ds",
                "logfile", "log", "plot"]
PREPARE_DEFAULT_VALS = ["rbf", 0.1, 0.5, 0.1, None, 0, 0, 0.1, 0.1, None, "INFO", None]
PREPARE_DEFAULTS = dict(zip(PREPARE_ARGS[3:], PREPARE_DEFAULT_VALS))  # 3 positional args

# run
CLUSTER_METHODS = ["max_value", "spectral"]
SPARSITY_RANGE = [1e-04, 1e-03, 1e-02, 1e-01, 1, 1e01, 1e02]

RUN_ARGS = ["infile", "k", "outdir", "sparsity", "n", "method", "max_iter", "tol", "calc_cost", "logfile", "log",
            "h_init", "t"]
RUN_DEFAULT_VALS = [SPARSITY_RANGE, 50, "max_value", 500, 1e-5, 20, None, "INFO", None, 1]
RUN_DEFAULTS = dict(zip(RUN_ARGS[3:], RUN_DEFAULT_VALS))  # 3 positional args

# evaluate args
EVALUATE_ARGS = ["infile", "labels", "metric", "logfile", "npz"]
EVALUATE_DEFAULT_VALS = [None, None, None]
EVALUATE_DEFAULTS = dict(zip(EVALUATE_ARGS[2:], EVALUATE_DEFAULT_VALS))  # 2 positional args

# utils
LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING']
CLUSTER_METRICS = ['NMI', 'purity', 'ARI']
