# command_line
__version__ = "0.2.3"
SUMO_COMMANDS = ["prepare", "run", "evaluate", "interpret"]

# prepare
CORR_METHODS = ["pearson", "spearman"]
SIMILARITY_METHODS = ["euclidean", "cosine"] + CORR_METHODS
TXT_EXT = [".txt", '.txt.gz', '.txt.bz2']
TSV_EXT = [".tsv", '.tsv.gz', '.tsv.bz2']
SUPPORTED_EXT = TXT_EXT + TSV_EXT

PREPARE_DEFAULTS = {
    "method": ["euclidean"],
    "k": 0.1,
    "alpha": 0.5,
    "missing": [0.1],
    "sn": 0,
    "fn": 0,
    "df": 0.1,
    "ds": 0.1,
    "logfile": None,
    "log": "INFO",
    "plot": None,
    "atol": 1e-2
}
PREPARE_ARGS = ["infiles", "outfile"] + list(PREPARE_DEFAULTS.keys())  # 3 positional args

# run
CLUSTER_METHODS = ["max_value", "spectral"]
RUN_DEFAULTS = {
    "sparsity": [0.1],
    "n": 50,
    "method": "max_value",
    "max_iter": 500,
    "tol": 1e-5,
    "calc_cost": 20,
    "logfile": None,
    "log": "INFO",
    "h_init": None,
    "t": 1
}
RUN_ARGS = ["infile", "k", "outdir"] + list(RUN_DEFAULTS.keys())  # 3 positional args

# evaluate
EVALUATE_DEFAULTS = {
    "metric": None,
    "logfile": None,
    "log": "INFO"
}
EVALUATE_ARGS = ["infile", "labels_file"] + list(EVALUATE_DEFAULTS.keys())  # 2 positional args

# interpret
INTERPRET_DEFAULTS = {
    "logfile": None,
    "log": "INFO",
    "max_iter": 50,
    "n_folds": 5,
    "sn": 0,
    "fn": 0,
    "df": 0.1,
    "ds": 0.1,
    "t": 1,
    "seed": 1
}
INTERPRET_ARGS = ['sumo_results', 'infiles', 'outfile'] + list(INTERPRET_DEFAULTS.keys())  # 3 positional args

# utils
LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING']
CLUSTER_METRICS = ['NMI', 'purity', 'ARI']
