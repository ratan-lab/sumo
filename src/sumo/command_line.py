from sumo.constants import __version__, CLUSTER_METHODS, LOG_LEVELS, SIMILARITY_METHODS, VAR_TYPES, CLUSTER_METRICS, \
    PREPARE_DEFAULTS, EVALUATE_DEFAULTS, RUN_DEFAULTS
from sumo.modes import SUMO_COMMANDS
import argparse


def add_prepare_command_options(subparsers):
    """ Add subparser for 'prepare' command """

    description = "Generate similarity matrices for samples based on biological data"
    prepare_parser = subparsers.add_parser('prepare', description=description,
                                           help='data pre-processing, create a multiplex network file')

    prepare_parser.add_argument('infiles', metavar='infile1,infile2,...',
                                type=lambda s: [i for i in s.split(',')],
                                help='comma-delimited list of paths to input .npz or .txt files (all input files ' +
                                     'should be structured in following way: consecutive samples in columns, ' +
                                     'consecutive features in rows")')

    prepare_parser.add_argument('vars', metavar='var1(,var2,...)',
                                type=lambda s: [i for i in s.split(',')],
                                help="either one variable type for every data matrix in input file(s) or " +
                                     "comma-delimited list of variable types {}".format(VAR_TYPES))

    prepare_parser.add_argument('outfile', metavar='outfile.npz',
                                type=str, help='path to output .npz file')

    prepare_parser.add_argument('-method', action='store', choices=SIMILARITY_METHODS,
                                type=str, required=False, default=PREPARE_DEFAULTS["method"],
                                help='method of sample-sample similarity calculation (default of "%(default)s")')

    prepare_parser.add_argument('-k', action='store',
                                type=float, required=False, default=PREPARE_DEFAULTS["k"],
                                help='fraction of nearest neighbours to use for sample similarity calculation using ' +
                                     'RBF method (default of %(default)s)')

    prepare_parser.add_argument('-alpha', action='store',
                                type=float, required=False, default=PREPARE_DEFAULTS["alpha"],
                                help='hypherparameter of RBF similarity kernel (default of %(default)s)')

    prepare_parser.add_argument('-missing', action='store',
                                type=float, required=False, default=PREPARE_DEFAULTS["missing"],
                                help='acceptable fraction of available values for assessment of distance/similarity' +
                                     ' between pairs of samples (default of %(default)s)')
    # TODO: turn this parameter into fraction of missing samples

    prepare_parser.add_argument('-names', action='store',
                                type=str, required=False, default=PREPARE_DEFAULTS["names"],
                                help='optional key of array containing custom sample names in every .npz file ' +
                                     '(if not set ids of samples are used, which can cause problems ' +
                                     'when layers have missing samples)')

    prepare_parser.add_argument('-sn', action='store',
                                type=int, required=False, default=PREPARE_DEFAULTS["sn"],
                                help='index of row with sample names for .txt input files (default of %(default)s)')

    prepare_parser.add_argument('-fn', action='store',
                                type=int, required=False, default=PREPARE_DEFAULTS["fn"],
                                help='index of column with feature names for .txt input files (default of %(default)s)')

    prepare_parser.add_argument('-df', action='store',
                                type=float, required=False, default=PREPARE_DEFAULTS["df"],
                                help='if percentage of missing values for feature exceeds this value, remove feature ' +
                                     '(default of %(default)s)')

    prepare_parser.add_argument('-ds', action='store',
                                type=float, required=False, default=PREPARE_DEFAULTS["ds"],
                                help='if percentage of missing values for sample (that remains after feature ' +
                                     'dropping) exceeds this value, remove sample (default of %(default)s)')

    prepare_parser.add_argument('-logfile', action='store',
                                type=str, required=False, default=PREPARE_DEFAULTS["logfile"],
                                help='path to save log file, by default stdout is used')

    prepare_parser.add_argument('-log', default=PREPARE_DEFAULTS["log"], choices=LOG_LEVELS,
                                help="Sets the logging level (default of %(default)s)")

    prepare_parser.add_argument('-plot', action='store',
                                type=str, required=False, default=PREPARE_DEFAULTS["plot"],
                                help='path to save adjacency matrix heatmap(s), by default plots are displayed on' +
                                     ' screen')


def add_run_command_options(subparsers):
    """ Add subparser for 'run' command """

    description = "Cluster multiplex network using non-negative matrix tri-factorization"
    cluster_parser = subparsers.add_parser('run', description=description,
                                           help='factorize the multiplex network to identify molecular subtypes')

    cluster_parser.add_argument('infile', metavar='infile.npz', type=str,
                                help='input .npz file containing adjacency matrices for every network layer and ' +
                                     'sample names (file created by running program with mode "run") - consecutive ' +
                                     'adjacency arrays in file are indexed in following way: "0", "1" ... and ' +
                                     'index of sample name vector is "samples"')

    cluster_parser.add_argument('k', metavar="k", type=lambda s: [int(i) for i in s.split(',')],
                                help='either one value describing number of clusters or coma-delimited range of ' +
                                     'values to check (sumo will suggest cluster structure based on cophenetic ' +
                                     'correlation coefficient)')

    cluster_parser.add_argument('outdir', type=str,
                                help='path to save output files')

    cluster_parser.add_argument('-sparsity', type=lambda s: [float(i) for i in s.split(',')], required=False,
                                default=RUN_DEFAULTS['sparsity'],
                                help='either one value or coma-delimited list of sparsity penalty values for H matrix' +
                                     ' (sumo will try different values and select the best results; ' +
                                     'default of  %(default)s)')

    cluster_parser.add_argument('-n', action='store',
                                type=int, required=False, default=RUN_DEFAULTS['n'],
                                help='number of repetitions (default of %(default)s)')

    cluster_parser.add_argument('-method', action='store', choices=CLUSTER_METHODS,
                                type=str, required=False, default=RUN_DEFAULTS['method'],
                                help='method of cluster extraction (default of "%(default)s")')

    cluster_parser.add_argument('-max_iter', action='store',
                                type=int, required=False, default=RUN_DEFAULTS['max_iter'],
                                help='maximum number of iterations for factorization (default of %(default)s)')

    cluster_parser.add_argument('-tol', action='store',
                                type=float, required=False, default=RUN_DEFAULTS['tol'],
                                help='if objective cost function value fluctuation (|Δℒ|) is smaller than this value' +
                                     ', stop iterations before reaching max_iter (default of %(default)s)')

    cluster_parser.add_argument('-calc_cost', action='store',
                                type=int, required=False, default=RUN_DEFAULTS['calc_cost'],
                                help='number of steps between every calculation of objective cost function ' +
                                     '(default of %(default)s)')

    cluster_parser.add_argument('-logfile', action='store',
                                type=str, required=False, default=RUN_DEFAULTS['logfile'],
                                help='path to save log file (by default printed to stdout)')

    cluster_parser.add_argument('-log', default=RUN_DEFAULTS['log'], choices=LOG_LEVELS,
                                help="Set the logging level (default of %(default)s)")

    cluster_parser.add_argument('-h_init', action='store',
                                type=int, required=False, default=RUN_DEFAULTS['h_init'],
                                help='index of adjacency matrix to use for H matrix initialization (by default ' +
                                     'using average adjacency)')

    cluster_parser.add_argument('-t', action='store', type=int, default=RUN_DEFAULTS['t'], required=False,
                                help='number of threads (default of %(default)s)')


def add_evaluate_command_options(subparsers):
    """ Add subparser for 'evaluate' command """

    description = "Evaluate clustering results, given set of labels"
    evaluate_parser = subparsers.add_parser('evaluate', description=description,
                                            help='evaluate clustering results')

    evaluate_parser.add_argument('infile', metavar='infile.npz', type=str,
                                 help="input .npz file containing array indexed as 'clusters', with sample names " +
                                      "in first column and clustering labels in second column " +
                                      "(file created by running sumo with mode 'run')")

    evaluate_parser.add_argument('labels', metavar='labels', type=str,
                                 help="either .npy file containing array with sample names in first column and " +
                                      "labels in second column or .npz file (requires using '-npz' option)")

    evaluate_parser.add_argument('-npz', action='store',
                                 type=str, required=False, default=EVALUATE_DEFAULTS['npz'],
                                 help="key of array containing labels in .npz file")

    evaluate_parser.add_argument('-metric', action='store', choices=CLUSTER_METRICS,
                                 type=str, required=False, default=EVALUATE_DEFAULTS['metric'],
                                 help='metric for accuracy evaluation (by default all metrics are calculated)')

    evaluate_parser.add_argument('-logfile', action='store',
                                 type=str, required=False, default=EVALUATE_DEFAULTS['logfile'],
                                 help='path to save log file (by default printed to stdout)')


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="sumo: subtyping tool for multi-omic data")
    parser.add_argument('-v', '--version', action='version', version=str(__version__))

    subparsers = parser.add_subparsers(help="program mode", dest='command')
    map_subparsers = {"prepare": add_prepare_command_options,
                      "run": add_run_command_options,
                      "evaluate": add_evaluate_command_options}
    assert all([command in SUMO_COMMANDS for command in map_subparsers.keys()])

    for mode in map_subparsers.keys():
        map_subparsers[mode](subparsers)

    args = parser.parse_args(args=argv)

    if args.command not in SUMO_COMMANDS:
        parser.print_help()

    return args
