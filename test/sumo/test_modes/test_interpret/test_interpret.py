from sumo.constants import INTERPRET_DEFAULTS
from sumo.modes.interpret.interpret import SumoInterpret
from sumo.utils import save_arrays_to_npz
import numpy as np
import pandas as pd
import os
import pytest


def _get_args(sumo_results: str, labels_file: list, outfile: str):
    args = INTERPRET_DEFAULTS.copy()
    args["sumo_results"] = sumo_results
    args["infiles"] = labels_file
    args["output_prefix"] = outfile
    return args


def test_init(tmpdir):
    # incorrect parameters
    with pytest.raises(AttributeError):
        SumoInterpret()

    fname = os.path.join(tmpdir, "indata.npz")
    feature1 = os.path.join(tmpdir, "feature1.tsv")
    feature2 = os.path.join(tmpdir, "feature2.tsv")
    output_prefix = os.path.join(tmpdir, "outfile")
    args = _get_args(fname, [feature1, feature2], output_prefix)

    # no input file
    with pytest.raises(FileNotFoundError):
        SumoInterpret(**args)

    samples = 10
    labels = [0] * int(samples / 2) + [1] * int(samples / 2)
    data_array = np.array([['sample_{}'.format(i) for i in range(samples)], labels]).T
    data = pd.DataFrame(data_array, columns=['sample', 'label'])
    save_arrays_to_npz({'clusters': data.values}, file_path=fname)

    # no feature files
    with pytest.raises(FileNotFoundError):
        SumoInterpret(**args)

    f1 = pd.DataFrame(np.random.normal(size=(20, 10)), columns=['sample_' + str(i) for i in range(10)],
                      index=['feature_' + str(i) for i in range(20)])
    f1.to_csv(feature1, sep="\t")
    f2 = pd.DataFrame(np.random.normal(size=(10, 10)), columns=['sample_' + str(i) for i in range(10)],
                      index=['feature_' + str(i) for i in range(10)])
    f2.to_csv(feature2, sep="\t")

    # overwriting output file
    tmp = pd.DataFrame(np.array([]))
    tmp.to_csv("{}.tsv".format(output_prefix))
    SumoInterpret(**args)

    # incorrect number of threads
    args['t'] = 0
    with pytest.raises(ValueError):
        SumoInterpret(**args)


def test_run(tmpdir):
    fname = os.path.join(tmpdir, "indata.npz")
    features = os.path.join(tmpdir, "features.tsv")
    outfile = os.path.join(tmpdir, "outfile.tsv")
    args = _get_args(fname, [features], outfile)

    n_samples, n_features = 100, 20
    f = pd.DataFrame(np.random.normal(size=(n_features, n_samples)),
                     columns=['sample_' + str(i) for i in range(n_samples)],
                     index=['feature_' + str(i) for i in range(n_features)])
    f.to_csv(features, sep="\t")

    labels = [0] * int(n_samples / 2) + [1] * int(n_samples / 2)
    data_array = np.array([['sample_{}'.format(i) for i in range(n_samples)], labels]).T
    data = pd.DataFrame(data_array, columns=['sample', 'label'])

    save_arrays_to_npz({'my_values': data.values}, file_path=fname)
    with pytest.raises(ValueError):
        si = SumoInterpret(**args)
        si.run()

    # two classes (special binary case)
    save_arrays_to_npz({'clusters': data.values}, file_path=fname)
    si = SumoInterpret(**args)
    si.run()

    # three classes
    labels[-10:] = [2] * len(labels[-10:])
    data_array = np.array([['sample_{}'.format(i) for i in range(n_samples)], labels]).T
    data = pd.DataFrame(data_array, columns=['sample', 'label'])
    save_arrays_to_npz({'clusters': data.values}, file_path=fname)
    si = SumoInterpret(**args)
    si.run()
