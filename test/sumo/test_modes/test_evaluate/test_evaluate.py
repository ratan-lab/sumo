from sumo.constants import EVALUATE_DEFAULTS
from sumo.modes.evaluate.evaluate import SumoEvaluate
from sumo.utils import save_arrays_to_npz
import numpy as np
import os
import pytest
import re


def _get_args(infile: str, labels: str):
    args = EVALUATE_DEFAULTS.copy()
    args["infile"] = infile
    args["labels"] = labels
    return args


def test_init(tmpdir):
    # incorrect parameters
    with pytest.raises(AttributeError):
        SumoEvaluate()

    fname = os.path.join(tmpdir, "indata.npz")
    labels_npy = os.path.join(tmpdir, "labels.npy")
    args = _get_args(fname, labels_npy)

    # no input file
    with pytest.raises(FileNotFoundError):
        SumoEvaluate(**args)

    samples = 10
    labels = [0] * int(samples / 2) + [1] * int(samples / 2)
    data = np.array([
        ['sample_{}'.format(i) for i in range(samples)], labels]).T
    save_arrays_to_npz({'clusters': data}, fname)

    # no labels file
    with pytest.raises(FileNotFoundError):
        SumoEvaluate(**args)

    np.save(labels_npy, np.array(labels), allow_pickle=True)
    SumoEvaluate(**args)


def test_run(tmpdir):
    fname = os.path.join(tmpdir, "indata.npz")
    other_fname = os.path.join(tmpdir, "other.npz")
    labels_npz = os.path.join(tmpdir, "labels.npz")
    other_labels = os.path.join(tmpdir, "other_labels.npz")
    args = _get_args(fname, labels_npz)

    samples = 10
    labels = [0] * int(samples / 2) + [1] * int(samples / 2)
    data = np.array([
        ['sample_{}'.format(i) for i in range(samples)], labels]).T
    save_arrays_to_npz({'clusters': data}, fname)

    # .npz labels file
    save_arrays_to_npz({'labels': data}, labels_npz)

    # no npz label idx
    with pytest.raises(ValueError):
        se = SumoEvaluate(**args)
        se.run()

    # incorrect npz label idx
    with pytest.raises(ValueError):
        args['npz'] = 'my_labels'
        se = SumoEvaluate(**args)
        se.run()

    # incorrect input file
    save_arrays_to_npz({'other_data': data}, other_fname)
    args = _get_args(other_fname, labels_npz)
    args['npz'] = 'labels'
    with pytest.raises(ValueError):
        se = SumoEvaluate(**args)
        se.run()

    # incorrect label file
    args = _get_args(fname, other_labels)
    args['npz'] = 'labels'
    save_arrays_to_npz({'labels': data[:, 1]}, other_labels)
    with pytest.raises(ValueError):
        se = SumoEvaluate(**args)
        se.run()

    args = _get_args(fname, labels_npz)
    args['npz'] = 'labels'
    logfile = os.path.join(tmpdir, "results.log")
    args['logfile'] = logfile
    se = SumoEvaluate(**args)
    se.run()

    with open(logfile, 'r') as f:
        for line in f.readlines():
            if re.search('ARI', line):
                ari = float(line.split()[1])
                assert ari == 1.0
                break
