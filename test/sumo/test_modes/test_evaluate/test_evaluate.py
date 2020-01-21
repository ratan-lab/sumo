from sumo.constants import EVALUATE_DEFAULTS
from sumo.modes.evaluate.evaluate import SumoEvaluate
import numpy as np
import pandas as pd
import os
import pytest


def _get_args(infile: str, labels: str):
    args = EVALUATE_DEFAULTS.copy()
    args["infile"] = infile
    args["labels_file"] = labels
    return args


def test_init(tmpdir):
    # incorrect parameters
    with pytest.raises(AttributeError):
        SumoEvaluate()

    fname = os.path.join(tmpdir, "indata.tsv")
    labels_file = os.path.join(tmpdir, "labels.tsv")
    args = _get_args(fname, labels_file)

    # no input file
    with pytest.raises(FileNotFoundError):
        SumoEvaluate(**args)

    samples = 10
    labels = [0] * int(samples / 2) + [1] * int(samples / 2)
    data_array = np.array([['sample_{}'.format(i) for i in range(samples)], labels]).T
    data = pd.DataFrame(data_array, columns=['sample', 'label'])
    data.to_csv(fname, sep="\t")

    # no labels file
    with pytest.raises(FileNotFoundError):
        SumoEvaluate(**args)

    data.to_csv(labels_file, sep="\t")
    SumoEvaluate(**args)


def test_run(tmpdir):
    fname = os.path.join(tmpdir, "indata.tsv")
    labels_file = os.path.join(tmpdir, "labels.tsv")
    args = _get_args(fname, labels_file)

    samples = 10
    labels = [0] * int(samples / 2) + [1] * int(samples / 2)
    data_array = np.array([['sample_{}'.format(i) for i in range(samples)], labels]).T
    data = pd.DataFrame(data_array, columns=['sample', 'label'])

    data.to_csv(fname, sep="\t")
    data.to_csv(labels_file, sep="\t")
    se = SumoEvaluate(**args)
    se.run()

    # incorrect file headers
    data.columns = ['A', 'B']
    data.to_csv(fname, sep="\t")
    with pytest.raises(ValueError):
        se = SumoEvaluate(**args)
        se.run()

    # incorrect labels file
    data.columns = ['sample', 'label']
    data.to_csv(fname, sep="\t")
    incorrect_data = data.loc[:, data.columns != 'label']
    incorrect_data.to_csv(labels_file, sep="\t")
    with pytest.raises(ValueError):
        se = SumoEvaluate(**args)
        se.run()

    # incorrect input file
    data.to_csv(labels_file, sep="\t")
    incorrect_data.to_csv(fname, sep="\t")
    with pytest.raises(ValueError):
        se = SumoEvaluate(**args)
        se.run()

    # not all common samples
    data.to_csv(fname, sep="\t")
    data['sample'][0] = 'sample_new'
    data.to_csv(labels_file, sep="\t")
    se = SumoEvaluate(**args)
    se.run()

    # additional columns in files
    data['importance'] = ['very'] * data.shape[0]
    data.to_csv(labels_file, sep="\t")
    se = SumoEvaluate(**args)
    se.run()
    data.to_csv(fname, sep="\t")
    se = SumoEvaluate(**args)
    se.run()

    # labels does not correspond
    data_array = np.array([['other_sample_{}'.format(i) for i in range(samples)], labels]).T
    data = pd.DataFrame(data_array, columns=['sample', 'label'])
    data.to_csv(labels_file, sep="\t")
    with pytest.raises(ValueError):
        se = SumoEvaluate(**args)
        se.run()
