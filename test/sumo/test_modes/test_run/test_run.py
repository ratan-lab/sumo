from sumo.constants import RUN_DEFAULTS
from sumo.modes.run.run import SumoRun
from sumo.utils import save_arrays_to_npz
import numpy as np
import os
import pytest


def _get_args(infile: str, k: list, outdir: str):
    args = RUN_DEFAULTS.copy()
    args['outdir'] = outdir
    args['k'] = k
    args["infile"] = infile
    return args


def test_init(tmpdir):
    # incorrect parameters
    with pytest.raises(AttributeError):
        SumoRun()

    fname = os.path.join(tmpdir, "data.npz")
    outdir = os.path.join(tmpdir, "outdir")
    samples = 10
    sample_labels = ['sample_{}'.format(i) for i in range(samples)]
    args = _get_args(fname, [2], outdir)

    # no input file
    with pytest.raises(FileNotFoundError):
        SumoRun(**args)

    save_arrays_to_npz({'0': np.random.random((samples, samples)),
                        '1': np.random.random((samples, samples)),
                        'samples': np.array(sample_labels)}, fname)

    # incorrect number of repetitions
    args['n'] = -1
    with pytest.raises(ValueError):
        SumoRun(**args)

    # incorrect number of threads
    args = _get_args(fname, [2], outdir)
    args['t'] = -1
    with pytest.raises(ValueError):
        SumoRun(**args)

    # incorrect outdir
    args = _get_args(fname, [2], fname)
    with pytest.raises(NotADirectoryError):
        SumoRun(**args)

    # incorrect k
    args = _get_args(fname, [2, 3, 4], outdir)
    with pytest.raises(ValueError):
        SumoRun(**args)

    args = _get_args(fname, [2], outdir)
    SumoRun(**args)

    args = _get_args(fname, [2, 5], outdir)
    SumoRun(**args)

    # incorrect k range
    args = _get_args(fname, [5, 2], outdir)
    with pytest.raises(ValueError):
        SumoRun(**args)


def test_run(tmpdir):
    fname = os.path.join(tmpdir, "data.npz")
    outdir = os.path.join(tmpdir, "outdir")
    samples = 10
    a0 = np.random.random((samples, samples))
    a0 = (a0 * a0.T) / 2
    a1 = np.random.random((samples, samples))
    a1 = (a1 * a1.T) / 2
    sample_labels = ['sample_{}'.format(i) for i in range(samples)]

    args = _get_args(fname, [2], outdir)

    # no sample names
    save_arrays_to_npz({'0': a0, '1': a1}, fname)
    with pytest.raises(ValueError):
        sr = SumoRun(**args)
        sr.run()

    # incorrect sample names
    save_arrays_to_npz({'0': a0, '1': a1, 'samples': np.array(sample_labels[1:])}, fname)
    with pytest.raises(ValueError):
        sr = SumoRun(**args)
        sr.run()

    # incorrect adjacency matrices
    save_arrays_to_npz({'samples': np.array(sample_labels)}, fname)
    with pytest.raises(ValueError):
        sr = SumoRun(**args)
        sr.run()

    # incorrect value of h_init
    save_arrays_to_npz({'0': a0, '1': a1, 'samples': np.array(sample_labels)}, fname)

    args = _get_args(fname, [2], outdir)
    args['h_init'] = -1
    with pytest.raises(ValueError):
        sr = SumoRun(**args)
        sr.run()

    args['h_init'] = 3
    with pytest.raises(ValueError):
        sr = SumoRun(**args)
        sr.run()

    args = _get_args(fname, [2], outdir)
    args['sparsity'] = [10]
    args['n'] = 10  # makes test run quicker
    sr = SumoRun(**args)
    sr.run()

    assert all([os.path.exists(os.path.join(outdir, x)) for x in ['k2', 'plots',
                                                                  os.path.join('plots', 'consensus_k2.png'),
                                                                  os.path.join('k2', 'sumo_results.npz')]])
