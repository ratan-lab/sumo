from sklearn.preprocessing import StandardScaler
from sumo.modes.prepare.prepare import SumoPrepare
from sumo.constants import PREPARE_DEFAULTS
from sumo.utils import load_npz
from pandas import DataFrame
import numpy as np
import os
import pytest


def _get_args(infiles: list, outfile: str):
    args = PREPARE_DEFAULTS.copy()
    args['outfile'] = outfile
    args["infiles"] = infiles
    return args


def test_init(tmpdir):
    # incorrect parameters
    with pytest.raises(AttributeError):
        SumoPrepare()

    fname_outfile = os.path.join(tmpdir, "outdata.npz")
    txt_infile = os.path.join(tmpdir, "indata.txt")
    args = _get_args([txt_infile], fname_outfile)

    # no input file
    with pytest.raises(FileNotFoundError):
        SumoPrepare(**args)

    # one txt input file
    data = np.random.random((10, 20))
    df = DataFrame(data, columns=['sample_{}'.format(i) for i in range(data.shape[1])],
                   index=['feature_{}'.format(i) for i in range(data.shape[0])])
    df.to_csv(txt_infile, sep=" ")
    args = _get_args([txt_infile], fname_outfile)
    SumoPrepare(**args)

    # unsupported file type
    csv_infile = os.path.join(tmpdir, "indata.csv")
    df.to_csv(csv_infile, sep=",")
    args = _get_args([csv_infile], fname_outfile)
    with pytest.raises(ValueError):
        SumoPrepare(**args)

    # two input files
    args = _get_args([txt_infile, txt_infile], fname_outfile)
    SumoPrepare(**args)

    # unsupported similarity method
    args = _get_args([txt_infile], fname_outfile)
    args['method'] = ['random']
    with pytest.raises(ValueError):
        SumoPrepare(**args)


def test_run(tmpdir):
    infile_1 = os.path.join(tmpdir, "indata1.tsv")
    infile_2 = os.path.join(tmpdir, "indata2.tsv")
    fname_outfile = os.path.join(tmpdir, "outdata.npz")
    logfile = os.path.join(tmpdir, "prepare.log")
    plots = os.path.join(tmpdir, "plot.png")

    sc = StandardScaler()

    f0 = np.random.random((20, 10))
    f0 = sc.fit_transform(f0.T).T
    samples1 = ['sample_{}'.format(i) for i in range(10)]
    df = DataFrame(f0, columns=samples1)
    df.to_csv(infile_1, sep="\t")

    f1 = np.random.random((40, 12))
    f1 = sc.fit_transform(f1.T).T
    samples2 = ['sample_{}'.format(i) for i in range(12)]
    df2 = DataFrame(f1, columns=samples2)
    df2.to_csv(infile_2, sep="\t")

    # incorrect number of similarity methods
    args = _get_args([infile_1, infile_2], fname_outfile)
    args['names'] = 'samples'
    args['logfile'] = logfile
    args['plot'] = plots
    args['method'] = ['euclidean', 'cosine', 'pearson']

    # incorrect number of variable types
    with pytest.raises(ValueError):
        sp = SumoPrepare(**args)
        sp.run()

    args['method'] = ['euclidean', 'cosine']
    sp = SumoPrepare(**args)
    sp.run()

    args['method'] = ['euclidean']
    sp = SumoPrepare(**args)
    sp.run()

    assert os.path.exists(fname_outfile)
    assert os.path.exists(logfile)

    d = load_npz(fname_outfile)
    # file structure
    assert all([x in d.keys() for x in ['0', '1', 'samples']])
    # missing samples
    assert d['0'].shape == (12, 12) and d['1'].shape == (12, 12) and d['samples'].shape[0] == 12
    assert np.sum(np.sum(np.isnan(d['0']), axis=0) == 12) == 2
    assert np.sum(np.sum(np.isnan(d['0']), axis=1) == 12) == 2


def test_load_all_data(tmpdir):
    txt_infile = os.path.join(tmpdir, "indata.txt")
    args = _get_args([txt_infile], os.path.join(tmpdir, "outdata.npz"))

    data_vals = np.random.random((10, 20))
    data = DataFrame(data_vals.T, columns=['sample_{}'.format(i) for i in range(data_vals.shape[0])],
                     index=['feature_{}'.format(i) for i in range(data_vals.shape[1])])
    data.to_csv(txt_infile, sep=" ")

    sc = SumoPrepare(**args)
    matrices = sc.load_all_data()
    fname, mat = matrices[0]
    assert len(matrices) == 1
    assert fname == txt_infile
    assert mat.values.shape == (20, 10)
