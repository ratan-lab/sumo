from sumo.modes.prepare.prepare import SumoPrepare, filter_features_and_samples, load_data_npz, load_data_txt
from sumo.constants import PREPARE_DEFAULTS
from sumo.utils import save_arrays_to_npz, load_npz
from pandas import DataFrame
import numpy as np
import os
import pytest


def _get_args(infiles: list, vars: list, outfile):
    args = PREPARE_DEFAULTS.copy()
    args['outfile'] = outfile
    args['vars'] = vars
    args["infiles"] = infiles
    return args


def test_init(tmpdir):
    # incorrect parameters
    with pytest.raises(AttributeError):
        SumoPrepare()

    npz_infile = os.path.join(tmpdir, "indata.npz")
    fname_outfile = os.path.join(tmpdir, "outdata.npz")
    args = _get_args([npz_infile], ['continuous'], fname_outfile)

    # no input file
    with pytest.raises(FileNotFoundError):
        SumoPrepare(**args)

    # one npz input file
    data = np.random.random((10, 20))
    save_arrays_to_npz({'f': data}, npz_infile)
    SumoPrepare(**args)

    # one txt input file
    txt_infile = os.path.join(tmpdir, "indata.txt")
    df = DataFrame(data, columns=['sample_{}'.format(i) for i in range(data.shape[1])],
                   index=['feature_{}'.format(i) for i in range(data.shape[0])])
    df.to_csv(txt_infile, sep=" ")
    args = _get_args([txt_infile], ['continuous'], fname_outfile)
    SumoPrepare(**args)

    # unsupported file type
    csv_infile = os.path.join(tmpdir, "indata.csv")
    df.to_csv(csv_infile, sep=",")
    args = _get_args([csv_infile], ['continuous'], fname_outfile)
    with pytest.raises(ValueError):
        SumoPrepare(**args)

    # two input files
    args = _get_args([npz_infile, npz_infile], ['continuous'], fname_outfile)
    SumoPrepare(**args)

    # two different variable types
    args = _get_args([npz_infile, npz_infile], ["continuous", "categorical"], fname_outfile)
    SumoPrepare(**args)

    # more variable types than input files
    args = _get_args([npz_infile], ["continuous", "categorical"], fname_outfile)
    with pytest.raises(ValueError):
        SumoPrepare(**args)

    # unsupported variable type
    args = _get_args([npz_infile], ['random'], fname_outfile)
    with pytest.raises(ValueError):
        SumoPrepare(**args)


def test_run(tmpdir):
    npz_infile_1 = os.path.join(tmpdir, "indata1.npz")
    npz_infile_2 = os.path.join(tmpdir, "indata2.npz")
    fname_outfile = os.path.join(tmpdir, "outdata.npz")
    logfile = os.path.join(tmpdir, "prepare.log")
    plots = os.path.join(tmpdir, "plot.png")

    f0 = np.random.random((20, 10))
    samples1 = ['sample_{}'.format(i) for i in range(10)]
    save_arrays_to_npz({'f': f0, 'samples': np.array(samples1)}, npz_infile_1)
    f1 = np.random.random((40, 12))
    samples2 = ['sample_{}'.format(i) for i in range(12)]
    save_arrays_to_npz({'f': f1, 'samples': np.array(samples2)}, npz_infile_2)

    args = _get_args([npz_infile_1, npz_infile_2], ['continuous', 'continuous', 'continuous'], fname_outfile)
    args['names'] = 'samples'
    args['logfile'] = logfile
    args['plot'] = plots

    # incorrect number of variable types
    with pytest.raises(ValueError):
        sp = SumoPrepare(**args)
        sp.run()

    args['vars'] = ['continuous', 'continuous']
    sp = SumoPrepare(**args)
    sp.run()

    args['vars'] = ['continuous']
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
    args = _get_args([txt_infile], ['continuous'], os.path.join(tmpdir, "outdata.npz"))

    data_vals = np.random.random((10, 20))
    data = DataFrame(data_vals.T, columns=['sample_{}'.format(i) for i in range(data_vals.shape[0])],
                     index=['feature_{}'.format(i) for i in range(data_vals.shape[1])])
    data.to_csv(txt_infile,
                sep="\t")  # TODO: change to accept tsv (interpret txt as space delimited), change command line doc string

    sc = SumoPrepare(**args)
    matrices = sc.load_all_data()
    fname, mat = matrices[0]
    assert len(matrices) == 1
    assert fname == txt_infile
    assert mat.values.shape == (20, 10)


def test_filter_features_and_samples():
    data_vals = np.random.random((10, 20))
    data = DataFrame(data_vals.T, columns=['sample_{}'.format(i) for i in range(data_vals.shape[0])],
                     index=['feature_{}'.format(i) for i in range(data_vals.shape[1])])

    filtered = filter_features_and_samples(data)
    assert filtered.values.shape == (20, 10)

    # missing samples and features
    new_data = data.copy()
    new_data['sample_0'] = np.nan
    new_data.loc['feature_0'] = np.nan

    filtered = filter_features_and_samples(new_data)
    assert 'feature_0' not in filtered.index
    assert 'sample_0' not in filtered.columns

    # missing values sample filtering
    new_data = data.copy()
    new_data['sample_0'][1:3] = np.nan

    filtered = filter_features_and_samples(new_data)
    assert 'sample_0' in filtered.columns

    filtered = filter_features_and_samples(new_data, drop_samples=0.05)
    assert 'sample_0' not in filtered.columns

    # missing values feature filtering
    new_data = data.copy()
    new_data.loc['feature_0'][1] = np.nan

    filtered = filter_features_and_samples(new_data)
    assert 'feature_0' in filtered.index

    filtered = filter_features_and_samples(new_data, drop_features=0.05)
    assert 'feature_0' not in filtered.index


def test_load_data_txt(tmpdir):
    fname = os.path.join(tmpdir, "data.npz")
    with pytest.raises(FileNotFoundError):
        load_data_txt(file_path=fname)

    # empty data
    empty_data = DataFrame()
    empty_data.to_csv(fname, sep="\t")
    with pytest.raises(ValueError):
        load_data_txt(file_path=fname)

    data_vals = np.random.random((10, 20))
    data = DataFrame(data_vals.T, columns=['sample_{}'.format(i) for i in range(data_vals.shape[0])],
                     index=['feature_{}'.format(i) for i in range(data_vals.shape[1])])

    # incorrectly formatted data
    data.to_csv(fname, sep=",")
    with pytest.raises(ValueError):
        load_data_txt(file_path=fname)

    # non-nummerical values
    data.to_csv(fname, sep="\t")
    with pytest.raises(ValueError):
        load_data_txt(file_path=fname)

    load_data_txt(file_path=fname, sample_names=0, feature_names=0)


def test_load_data_npz(tmpdir):
    fname = os.path.join(tmpdir, "data.npz")
    with pytest.raises(FileNotFoundError):
        load_data_npz(file_path=fname)

    # empty data
    save_arrays_to_npz({}, file_path=fname)
    with pytest.raises(ValueError):
        load_data_npz(file_path=fname)

    # no sample names
    data_vals = np.random.random((10, 20))
    save_arrays_to_npz({'f': data_vals.T}, file_path=fname)
    loaded = load_data_npz(file_path=fname)
    assert len(loaded) == 1
    assert loaded[0].shape == (20, 10)

    # sample names
    sample_names = ['sample_{}'.format(i) for i in range(data_vals.shape[0])]
    save_arrays_to_npz({'f': data_vals.T, 'samples': np.array(sample_names)}, file_path=fname)
    loaded = load_data_npz(file_path=fname, sample_idx='samples')
    assert all([loaded[0].columns[i] == sample_names[i] for i in range(len(sample_names))])

    # missing sample names idx
    with pytest.raises(AttributeError):
        load_data_npz(file_path=fname)

    # incorrect sample names idx
    with pytest.raises(ValueError):
        load_data_npz(file_path=fname, sample_idx='sample_names')

    # incorrect sample names
    sample_names = ['sample_{}'.format(i) for i in range(100)]
    save_arrays_to_npz({'f': data_vals.T, 'samples': np.array(sample_names)}, file_path=fname)
    with pytest.raises(ValueError):
        load_data_npz(file_path=fname, sample_idx='sample_names')
