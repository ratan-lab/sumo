from lightgbm import LGBMClassifier
from sumo.constants import PREDICT_DEFAULTS
from sumo.modes.predict.predict import SumoPredict
import numpy as np
import os
import pandas as pd
import pickle
import pytest


def _get_args(infiles: list, classifier: str, output_prefix: str):
    args = PREDICT_DEFAULTS.copy()
    args["infiles"] = infiles
    args['classifier'] = classifier
    args['output_prefix'] = output_prefix
    return args


def test_init(tmpdir):
    # incorrect parameters
    with pytest.raises(AttributeError):
        SumoPredict()

    feature1 = os.path.join(tmpdir, "feature1.tsv")
    feature2 = os.path.join(tmpdir, "feature2.tsv")
    classifier = os.path.join(tmpdir, "interpreted.pickle")
    output_prefix = os.path.join(tmpdir, "outfile")

    args = _get_args([feature1, feature2], classifier, output_prefix)

    # no input file
    with pytest.raises(FileNotFoundError):
        SumoPredict(**args)

    f1 = pd.DataFrame(np.random.normal(size=(20, 10)), columns=['sample_' + str(i) for i in range(10)],
                      index=['feature_' + str(i) for i in range(20)])
    f1.to_csv(feature1, sep="\t")
    f2 = pd.DataFrame(np.random.normal(size=(10, 10)), columns=['sample_' + str(i) for i in range(10)],
                      index=['feature_' + str(i) for i in range(10)])
    f2.to_csv(feature2, sep="\t")

    # no classifier file
    with pytest.raises(FileNotFoundError):
        SumoPredict(**args)

    pickle_handle = open(classifier, "wb")
    pickle.dump({"model": LGBMClassifier(), "features": np.array([])}, pickle_handle)
    pickle_handle.close()

    SumoPredict(**args)

    # overwriting output file
    tmp = pd.DataFrame(np.array([]))
    tmp.to_csv("{}.tsv".format(output_prefix))
    SumoPredict(**args)

    # incorrect classifier file (not .pickle)
    args['classifier'] = feature1
    with pytest.raises(ValueError):
        SumoPredict(**args)


def test_run(tmpdir):
    feature1 = os.path.join(tmpdir, "feature1.tsv")
    classifier = os.path.join(tmpdir, "interpreted.pickle")
    output_prefix = os.path.join(tmpdir, "outfile")
    args = _get_args([feature1], classifier, output_prefix)

    f1 = pd.DataFrame(np.random.normal(size=(20, 10)), columns=['sample_' + str(i) for i in range(10)],
                      index=['feature_' + str(i) for i in range(20)])
    f1.to_csv(feature1, sep="\t")

    # incorrect classifier file (no model)
    pickle_handle = open(classifier, "wb")
    pickle.dump({"model": np.array([]), "features": np.array([])}, pickle_handle)
    pickle_handle.close()

    with pytest.raises(ValueError):
        sp = SumoPredict(**args)
        sp.run()

    # incorrect classifier file (model not fitted)
    pickle_handle = open(classifier, "wb")
    pickle.dump({"model": LGBMClassifier(), "features": np.array([])}, pickle_handle)
    pickle_handle.close()

    with pytest.raises(ValueError):
        sp = SumoPredict(**args)
        sp.run()

    m = LGBMClassifier()
    m.fit(f1.values.T, np.random.choice([0, 1], f1.shape[1]))

    # incorrect classifier file (incorrect size of features array)
    pickle_handle = open(classifier, "wb")
    pickle.dump({"model": m, "features": np.array(['feature_' + str(i) for i in range(10)])}, pickle_handle)
    pickle_handle.close()

    with pytest.raises(ValueError):
        sp = SumoPredict(**args)
        sp.run()

    pickle_handle = open(classifier, "wb")
    pickle.dump({"model": m, "features": np.array(['feature_' + str(i) for i in range(20)])}, pickle_handle)
    pickle_handle.close()
    sp = SumoPredict(**args)
    sp.run()

    for fname in [os.path.join(tmpdir, "{}.tsv".format(output_prefix)),
                  os.path.join(tmpdir, "{}.labels.tsv".format(output_prefix))]:
        assert os.path.exists(fname)
