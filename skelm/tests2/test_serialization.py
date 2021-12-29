import pytest
import tempfile
from sklearn.datasets import load_boston, load_iris
from sklearn.preprocessing import RobustScaler
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, lil_matrix
import pickle

from skelm import ELMRegressor, ELMClassifier


@pytest.fixture
def data_class():
    return load_iris(return_X_y=True)


@pytest.fixture
def data_reg():
    return load_boston(return_X_y=True)


def test_Serialize_Solver(data_reg):
    X, Y = data_reg
    elm = ELMRegressor(random_state=0)
    elm.fit(X, Y)
    Yh1 = elm.predict(X)

    solver_data = pickle.dumps(elm.solver_, pickle.HIGHEST_PROTOCOL)
    del elm.solver_

    elm.solver_ = pickle.loads(solver_data)
    Yh2 = elm.predict(X)
    assert Yh1 == pytest.approx(Yh2)


def test_Serialize_HiddenLayer(data_class):
    X, Y = data_class
    elm = ELMClassifier(
        n_neurons=(5,6,7), ufunc=('tanh', None, 'sigm'), density=(None, None, 0.5),
        pairwise_metric=(None, 'l1', None), random_state=0)
    elm.fit(X, Y)
    Yh1 = elm.predict(X)

    hl_data = [pickle.dumps(hl, pickle.HIGHEST_PROTOCOL) for hl in elm.hidden_layers_]
    del elm.hidden_layers_

    elm.hidden_layers_ = [pickle.loads(z) for z in hl_data]
    Yh2 = elm.predict(X)
    assert Yh1 == pytest.approx(Yh2)


def test_Serialize_ELM(data_class):
    X, Y = data_class
    elm = ELMClassifier(
        n_neurons=(5,6,7), ufunc=('tanh', None, 'sigm'), density=(None, None, 0.5),
        pairwise_metric=(None, 'l1', None), random_state=0)
    elm.fit(X, Y)
    Yh1 = elm.predict(X)

    elm_data = pickle.dumps(elm, pickle.HIGHEST_PROTOCOL)
    elm2 = pickle.loads(elm_data)

    Yh2 = elm.predict(X)
    assert Yh1 == pytest.approx(Yh2)


def test_Serialize_ContinueTraining(data_class):
    X, Y = data_class
    x1, y1 = X[0::2], Y[0::2]
    x2, y2 = X[1::2], Y[1::2]

    elm1 = ELMClassifier(n_neurons=10, random_state=0)
    elm1.fit(X, Y)
    Yh1 = elm1.predict(X)

    elm2 = ELMClassifier(n_neurons=10, random_state=0)
    elm2.fit(x1, y1)
    elm2_data = pickle.dumps(elm2, pickle.HIGHEST_PROTOCOL)
    del elm2
    elm2_loaded = pickle.loads(elm2_data)
    elm2_loaded.partial_fit(x2, y2)
    Yh2 = elm2_loaded.predict(X)

    assert Yh1 == pytest.approx(Yh2)


def test_SerializeToFile_ContinueTraining(data_class):
    X, Y = data_class
    x1, y1 = X[0::2], Y[0::2]
    x2, y2 = X[1::2], Y[1::2]

    elm1 = ELMClassifier(n_neurons=10, random_state=0)
    elm1.fit(X, Y)
    Yh1 = elm1.predict(X)

    elm2 = ELMClassifier(n_neurons=10, random_state=0)
    elm2.fit(x1, y1)
    with tempfile.TemporaryFile() as ftemp:
        pickle.dump(elm2, ftemp, pickle.HIGHEST_PROTOCOL)
        del elm2
        ftemp.seek(0)
        elm2_reloaded = pickle.load(ftemp)

    elm2_reloaded.partial_fit(x2, y2)
    Yh2 = elm2_reloaded.predict(X)

    assert Yh1 == pytest.approx(Yh2)