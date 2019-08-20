import pytest
from pytest import approx
import numpy as np
from sklearn.datasets import load_iris, make_multilabel_classification, load_diabetes
from skelm import ELMRegressor, ELMClassifier


@pytest.fixture
def data_class():
    return load_iris(return_X_y=True)

@pytest.fixture
def data_ml():
    return make_multilabel_classification()

@pytest.fixture
def data_reg():
    return load_diabetes(return_X_y=True)

def test_SineWave_Solves():
    """A highly non-linear regression problem, with added strong noise.
    """
    X = np.linspace(-1, 1, num=1000)[:, None]
    Y = np.sin(16 * X) * X + 0.2*np.random.randn(1000)[:, None]

    elm = ELMRegressor(random_state=0)
    elm.fit(X, Y)
    Yt = elm.predict(X)

    MSE = np.mean((Y - Yt) ** 2)
    assert MSE < 0.3


def test_Xor_OneNeuron_Solved():
    """ELM should be able to solve XOR problem.
    """
    X = np.array([[0, 0],
                  [1, 1],
                  [1, 0],
                  [0, 1]])
    Y = np.array([1, 1, -1, -1])

    elm = ELMClassifier(n_neurons=3, random_state=0)
    elm.fit(X, Y)
    Yh = elm.predict(X)
    assert Yh[0] > 0
    assert Yh[1] > 0
    assert Yh[2] < 0
    assert Yh[3] < 0


def test_ELMClassifier_ReportedScore_ActuallyIsClassificationScore(data_class):
    X, Y = data_class
    Yr = np.vstack((Y == 0, Y == 1, Y == 2)).T

    elm_c = ELMClassifier(random_state=0).fit(X, Y)
    elm_r = ELMRegressor(random_state=0).fit(X, Yr)

    Yc_hat = elm_c.predict(X)
    Yr_hat = elm_r.predict(X).argmax(1)

    assert Yc_hat == approx(Yr_hat)


def test_ELMClassifier_MultilabelClassification_Works(data_ml):
    X, Y = data_ml
    elm_c = ELMClassifier(random_state=0).fit(X, Y)
    elm_r = ELMRegressor(random_state=0).fit(X, Y)

    Yc_hat = elm_c.predict(X)
    Yr_hat = (elm_r.predict(X) >= 0.5).astype(np.int)

    assert Yc_hat == approx(Yr_hat)


def test_RegularizationL2_DifferentValue_ChangesPrediction(data_reg):
    X, Y = data_reg
    Yh_1 = ELMRegressor(alpha=1e-7, random_state=0).fit(X, Y).predict(X)
    Yh_2 = ELMRegressor(alpha=1e+3, random_state=0).fit(X, Y).predict(X)

    assert Yh_1 != approx(Yh_2)
