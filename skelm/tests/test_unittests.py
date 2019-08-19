import pytest
from pytest import approx
import numpy as np
from skelm import ELMClassifier, ELMRegressor
from sklearn.datasets import load_iris, load_boston

@pytest.fixture
def data_class():
    return load_iris(return_X_y=True)


@pytest.fixture
def data_reg():
    return load_boston(return_X_y=True)


def test_Classifier_predict_ReturnsIntegerArray():
    X = np.array([[1], [2], [3]])
    Y = np.array([0, 0, 1])
    elm = ELMClassifier().fit(X, Y)
    Yh = elm.predict(X)
    assert isinstance(Yh, np.ndarray)
    assert Yh == approx(Yh.astype(np.int))


def test_Classifier_WrongNumberOfFeatures_RaisesError(data_class):
    X, T = data_class
    elm = ELMClassifier().fit(X, T)
    with pytest.raises(ValueError):
        elm.predict(X[:, 1:])


def test_Regressor_WrongNumberOfFeatures_RaisesError(data_reg):
    X, T = data_reg
    elm = ELMRegressor().fit(X, T)
    with pytest.raises(ValueError):
        elm.predict(X[:, 1:])


def test_Classifier_Multilabel(data_class):
    X, T = data_class
    Y = np.ones((T.shape[0], 2))
    Y[:, 0] = T
    elm = ELMClassifier()
    elm.fit(X, Y)


def test_Classifier_SetClasses_IgnoresOther(data_class):
    X, T = data_class
    elm = ELMClassifier(classes=[0, 1])
    Yh = elm.fit(X, T).predict(X)
    assert set(Yh) == {0, 1}


def test_Classifier_PartialFit(data_class):
    X, T = data_class
    elm0 = ELMClassifier(n_neurons=4, alpha=1, random_state=0)
    elm1 = ELMClassifier(n_neurons=4, alpha=1, random_state=0)

    elm0.fit(X, T)
    elm1.partial_fit(X[::2], T[::2])
    elm1.partial_fit(X[1::2], T[1::2])

    assert elm0.solver_.coef_ == approx(elm1.solver_.coef_)


def test_IterativeClassification_FeedClassesOneByOne(data_class):
    X, T = data_class
    elm = ELMClassifier(classes=[0, -1, -2], n_neurons=10, alpha=1)

    X0 = X[T == 0]
    Y0 = T[T == 0]
    elm.partial_fit(X0, Y0)

    X1 = X[T == 1]
    Y1 = T[T == 1]
    elm.partial_fit(X1, Y1, update_classes=True)

    X2 = X[T == 2]
    Y2 = T[T == 2]
    elm.partial_fit(X2, Y2, update_classes=True)

    Yh = elm.predict(X)
    assert set(Yh) == {0, 1, 2}


def test_IterativeSolver_SkipIntermediateSolution(data_class):
    X, T = data_class
    elm = ELMClassifier(classes=[0, 1, 2], n_neurons=10, alpha=1)

    X0 = X[T == 0]
    Y0 = T[T == 0]
    elm.partial_fit(X0, Y0, compute_output_weights=False)

    X1 = X[T == 1]
    Y1 = T[T == 1]
    elm.partial_fit(X1, Y1, compute_output_weights=False)

    X2 = X[T == 2]
    Y2 = T[T == 2]
    elm.partial_fit(X2, Y2)

    Yh = elm.predict(X)
    assert set(Yh) == {0, 1, 2}


def test_MultipleHiddenLayers(data_reg):
    X, Y = data_reg
    elm = ELMRegressor(n_neurons=[2, 3], ufunc=['tanh', 'sigm'],
                       density=[None, None], pairwise_metric=[None, None])
    elm.fit(X, Y)
    assert len(elm.hidden_layers_) == 2


def test_MultipleHiddenLayers_MoreCombinations(data_reg):
    X, Y = data_reg
    elm = ELMRegressor(n_neurons=[1, 1, 1, 1, 1],
                       ufunc=['relu', 'sigm', np.sin, None, None],
                       density=[None, 0.5, 0.8, None, None],
                       pairwise_metric=[None, None, None, 'l1', 'chebyshev'])
    elm.fit(X, Y)
    assert len(elm.hidden_layers_) == 5

def test_MultipleHL_DefaultValues(data_reg):
    X, Y = data_reg
    elm = ELMRegressor(n_neurons=[2, 3])
    elm.fit(X, Y)

def test_MultipleHL_Ufunc_SingleValue(data_reg):
    X, Y = data_reg
    elm = ELMRegressor(n_neurons=[2, 3], ufunc='sigm')
    elm.fit(X, Y)

def test_MultipleHL_Density_SingleValue(data_reg):
    X, Y = data_reg
    elm = ELMRegressor(n_neurons=[2, 3], density=0.7)
    elm.fit(X, Y)

def test_MultipleHL_Pairwise_SingleValue(data_reg):
    X, Y = data_reg
    elm = ELMRegressor(n_neurons=[2, 3], pairwise_metric='l2')
    elm.fit(X, Y)

def test_MultipleHL_WrongDimensions_Raises(data_reg):
    X, Y = data_reg
    elm = ELMRegressor(n_neurons=[1, 2, 3, 4], ufunc=['relu', 'sigm'])
    with pytest.raises(ValueError):
        elm.fit(X, Y)

def test_RegularizationAlpha_NegativeValue_Raises(data_class):
    X, Y = data_class
    elm = ELMClassifier(alpha=-1)
    with pytest.raises(ValueError):
        elm.fit(X, Y)