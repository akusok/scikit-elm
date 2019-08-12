import pytest
from pytest import approx
import numpy as np
from skelm import ELMClassifier, ELMRegressor
from sklearn.datasets import load_iris, load_boston


def test_Classifier_predict_ReturnsIntegerArray():
    X = np.array([[1], [2], [3]])
    Y = np.array([0, 0, 1])
    elm = ELMClassifier().fit(X, Y)
    Yh = elm.predict(X)
    assert isinstance(Yh, np.ndarray)
    assert Yh == approx(Yh.astype(np.int))


def test_Classifier_WrongNumberOfFeatures_RaisesError():
    X, T = load_iris(return_X_y=True)
    elm = ELMClassifier().fit(X, T)
    with pytest.raises(ValueError):
        elm.predict(X[:, 1:])


def test_Regressor_WrongNumberOfFeatures_RaisesError():
    X, T = load_boston(return_X_y=True)
    elm = ELMRegressor().fit(X, T)
    with pytest.raises(ValueError):
        elm.predict(X[:, 1:])


def test_Classifier_Multilabel():
    X, T = load_iris(return_X_y=True)
    Y = np.ones((T.shape[0], 2))
    Y[:, 0] = T
    elm = ELMClassifier()
    elm.fit(X, Y)


def test_Classifier_SetClasses_IgnoresOther():
    X, T = load_iris(return_X_y=True)
    elm = ELMClassifier(classes=[0, 1])
    Yh = elm.fit(X, T).predict(X)
    assert set(Yh) == {0, 1}


@pytest.mark.skip("Custom solvers unsupported yet")
def test_ClassifierRidge_PartialFit_NotSupported():
    X, T = load_iris(return_X_y=True)
    elm = ELMClassifier(solver='ridge', random_state=0)
    with pytest.raises(ValueError):
        elm.partial_fit(X, T)

#todo: add custom solver support


@pytest.mark.skip("Partial_fit broken")
def test_Classifier_PartialFit():
    X, T = load_iris(return_X_y=True)
    elm0 = ELMClassifier(n_neurons=10, alpha=1, random_state=0)
    elm1 = ELMClassifier(n_neurons=10, alpha=1, random_state=0)

    elm0.fit(X, T)
    elm1.partial_fit(X[::2], T[::2])
    elm1.partial_fit(X[1::2], T[1::2])

    assert elm0.solver_.coef_ == approx(elm1.solver_.coef_)


@pytest.mark.skip("Partial_fit broken")
def test_IterativeClassification_FeedClassesOneByOne():
    X, T = load_iris(return_X_y=True)
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


@pytest.mark.skip("Partial_fit broken")
def test_IterativeSolver_SkipIntermediateSolution():
    X, T = load_iris(return_X_y=True)
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
