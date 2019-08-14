import pytest
import numpy as np
from sklearn.datasets import load_iris, load_boston

from skelm import ELMClassifier, ELMRegressor

pd = pytest.importorskip("pandas")  # tests if Pandas is installed

@pytest.fixture
def data_class():
    X, y = load_iris(return_X_y=True)
    return pd.DataFrame(X), pd.DataFrame(y)

@pytest.fixture
def data_reg():
    X, y = load_boston(return_X_y=True)
    return pd.DataFrame(X), pd.DataFrame(y)


def test_Pandas_ActuallyUsesDataFrames(data_class, data_reg):
    X, y = data_class
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)

    X, y = data_reg
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)


def test_Classifier_FitOnPandas_ReturnsNumpy(data_class):
    X, Y = data_class
    elm = ELMClassifier().fit(X, Y)
    Yh = elm.predict(X)
    assert isinstance(Yh, np.ndarray)

def test_Classification_BetterThanNaive(data_class):
    X, Y = data_class
    elm = ELMClassifier(random_state=0).fit(X, Y)
    score = elm.score(X, Y)
    assert score > 0.33

def test_Regressor_FitOnPandas_ReturnsNumpy(data_reg):
    X, Y = data_reg
    elm = ELMRegressor().fit(X, Y)
    Yh = elm.predict(X)
    assert isinstance(Yh, np.ndarray)

def test_Regression_BetterThanNaive(data_reg):
    X, Y = data_reg
    elm = ELMRegressor(random_state=0).fit(X, Y)
    r2score = elm.score(X, Y)
    assert r2score > 0.3
