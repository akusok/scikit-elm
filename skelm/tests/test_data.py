import pytest
from sklearn.datasets import load_iris, load_boston
from skelm import ELMClassifier, ELMRegressor

@pytest.fixture
def data_class():
    return load_iris(return_X_y=True)

@pytest.fixture
def data_reg():
    return load_boston(return_X_y=True)


def test_Classification_Iris_BetterThanNaive(data_class):
    elm = ELMClassifier(random_state=0)
    elm.fit(*data_class)
    score = elm.score(*data_class)
    assert score > 0.33


def test_Regression_Boston_BetterThanNaive(data_reg):
    elm = ELMRegressor(random_state=0)
    elm.fit(*data_reg)
    r2score = elm.score(*data_reg)
    assert r2score > 0.3

