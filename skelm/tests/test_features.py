import pytest
from sklearn.datasets import load_iris, load_boston
from skelm import ELMClassifier, ELMRegressor


@pytest.mark.parametrize(
    "data,elm", [(load_iris(return_X_y=True), ELMClassifier), (load_boston(return_X_y=True), ELMRegressor)]
)
def test_NeuronsAmount_CanSetManually(data, elm):
    elm_3 = elm(n_neurons=3, random_state=0)
    elm_3.fit(*data)
    score_3 = elm_3.score(*data)

    elm_10 = elm(n_neurons=10, random_state=0)
    elm_10.fit(*data)
    score_10 = elm_10.score(*data)

    assert score_10 > score_3
