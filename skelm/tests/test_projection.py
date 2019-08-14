import pytest
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.datasets import load_boston, load_iris
from sklearn.preprocessing import RobustScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, lil_matrix

from skelm import ELMRegressor, ELMClassifier


c_X, c_y = load_iris(return_X_y=True)
r_X, r_y = load_boston(return_X_y=True)
r_X = RobustScaler().fit_transform(r_X)

params = [
    ((r_X, r_y), ELMRegressor),
    ((c_X, c_y), ELMClassifier),
    ((csc_matrix(r_X), r_y), ELMRegressor),
    ((csc_matrix(c_X), c_y), ELMClassifier),
    ((csr_matrix(r_X), r_y), ELMRegressor),
    ((csr_matrix(c_X), c_y), ELMClassifier),
    ((coo_matrix(r_X), r_y), ELMRegressor),
    ((coo_matrix(c_X), c_y), ELMClassifier),
    ((lil_matrix(r_X), r_y), ELMRegressor),
    ((lil_matrix(c_X), c_y), ELMClassifier),
]


@pytest.mark.parametrize("data,elm_model", params)
def test_Default_SetNumberOfNeurons(data, elm_model):
    elm5 = elm_model(n_neurons=5, random_state=0).fit(*data)
    elm50 = elm_model(n_neurons=50, random_state=0).fit(*data)
    score5 = elm5.score(*data)
    score50 = elm50.score(*data)
    assert score50 > score5
    assert score50 > 0.33


@pytest.mark.parametrize("data,elm_model", params)
def test_LinearPart_CanBeIncluded(data, elm_model):
    elm = elm_model(include_original_features=True, random_state=0).fit(*data)
    score = elm.score(*data)
    assert score > 0.33


@pytest.mark.parametrize("data,elm_model", params)
def test_LinearPart_AddsExtraFeatures(data, elm_model):
    X, y = data
    n_neurons_basic = elm_model().fit(X, y).coef_.shape[0]
    n_neurons_with_orig = elm_model(include_original_features=True).fit(X, y).coef_.shape[0]
    assert n_neurons_basic < n_neurons_with_orig


@pytest.mark.parametrize("data,elm_model", params)
def test_DefaultNeurons_UseGaussianRandomProjection(data, elm_model):
    elm = elm_model().fit(*data)
    assert isinstance(elm.hidden_layers_[0].projection_, GaussianRandomProjection)


@pytest.mark.parametrize("data,elm_model", params)
def test_SparseELM_UseSparseRandomProjection(data, elm_model):
    elm = elm_model(density=0.1).fit(*data)
    assert isinstance(elm.hidden_layers_[0].projection_, SparseRandomProjection)


@pytest.mark.parametrize("data,elm_model", params)
def test_Ufunc_Sigmoid(data, elm_model):
    elm = elm_model(ufunc="sigm").fit(*data)
    assert elm.hidden_layers_[0].ufunc_ is sigmoid


@pytest.mark.parametrize("data,elm_model", params)
def test_DefaultUfunc_Tanh(data, elm_model):
    elm_default = elm_model().fit(*data)
    elm_explicit = elm_model(ufunc="tanh").fit(*data)
    assert elm_default.hidden_layers_[0].ufunc_ is np.tanh
    assert elm_explicit.hidden_layers_[0].ufunc_ is np.tanh


@pytest.mark.parametrize("data,elm_model", params)
def test_Ufunc_WrongName_ReturnsValueError(data, elm_model):
    elm = elm_model(ufunc="UnIcOrN")
    with pytest.raises(ValueError):
        elm.fit(*data)


@pytest.mark.parametrize("data,elm_model", params)
def test_Ufunc_CustomLambdaFunction_Works(data, elm_model):
    relu = lambda x: np.maximum(x, 0)
    elm = elm_model(ufunc=relu).fit(*data)
    assert elm.hidden_layers_[0].ufunc_ is relu


@pytest.mark.parametrize("data,elm_model", params)
def test_Ufunc_NumpyUfunc_Works(data, elm_model):
    elm = elm_model(ufunc=np.sin).fit(*data)
    assert elm.hidden_layers_[0].ufunc_ is np.sin


@pytest.mark.parametrize("data,elm_model", params)
def test_PairwiseKernel_Works(data, elm_model):
    elm = elm_model(pairwise_metric="euclidean").fit(*data)
    assert hasattr(elm.hidden_layers_[0].projection_, "pairwise_metric")


@pytest.mark.parametrize("data,elm_model", params)
def test_PairwiseKernel_TooManyNeurons_StillWorks(data, elm_model):
    X, y = data
    elm = elm_model(n_neurons=3 * X.shape[0], pairwise_metric="euclidean")
    elm.fit(X, y)


@pytest.mark.parametrize("data,elm_model", params)
def test_PairwiseDistances_AllKinds_FromScikitLearn(data, elm_model):
    elm_model(n_neurons=3, pairwise_metric="cityblock").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="cosine").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="euclidean").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="l1").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="l2").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="manhattan").fit(*data)


@pytest.mark.parametrize("data,elm_model", params)
def test_PairwiseDistances_AllKinds_FromScipy(data, elm_model):
    elm_model(n_neurons=3, pairwise_metric="braycurtis").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="canberra").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="chebyshev").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="correlation").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="dice").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="hamming").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="jaccard").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="kulsinski").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="mahalanobis").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="minkowski").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="rogerstanimoto").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="russellrao").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="seuclidean").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="sokalmichener").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="sokalsneath").fit(*data)
    elm_model(n_neurons=3, pairwise_metric="sqeuclidean").fit(*data)


