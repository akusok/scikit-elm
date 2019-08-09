import unittest
import warnings
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.datasets import load_boston, load_iris
from sklearn.preprocessing import RobustScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from skelm import ELMRegressor, ELMClassifier


class TestDifferentProjectionsRegression(unittest.TestCase):

    def setUp(self) -> None:
        self.elm_model = ELMRegressor
        X, Y = load_boston(return_X_y=True)
        X = RobustScaler().fit_transform(X)
        self.data = (X, Y)
        
        # suppress annoying warning for random projections into a higher-dimensional space
        warnings.filterwarnings("ignore", message="The number of components is higher than the number of features")
        warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.")
        warnings.filterwarnings("ignore", message="UserWarning: n_quantiles")
        warnings.filterwarnings("ignore", message="Data with input dtype float64 was converted to bool")

    def test_Default_SetNumberOfNeurons(self):
        X, Y = self.data
        elm5 = self.elm_model(n_neurons=5).fit(X, Y)
        elm50 = self.elm_model(n_neurons=50).fit(X, Y)
        score5 = elm5.score(X, Y)
        score50 = elm50.score(X, Y)
        self.assertGreater(score50, score5)
        self.assertGreater(score50, 0.33)

    def test_LinearPart_CanBeIncluded(self):
        X, Y = self.data
        elm = self.elm_model(include_original_features=True).fit(X, Y)
        score = elm.score(X, Y)
        self.assertGreater(score, 0.33)

    def test_LinearPart_AddsExtraFeatures(self):
        X, Y = self.data
        H_basic = self.elm_model().fit(X, Y)._project(X)
        H_with_orig = self.elm_model(include_original_features=True).fit(X, Y)._project(X)
        self.assertGreater(H_with_orig.shape[1], H_basic.shape[1])

    def test_DefaultNeurons_UseGaussianRandomProjection(self):
        X, Y = self.data
        elm = self.elm_model().fit(X, Y)
        self.assertIsInstance(elm.projection_, GaussianRandomProjection)

    def test_SparseELM_UseSparseRandomProjection(self):
        X, Y = self.data
        elm = self.elm_model(density=0.1).fit(X, Y)
        self.assertIsInstance(elm.projection_, SparseRandomProjection)

    def test_Ufunc_Sigmoid(self):
        X, Y = self.data
        elm = self.elm_model(ufunc="sigm").fit(X, Y)
        self.assertIs(elm.ufunc_, sigmoid)

    def test_DefaultUfunc_Tanh(self):
        X, Y = self.data
        elm_default = self.elm_model().fit(X, Y)
        elm_explicit = self.elm_model(ufunc="tanh").fit(X, Y)
        self.assertIs(elm_default.ufunc_, np.tanh)
        self.assertIs(elm_explicit.ufunc_, np.tanh)

    def test_Ufunc_WrongName_ReturnsValueError(self):
        X, Y = self.data
        elm = self.elm_model(ufunc="UnIcOrN")
        with self.assertRaises(ValueError):
            elm.fit(X, Y)

    def test_Ufunc_CustomLambdaFunction_Works(self):
        X, Y = self.data
        relu = lambda x: np.maximum(x, 0)
        elm = self.elm_model(ufunc=relu).fit(X, Y)
        self.assertIs(elm.ufunc_, relu)

    def test_Ufunc_NumpyUfunc_Works(self):
        X, Y = self.data
        elm = self.elm_model(ufunc=np.sin).fit(X, Y)
        self.assertIs(elm.ufunc_, np.sin)

    def test_PairwiseKernel_Works(self):
        X, Y = self.data
        elm = self.elm_model(pairwise_metric="euclidean").fit(X, Y)
        self.assertTrue(hasattr(elm, "centroids_"))

    def test_PairwiseKernel_TooManyNeurons_StillWorks(self):
        X, Y = self.data
        elm = self.elm_model(n_neurons=3 * X.shape[0], pairwise_metric="euclidean")
        elm.fit(X, Y)

    def test_PairwiseDistances_AllKinds_FromScikitLearn(self):
        X, Y = self.data
        self.elm_model(pairwise_metric="cityblock").fit(X, Y)
        # self.elm_model(pairwise_metric="cosine").fit(X, Y)
        # self.elm_model(pairwise_metric="euclidean").fit(X, Y)
        # self.elm_model(pairwise_metric="l1").fit(X, Y)
        # self.elm_model(pairwise_metric="l2").fit(X, Y)
        # self.elm_model(pairwise_metric="manhattan").fit(X, Y)

    def test_PairwiseDistances_AllKinds_FromScipy(self):
        X, Y = self.data
        self.elm_model(pairwise_metric="braycurtis").fit(X, Y)
        self.elm_model(pairwise_metric="canberra").fit(X, Y)
        self.elm_model(pairwise_metric="chebyshev").fit(X, Y)
        self.elm_model(pairwise_metric="correlation").fit(X, Y)
        self.elm_model(pairwise_metric="dice").fit(X, Y)
        self.elm_model(pairwise_metric="hamming").fit(X, Y)
        self.elm_model(pairwise_metric="jaccard").fit(X, Y)
        self.elm_model(pairwise_metric="kulsinski").fit(X, Y)
        self.elm_model(pairwise_metric="mahalanobis").fit(X, Y)
        self.elm_model(pairwise_metric="minkowski").fit(X, Y)
        self.elm_model(pairwise_metric="rogerstanimoto").fit(X, Y)
        self.elm_model(pairwise_metric="russellrao").fit(X, Y)
        self.elm_model(pairwise_metric="seuclidean").fit(X, Y)
        self.elm_model(pairwise_metric="sokalmichener").fit(X, Y)
        self.elm_model(pairwise_metric="sokalsneath").fit(X, Y)
        self.elm_model(pairwise_metric="sqeuclidean").fit(X, Y)


class TestDifferentProjectionsClassification(TestDifferentProjectionsRegression, unittest.TestCase):

    def setUp(self) -> None:
        self.elm_model = ELMClassifier
        self.data = load_iris(return_X_y=True)
        
        # suppress annoying warning for random projections into a higher-dimensional space
        warnings.filterwarnings("ignore", message="The number of components is higher than the number of features")
        warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.")
        warnings.filterwarnings("ignore", message="n_quantiles is set to n_samples.")

        
if __name__ == "__main__":
    unittest.main()
