import unittest
from numpy.testing import assert_allclose
import numpy as np
from skelm import ELMClassifier, ELMRegressor
from sklearn.datasets import load_iris, load_boston


class TestUnittests(unittest.TestCase):

    def setUp(self):
        self.data_class = load_iris(return_X_y=True)
        self.data_reg = load_boston(return_X_y=True)

    def test_Classifier_predict_ReturnsIntegerArray(self):
        X = np.array([[1], [2], [3]])
        Y = np.array([0, 0, 1])
        elm = ELMClassifier()
        elm.fit(X, Y)
        Yh = elm.predict(X)
        self.assertIsInstance(Yh, np.ndarray)
        assert_allclose(Yh, Yh.astype(np.int))

    def test_Classifier_WrongNumberOfFeatures_RaisesError(self):
        X, T = self.data_class
        elm = ELMClassifier()
        elm.fit(X, T)
        with self.assertRaises(ValueError):
            elm.predict(X[:, 1:])

    def test_Regressor_WrongNumberOfFeatures_RaisesError(self):
        X, T = self.data_reg
        elm = ELMRegressor()
        elm.fit(X, T)
        with self.assertRaises(ValueError):
            elm.predict(X[:, 1:])

    def test_Classifier_Multilabel(self):
        X, T = self.data_class
        Y = np.ones((T.shape[0], 2))
        Y[:, 0] = T
        elm = ELMClassifier()
        elm.fit(X, Y)

    def test_Classifier_SetClasses_IgnoresOther(self):
        X, T = self.data_class
        elm = ELMClassifier(classes=[0, 1])
        Yh = elm.fit(X, T).predict(X)
        self.assertEqual(set(Yh), {0, 1})

    def test_Classifier_PartialFit(self):
        X, T = self.data_class
        elm0 = ELMClassifier(n_neurons=4, alpha=1, random_state=0)
        elm1 = ELMClassifier(n_neurons=4, alpha=1, random_state=0)

        elm0.fit(X, T)
        elm1.partial_fit(X[::2], T[::2])
        elm1.partial_fit(X[1::2], T[1::2])

        assert_allclose(elm0.solver_.coef_, elm1.solver_.coef_)

    def test_IterativeClassification_FeedClassesOneByOne(self):
        X, T = self.data_class
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
        self.assertEqual(set(Yh), {0, 1, 2})

    def test_IterativeSolver_SkipIntermediateSolution(self):
        X, T = self.data_class
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
        self.assertEqual(set(Yh), {0, 1, 2})

    def test_MultipleHiddenLayers(self):
        X, Y = self.data_reg
        elm = ELMRegressor(n_neurons=[2, 3], ufunc=['tanh', 'sigm'],
                           density=[None, None], pairwise_metric=[None, None])
        elm.fit(X, Y)
        self.assertEqual(len(elm.SLFNs_), 2)

    def test_MultipleHiddenLayers_MoreCombinations(self):
        X, Y = self.data_reg
        elm = ELMRegressor(n_neurons=[1, 1, 1, 1, 1],
                           ufunc=['relu', 'sigm', np.sin, None, None],
                           density=[None, 0.5, 0.8, None, None],
                           pairwise_metric=[None, None, None, 'l1', 'chebyshev'])
        elm.fit(X, Y)
        self.assertEqual(len(elm.SLFNs_), 5)

    def test_MultipleHL_DefaultValues(self):
        X, Y = self.data_reg
        elm = ELMRegressor(n_neurons=[2, 3])
        elm.fit(X, Y)

    def test_MultipleHL_Ufunc_SingleValue(self):
        X, Y = self.data_reg
        elm = ELMRegressor(n_neurons=[2, 3], ufunc='sigm')
        elm.fit(X, Y)

    def test_MultipleHL_Density_SingleValue(self):
        X, Y = self.data_reg
        elm = ELMRegressor(n_neurons=[2, 3], density=0.7)
        elm.fit(X, Y)

    def test_MultipleHL_Pairwise_SingleValue(self):
        X, Y = self.data_reg
        elm = ELMRegressor(n_neurons=[2, 3], pairwise_metric='l2')
        elm.fit(X, Y)

    def test_MultipleHL_WrongDimensions_Raises(self):
        X, Y = self.data_reg
        elm = ELMRegressor(n_neurons=[1, 2, 3, 4], ufunc=['relu', 'sigm'])
        with self.assertRaises(ValueError):
            elm.fit(X, Y)

    def test_RegularizationAlpha_NegativeValue_Raises(self):
        X, Y = self.data_class
        elm = ELMClassifier(alpha=-1)
        with self.assertRaises(ValueError):
            elm.fit(X, Y)
