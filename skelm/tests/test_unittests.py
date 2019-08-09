import unittest
import numpy as np
from skelm import ELMClassifier, ELMRegressor
from sklearn.datasets import load_iris, load_boston


class TestELMClassifier(unittest.TestCase):

    def test_Classifier_predict_ReturnsIntegerArray(self):
        X = np.array([[1], [2], [3]])
        Y = np.array([0, 0, 1])
        elm = ELMClassifier().fit(X, Y)
        Yh = elm.predict(X)

        self.assertIsInstance(Yh, np.ndarray)
        np.testing.assert_array_almost_equal(Yh, Yh.astype(np.int))

    def test_Classifier_WrongNumberOfFeatures_RaisesError(self):
        X, T = load_iris(return_X_y=True)
        elm = ELMClassifier().fit(X, T)
        with self.assertRaises(ValueError):
            elm.predict(X[:, 1:])

    def test_Regressor_WrongNumberOfFeatures_RaisesError(self):
        X, T = load_boston(return_X_y=True)
        elm = ELMRegressor().fit(X, T)
        with self.assertRaises(ValueError):
            elm.predict(X[:, 1:])

    def test_Classifier_Multilabel(self):
        X, T = load_iris(return_X_y=True)
        Y = np.ones((T.shape[0], 2))
        Y[:, 0] = T

        elm = ELMClassifier()
        elm.fit(X, Y)

    def test_Classifier_SetClasses_IgnoresOther(self):
        X, T = load_iris(return_X_y=True)
        elm = ELMClassifier(classes=[0, 1])
        Yh = elm.fit(X, T).predict(X)
        self.assertEqual({0, 1}, set(Yh))

    def test_ClassifierRidge_PartialFit_NotSupported(self):
        X, T = load_iris(return_X_y=True)
        elm = ELMClassifier(solver='ridge', random_state=0)
        self.assertRaises(ValueError, elm.partial_fit, X, T)

    def test_Classifier_PartialFit(self):
        X, T = load_iris(return_X_y=True)
        elm0 = ELMClassifier(solver='iterative', n_neurons=10, alpha=1, random_state=0)
        elm1 = ELMClassifier(solver='iterative', n_neurons=10, alpha=1, random_state=0)

        elm0.fit(X, T)
        elm1.partial_fit(X[::2], T[::2])
        elm1.partial_fit(X[1::2], T[1::2])

        np.testing.assert_array_almost_equal(elm0.solver_.coef_, elm1.solver_.coef_)

    def test_IterativeClassification_FeedClassesOneByOne(self):
        X, T = load_iris(return_X_y=True)
        elm = ELMClassifier(solver='iterative', classes=[0, -1, -2], n_neurons=10, alpha=1)

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
        self.assertEqual({0, 1, 2}, set(Yh))

    def test_IterativeSolver_SkipIntermediateSolution(self):
        X, T = load_iris(return_X_y=True)
        elm = ELMClassifier(solver='iterative', classes=[0, 1, 2], n_neurons=10, alpha=1)

        X0 = X[T == 0]
        Y0 = T[T == 0]
        elm.partial_fit(X0, Y0, skip_solution=True)

        X1 = X[T == 1]
        Y1 = T[T == 1]
        elm.partial_fit(X1, Y1, skip_solution=True)

        X2 = X[T == 2]
        Y2 = T[T == 2]
        elm.partial_fit(X2, Y2)

        Yh = elm.predict(X)
        self.assertEqual({0, 1, 2}, set(Yh))
