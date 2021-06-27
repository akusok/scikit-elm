import unittest
from numpy.testing import assert_allclose
from sklearn.datasets import load_iris, load_boston
from skelm import ELMClassifier, ELMRegressor


class TestFeatures(unittest.TestCase):

    def setUp(self) -> None:
        self.params = [(load_iris(return_X_y=True), ELMClassifier), (load_boston(return_X_y=True), ELMRegressor)]

    def test_NeuronsAmount_CanSetManually(self):
        for data, elm in self.params:
            print("foo")
            elm_3 = elm(n_neurons=3, random_state=0)
            elm_3.fit(*data)
            score_3 = elm_3.score(*data)

            elm_10 = elm(n_neurons=10, random_state=0)
            elm_10.fit(*data)
            score_10 = elm_10.score(*data)

            self.assertGreater(score_10, score_3)

    def test_Forgetting_SameResults(self):
        for data, elm in self.params:
            X, y = data
            X1, X2, X3 = [X[i::3] for i in range(3)]
            y1, y2, y3 = [y[i::3] for i in range(3)]

            elm1 = elm(n_neurons=10, random_state=0)
            elm1.fit(X1, y1)
            score1 = elm1.score(X3, y3)

            elm2 = elm(n_neurons=10, random_state=0)
            elm2.fit(X, y)
            elm2.partial_fit(X2, y2, forget=True)
            elm2.partial_fit(X3, y3, forget=True)
            score2 = elm2.score(X3, y3)

            assert_allclose(score1, score2)
