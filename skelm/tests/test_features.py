import unittest
import warnings
from sklearn.datasets import load_iris, load_boston

from skelm import ELMClassifier, ELMRegressor


class TestNeuronsOfAllKinds(unittest.TestCase):

    def setUp(self) -> None:
        self.iris = load_iris(return_X_y=True)
        self.boston = load_boston(return_X_y=True)

        warnings.filterwarnings("ignore", message="The number of components is higher than the number of features")

    def test_Classification_NeuronsAmount_CanSetManually(self):
        X, T = self.iris
        elm_3 = ELMClassifier(n_neurons=3)
        score_3 = elm_3.fit(X, T).score(X, T)
        elm_10 = ELMClassifier(n_neurons=10)
        score_10 = elm_10.fit(X, T).score(X, T)
        self.assertGreater(score_10, score_3)

    def test_Regression_NeuronsAmount_CanSetManually(self):
        X, T = self.boston
        elm_3 = ELMRegressor(n_neurons=3, random_state=0)
        score_3 = elm_3.fit(X, T).score(X, T)
        elm_10 = ELMRegressor(n_neurons=10, random_state=0)
        score_10 = elm_10.fit(X, T).score(X, T)
        self.assertGreater(score_10, score_3)