import unittest
import warnings
from sklearn.datasets import load_iris, load_boston

from skelm import ELMClassifier, ELMRegressor


class TestAllDatasets(unittest.TestCase):

    def setUp(self) -> None:
        self.iris = load_iris(return_X_y=True)
        self.boston = load_boston(return_X_y=True)

        warnings.filterwarnings("ignore", message="The number of components is higher than the number of features")

    def test_Classification_Iris_BetterThanNaive(self):
        X, Y = self.iris
        elm = ELMClassifier()
        elm.fit(X, Y)
        score = elm.score(X, Y)
        self.assertGreater(score, 0.33)

    def test_Regression_Boston_BetterThanNaive(self):
        X, Y = self.boston
        elm = ELMRegressor()
        elm.fit(X, Y)
        r2score = elm.score(X, Y)
        self.assertGreater(r2score, 0.3)

