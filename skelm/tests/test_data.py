import unittest
from sklearn.datasets import load_iris, load_boston
from skelm import ELMClassifier, ELMRegressor


class TestData(unittest.TestCase):

    def setUp(self) -> None:
        self.data_class =  load_iris(return_X_y=True)
        self.data_reg = load_boston(return_X_y=True)

    def test_Classification_Iris_BetterThanNaive(self):
        elm = ELMClassifier(random_state=0)
        elm.fit(*self.data_class)
        score = elm.score(*self.data_class)
        assert score > 0.33

    def test_Regression_Boston_BetterThanNaive(self):
        elm = ELMRegressor(random_state=0)
        elm.fit(*self.data_reg)
        r2score = elm.score(*self.data_reg)
        assert r2score > 0.3
