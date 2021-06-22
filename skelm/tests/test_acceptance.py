import unittest
from numpy.testing import assert_allclose
import numpy as np
from sklearn.datasets import load_iris, make_multilabel_classification, load_diabetes
from skelm import ELMRegressor, ELMClassifier

import warnings
from sklearn.exceptions import DataDimensionalityWarning, DataConversionWarning


class TestAcceptance(unittest.TestCase):

    def setUp(self) -> None:
        self.data_class = load_iris(return_X_y=True)
        self.data_ml = make_multilabel_classification()
        self.data_reg = load_diabetes(return_X_y=True)
        warnings.simplefilter("ignore", DataDimensionalityWarning)
        warnings.simplefilter("ignore", DataConversionWarning)

    def test_SineWave_Solves(self):
        """A highly non-linear regression problem, with added strong noise.
        """
        X = np.linspace(-1, 1, num=1000)[:, None]
        Y = np.sin(16 * X) * X + 0.2*np.random.randn(1000)[:, None]

        elm = ELMRegressor(random_state=0)
        elm.fit(X, Y)
        Yt = elm.predict(X)

        MSE = np.mean((Y - Yt) ** 2)
        self.assertLess(MSE, 0.3)

    def test_Xor_OneNeuron_Solved(self):
        """ELM should be able to solve XOR problem.
        """
        X = np.array([[0, 0],
                      [1, 1],
                      [1, 0],
                      [0, 1]])
        Y = np.array([1, 1, -1, -1])

        elm = ELMClassifier(n_neurons=3, random_state=0)
        elm.fit(X, Y)
        Yh = elm.predict(X)
        self.assertGreater(Yh[0], 0)
        self.assertGreater(Yh[1], 0)
        self.assertLess(Yh[2], 0)
        self.assertLess(Yh[3], 0)

    def test_ELMClassifier_ReportedScore_ActuallyIsClassificationScore(self):
        X, Y = self.data_class
        Yr = np.vstack((Y == 0, Y == 1, Y == 2)).T

        elm_c = ELMClassifier(random_state=0).fit(X, Y)
        elm_r = ELMRegressor(random_state=0).fit(X, Yr)

        Yc_hat = elm_c.predict(X)
        Yr_hat = elm_r.predict(X).argmax(1)

        assert_allclose(Yc_hat, Yr_hat)

    def test_ELMClassifier_MultilabelClassification_Works(self):
        X, Y = self.data_ml
        elm_c = ELMClassifier(random_state=0).fit(X, Y)
        elm_r = ELMRegressor(random_state=0).fit(X, Y)

        Yc_hat = elm_c.predict(X)
        Yr_hat = (elm_r.predict(X) >= 0.5).astype(int)

        assert_allclose(Yc_hat, Yr_hat)

    def test_RegularizationL2_DifferentValue_ChangesPrediction(self):
        X, Y = self.data_reg
        Yh_1 = ELMRegressor(alpha=1e-7, random_state=0).fit(X, Y).predict(X)
        Yh_2 = ELMRegressor(alpha=1e+3, random_state=0).fit(X, Y).predict(X)

        self.assertFalse(np.allclose(Yh_1, Yh_2))

    def test_Default_SetNumberOfNeurons(self):
        X, y = self.data_reg
        elm5 = ELMRegressor(n_neurons=5, random_state=0).fit(X, y)
        elm50 = ELMRegressor(n_neurons=50, random_state=0).fit(X, y)
        score5 = elm5.score(X, y)
        score50 = elm50.score(X, y)
        self.assertGreater(score50, score5)
        self.assertGreater(score50, 0.33)
