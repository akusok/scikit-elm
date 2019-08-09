import unittest
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="DataDimensionalityWarning")

from sklearn.datasets import load_iris, make_multilabel_classification, load_diabetes

from skelm import ELMRegressor, ELMClassifier


class TestAcceptance(unittest.TestCase):
    
    def setUp(self):
        # suppress annoying warning for random projections into a higher-dimensional space
        warnings.filterwarnings("ignore", message="The number of components is higher than the number of features")
        warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.")

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
        self.assertLess(Y[2], 0)
        self.assertLess(Y[3], 0)
        
    def test_ELMClassifier_ReportedScore_ActuallyIsClassificationScore(self):
        X, Y = load_iris(return_X_y=True)
        Yr = np.vstack((Y == 0, Y == 1, Y == 2)).T

        elm_c = ELMClassifier(random_state=0).fit(X, Y)
        elm_r = ELMRegressor(random_state=0).fit(X, Yr)

        Yc_hat = elm_c.predict(X)
        Yr_hat = elm_r.predict(X).argmax(1)

        np.testing.assert_array_almost_equal(Yc_hat, Yr_hat)        
        
    def test_ELMClassifier_MultilabelClassification_Works(self):
        X, Y = make_multilabel_classification()

        elm_c = ELMClassifier(random_state=0).fit(X, Y)
        elm_r = ELMRegressor(random_state=0).fit(X, Y)

        Yc_hat = elm_c.predict(X)
        Yr_hat = (elm_r.predict(X) >= 0.5).astype(np.int)

        np.testing.assert_array_almost_equal(Yc_hat, Yr_hat)
        
    def test_RegularizationL2_DifferentValue_ChangesPrediction(self):
        X, Y = load_diabetes(return_X_y=True)
        Yh_1 = ELMRegressor(alpha=1e-7, random_state=0).fit(X, Y).predict(X)
        Yh_2 = ELMRegressor(alpha=1e+3, random_state=0).fit(X, Y).predict(X)
        self.assertFalse(np.allclose(Yh_1, Yh_2))
                

if __name__ == "__main__":
    unittest.main()