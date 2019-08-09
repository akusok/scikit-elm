import unittest
import warnings
import numpy as np
from sklearn.datasets import load_iris, load_boston
import pandas as pd

from skelm import ELMClassifier, ELMRegressor


class TestAllDatasets(unittest.TestCase):

    def setUp(self) -> None:
        irX, irY = load_iris(return_X_y=True)
        df_irX = pd.DataFrame(irX)
        df_irY = pd.DataFrame(irY)
        self.iris = df_irX, df_irY

        boX, boY = load_boston(return_X_y=True)
        df_boX = pd.DataFrame(boX)
        df_boY = pd.DataFrame(boY)
        self.boston = df_boX, df_boY

        warnings.filterwarnings("ignore", message="The number of components is higher than the number of features")
        warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.")


    def test_Pandas_ActuallyUsesDataFrames(self):
        irX, irY = self.iris
        self.assertIsInstance(irX, pd.DataFrame)
        self.assertIsInstance(irY, pd.DataFrame)

        boX, boY = self.boston
        self.assertIsInstance(boX, pd.DataFrame)
        self.assertIsInstance(boY, pd.DataFrame)

    def test_Classifier_FitOnPandas_ReturnsNumpy(self):
        X, Y = self.iris
        elm = ELMClassifier().fit(X, Y)
        Yh = elm.predict(X)
        self.assertIsInstance(Yh, np.ndarray)

    def test_Classification_Iris_BetterThanNaive(self):
        X, Y = self.iris
        elm = ELMClassifier().fit(X, Y)
        score = elm.score(X, Y)
        self.assertGreater(score, 0.33)

    def test_Regressor_FitOnPandas_ReturnsNumpy(self):
        X, Y = self.boston
        elm = ELMRegressor().fit(X, Y)
        Yh = elm.predict(X)
        self.assertIsInstance(Yh, np.ndarray)

    def test_Regression_Boston_BetterThanNaive(self):
        X, Y = self.boston
        elm = ELMRegressor().fit(X, Y)
        r2score = elm.score(X, Y)
        self.assertGreater(r2score, 0.3)

