import unittest
from sklearn.datasets import load_diabetes, load_wine
from sklearn.preprocessing import RobustScaler

from .test_projection import TestDifferentProjectionsRegression
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix

from skelm import ELMClassifier, ELMRegressor


class TestSparseRegression_CSR_Projections(TestDifferentProjectionsRegression, unittest.TestCase):
    def setUp(self) -> None:
        self.elm_model = ELMRegressor
        X, Y = load_diabetes(return_X_y=True)
        X = RobustScaler().fit_transform(X)
        X = csr_matrix(X)
        self.data = (X, Y)

    def test_PairwiseDistances_AllKinds_FromScipy(self):
        pass


class TestSparseRegression_CSC_Projections(TestDifferentProjectionsRegression, unittest.TestCase):
    def setUp(self) -> None:
        self.elm_model = ELMRegressor
        X, Y = load_diabetes(return_X_y=True)
        X = RobustScaler().fit_transform(X)
        X = csc_matrix(X)
        self.data = (X, Y)

    def test_PairwiseDistances_AllKinds_FromScipy(self):
        pass


class TestSparseRegression_COO_Projections(TestDifferentProjectionsRegression, unittest.TestCase):
    def setUp(self) -> None:
        self.elm_model = ELMRegressor
        X, Y = load_diabetes(return_X_y=True)
        X = RobustScaler().fit_transform(X)
        X = coo_matrix(X)
        self.data = (X, Y)

    def test_PairwiseDistances_AllKinds_FromScipy(self):
        pass


class TestSparseClassification_CSR_Projections(TestDifferentProjectionsRegression, unittest.TestCase):
    def setUp(self) -> None:
        self.elm_model = ELMClassifier
        X, Y = load_wine(return_X_y=True)
        X = RobustScaler().fit_transform(X)
        X = csr_matrix(X)
        self.data = (X, Y)

    def test_PairwiseDistances_AllKinds_FromScipy(self):
        pass


class TestSparseClassification_CSC_Projections(TestDifferentProjectionsRegression, unittest.TestCase):
    def setUp(self) -> None:
        self.elm_model = ELMClassifier
        X, Y = load_wine(return_X_y=True)
        X = RobustScaler().fit_transform(X)
        X = csc_matrix(X)
        self.data = (X, Y)

    def test_PairwiseDistances_AllKinds_FromScipy(self):
        pass


class TestSparseClassification_COO_Projections(TestDifferentProjectionsRegression, unittest.TestCase):
    def setUp(self) -> None:
        self.elm_model = ELMClassifier
        X, Y = load_wine(return_X_y=True)
        X = RobustScaler().fit_transform(X)
        X = coo_matrix(X)
        self.data = (X, Y)

    def test_PairwiseDistances_AllKinds_FromScipy(self):
        pass
