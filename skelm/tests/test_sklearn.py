import unittest
import warnings
from sklearn.utils.estimator_checks import check_estimator, check_estimator_sparse_data

from skelm import ELMRegressor, ELMClassifier
from skelm.solvers import IterativeSolver


class TestSKLearnCompliance(unittest.TestCase):
    
    def setUp(self):
        # suppress annoying warning for random projections into a higher-dimensional space
        warnings.filterwarnings("ignore", message="The number of components is higher than the number of features")
        warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.")

    def test_ELM_ScikitLearnCompliance(self):
        check_estimator(ELMRegressor)

    def test_ELMClassifier_ScikitLearnCompliance(self):
        check_estimator(ELMClassifier)

    def test_IterativeSolver_ScikitLearnCompliance(self):
        check_estimator(IterativeSolver)

