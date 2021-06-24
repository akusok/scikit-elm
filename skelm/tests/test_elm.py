import unittest
import warnings
from sklearn.exceptions import DataDimensionalityWarning

from sklearn.datasets import load_boston
from sklearn.utils.estimator_checks import check_estimator

from skelm.hidden_layer import RandomProjectionSLFN
from skelm.solver import BasicSolver, BatchRidgeSolver
from skelm.elm import BasicELM, BatchELM, ScikitELM


class TestScikitELM(unittest.TestCase):

    def setUp(self) -> None:
        self.X, y = load_boston(return_X_y=True)
        self.y = y[:, None]
        warnings.simplefilter("ignore", DataDimensionalityWarning)

    def test_ScikitELM_IsScikitLearnEstimator(self):
        model = ScikitELM(n_neurons=10, alpha=1e-3)
        check_estimator(model)

        model.fit(self.X, self.y)
        check_estimator(model)

    def test_IncludeOriginalFeature_AddsNewNeuronType(self):
        model = ScikitELM(n_neurons=3, include_original_features=True)
        model.fit(self.X, self.y)
        self.assertEqual(len(model.model_.SLFNs), 2)

    def test_IncludeOriginalFeature_MoreNeurons(self):
        model_basic = ScikitELM(n_neurons=3, include_original_features=False)
        model_orig = ScikitELM(n_neurons=3, include_original_features=True)
        model_basic.fit(self.X, self.y)
        model_orig.fit(self.X, self.y)
        self.assertGreater(model_orig.n_neurons_, model_basic.n_neurons_)

class TestBasicELM(unittest.TestCase):

    def setUp(self) -> None:
        self.X, y = load_boston(return_X_y=True)
        self.y = y[:, None]
        warnings.simplefilter("ignore", DataDimensionalityWarning)

    def test_ELM_MultipleHiddenLayers_Works(self):
        h1 = RandomProjectionSLFN(self.X, 1)
        h2 = RandomProjectionSLFN(self.X, 2)
        h3 = RandomProjectionSLFN(self.X, 5)
        simple_solver = BasicSolver()
        elm = BasicELM((h1, h2, h3), simple_solver)
        elm.fit(self.X, self.y)

    def test_BatchUpdate_NonBatchSolver_Raises(self):
        simple_hidden = RandomProjectionSLFN(self.X, 10)
        simple_solver = BasicSolver()
        elm = BatchELM([simple_hidden], simple_solver)
        with self.assertRaises(AttributeError):
            elm.partial_fit(self.X, self.y)

    def test_BatchUpdate_BatchSolver_Works(self):
        simple_hidden = RandomProjectionSLFN(self.X, 10)
        batch_solver = BatchRidgeSolver()
        elm = BatchELM([simple_hidden, ], batch_solver)
        elm.partial_fit(self.X, self.y)

