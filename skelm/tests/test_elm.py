import unittest
import warnings
import numpy as np
from sklearn.exceptions import DataDimensionalityWarning

from sklearn.datasets import load_boston
from sklearn.utils.estimator_checks import check_estimator

from skelm.hidden_layer import RandomProjectionSLFN
from skelm.solver import BasicSolver, BatchRidgeSolver
from skelm.elm import BasicELM, BatchELM, ScikitELM
from skelm.elm_lanczos import LanczosELM


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
        elm = ScikitELM(n_neurons=3, include_original_features=True)
        elm.fit(self.X, self.y)
        self.assertEqual(len(elm.SLFNs_), 2)

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


class TestLanczosELM(unittest.TestCase):

    def setUp(self) -> None:
        self.X, y = load_boston(return_X_y=True)
        self.y = y[:, None]
        warnings.simplefilter("ignore", DataDimensionalityWarning)

    def test_LancsozELM_IsScikitLearnEstimator(self):
        model = LanczosELM(n_neurons=10, include_original_features=True)
        check_estimator(model)

        model.fit(self.X, self.y)
        check_estimator(model)

    def test_ELM_MultipleHiddenLayers_Works(self):
        elm = LanczosELM(
            include_original_features=True,
            n_neurons=(1,2,3),
            ufunc='tanh')

        elm.fit(self.X, self.y, self.X, self.y)

    def test_ValidationEarlyStopping_ImprovesSolution(self):
        def never_stop(_):
            return False

        Xt, Xv, Xs = self.X[0::3], self.X[1::3], self.X[2::3]
        yt, yv, ys = self.y[0::3], self.y[1::3], self.y[2::3]

        elm = LanczosELM(n_neurons=99, ufunc='tanh')
        elm.fit(Xt, yt, Xv, yv)
        rmse_validation = np.mean((ys - elm.predict(Xs))**2)**0.5

        elm.fit(Xt, yt, Xv, yv, stopping_condition=never_stop)
        rmse_full_solution = np.mean((ys - elm.predict(Xs))**2)**0.5
        self.assertLess(rmse_validation, rmse_full_solution)
