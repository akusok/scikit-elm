import unittest
import numpy as np
from numpy.testing import assert_allclose

from sklearn.utils.estimator_checks import check_estimator

from skelm.solver_batch import BatchCholeskySolver


class TestCholeskySolver(unittest.TestCase):

    def test_CholeskySolverSklearn_IsScikitLearnEstimator(self):
        solver = BatchCholeskySolver()
        check_estimator(solver)

    def test_SingleStepSolution(self):
        X = np.random.randn(100, 3)
        Y = X @ np.array([1, 2, 3]) - 2
        solver = BatchCholeskySolver().fit(X[::2], Y[::2])
        Yh = solver.predict(X[1::2])
        assert_allclose(Y[1::2], Yh, rtol=1e-3)
        assert_allclose(solver.coef_, np.array([1, 2, 3]), rtol=1e-3)
        assert_allclose(solver.intercept_, -2)

    def test_PartialFitSolution(self):
        X = np.random.randn(100, 3)
        Y = X @ np.array([1, 2, 3]) - 2
        solver = BatchCholeskySolver().partial_fit(X[::2], Y[::2])
        Yh = solver.predict(X[1::2])
        assert_allclose(Y[1::2], Yh, rtol=1e-3)
        assert_allclose(solver.coef_, np.array([1, 2, 3]), rtol=1e-3)
        assert_allclose(solver.intercept_, -2)
    
    def test_OutputShape_1d(self):
        X = np.random.randn(100, 3)
        Y = X @ np.array([1, 2, 3]) - 2
        self.assertEqual(len(Y.shape), 1)
        solver = BatchCholeskySolver().partial_fit(X[::2], Y[::2])
        Yh = solver.predict(X[1::2])
        self.assertEqual(len(Yh.shape), 1)
    
    
    def test_OutputShape_2d(self):
        X = np.random.randn(100, 3)
        W = np.array([[1, 4],
                      [2, 5],
                      [3, 6]])
        Y = X @ W - 2
        self.assertEqual(len(Y.shape), 2)
        solver = BatchCholeskySolver().partial_fit(X[::2], Y[::2])
        Yh = solver.predict(X[1::2])
        self.assertEqual(len(Yh.shape), 2)
    
    
    def test_PartialFit_SeveralParts(self):
        X = np.random.randn(100, 3)
        W = np.array([[1, 4],
                      [2, 5],
                      [3, 6]])
        Y = X @ W - 2
    
        solver = BatchCholeskySolver()
        
        # give 1st and 2nd output column separately, multiply by 2
        # to get same results as if both columns were present both times
        solver.partial_fit(X[::2], 2 * Y[::2] @ np.array([[1, 0], [0, 0]]))
        solver.partial_fit(X[::2], 2 * Y[::2] @ np.array([[0, 0], [0, 1]]))
    
        Yh = solver.predict(X[1::2])
        assert_allclose(Y[1::2], Yh, rtol=1e-3)
        assert_allclose(solver.coef_, W, rtol=1e-3)
        assert_allclose(solver.intercept_, np.array([-2, -2]))
