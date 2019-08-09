import unittest
import numpy as np


from skelm.solvers import IterativeSolver


class TestIterativeSolver(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(0)
    
    def test_SingleStepSolution(self):
        X = np.random.randn(100, 3)
        Y = X @ np.array([1,2,3]) - 2
        solver = IterativeSolver().fit(X[::2], Y[::2])
        Yh = solver.predict(X[1::2])
        
        np.testing.assert_almost_equal(Y[1::2], Yh)
        np.testing.assert_almost_equal(solver.coef_, np.array([1,2,3]))
        np.testing.assert_almost_equal(solver.intercept_, -2)
        
    def test_PartialFitSolution(self):
        X = np.random.randn(100, 3)
        Y = X @ np.array([1,2,3]) - 2
        solver = IterativeSolver().partial_fit(X[::2], Y[::2])
        Yh = solver.predict(X[1::2])
        
        np.testing.assert_almost_equal(Y[1::2], Yh)
        np.testing.assert_almost_equal(solver.coef_, np.array([1,2,3]))
        np.testing.assert_almost_equal(solver.intercept_, -2)
        
    def test_OutputShape_1d(self):
        X = np.random.randn(100, 3)
        Y = X @ np.array([1,2,3]) - 2
        assert len(Y.shape) == 1
        solver = IterativeSolver().partial_fit(X[::2], Y[::2])
        Yh = solver.predict(X[1::2])
        self.assertEqual(len(Yh.shape), 1)

    def test_OutputShape_2d(self):
        X = np.random.randn(100, 3)
        W = np.array([[1, 4],
                      [2, 5],
                      [3, 6]])
        Y = X @ W - 2
        assert len(Y.shape) == 2
        solver = IterativeSolver().partial_fit(X[::2], Y[::2])
        Yh = solver.predict(X[1::2])
        self.assertEqual(len(Yh.shape), 2)

    def test_PartialFit_SeveralParts(self):
        X = np.random.randn(100, 3)
        W = np.array([[1, 4],
                      [2, 5],
                      [3, 6]])
        Y = X @ W - 2

        solver = IterativeSolver()
        
        # give 1st and 2nd output column separately, multiply by 2 to get same results as if both columns were present both times
        solver.partial_fit(X[::2], 2 * Y[::2] @ np.array([[1,0], [0,0]]))
        solver.partial_fit(X[::2], 2 * Y[::2] @ np.array([[0,0], [0,1]]))

        Yh = solver.predict(X[1::2])
        
        np.testing.assert_almost_equal(Y[1::2], Yh)
        np.testing.assert_almost_equal(solver.coef_, W)
        np.testing.assert_almost_equal(solver.intercept_, np.array([-2, -2]))

        
