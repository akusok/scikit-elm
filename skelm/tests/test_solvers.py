import unittest
import numpy as np
from numpy.testing import assert_allclose

from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_diabetes

from skelm.solver_batch import BatchCholeskySolver
from skelm.solver_lanczos import LanczosSolver


class TestLanczosSolver(unittest.TestCase):

    def test_SolveNoBias_Works(self):
        X = np.random.randn(100, 3)
        Y = X @ np.array([1, 2, 3])
        solver = LanczosSolver().fit(X[::2], Y[::2])
        Yh = X[1::2] @ solver.coef_
        assert_allclose(Y[1::2], Yh, rtol=1e-3)
        assert_allclose(solver.coef_, np.array([1, 2, 3]), rtol=1e-3)

    def test_SolveMultiOutputs_Works(self):
        X = np.random.randn(100, 3)
        Y = X @ np.random.randn(3, 5)
        with self.assertRaises(ValueError):
            LanczosSolver().fit(X, Y)

    def test_SolveArtificialIntercept_Works(self):
        X_no_intercept = np.random.randn(100, 3)
        Y = X_no_intercept @ np.array([1, 2, 3]) - 2
        X = np.hstack((np.ones((100, 1)), X_no_intercept))
        solver = LanczosSolver().fit(X[::2], Y[::2])
        Yh = X[1::2] @ solver.coef_
        assert_allclose(Y[1::2], Yh, rtol=1e-3)
        assert_allclose(solver.coef_[1:], np.array([1, 2, 3]), rtol=1e-3)
        assert_allclose(solver.coef_[:1], -2)

    def test_IncrementalSoluiton_MoreIterationsDecreaseError(self):

        def stop_two_iter(e):
            return len(e) >= 2

        def stop_five_iter(e):
            return len(e) >= 5

        def never_stop(_):
            return False

        X = np.random.randn(100, 10)
        true_coef = np.random.randn(10)
        Y = X @ true_coef
        solver = LanczosSolver()

        solver.fit(X[::2], Y[::2], X[1::2], Y[1::2], stopping_condition=stop_two_iter)
        rmse_two_iter = np.mean((Y[1::2] - X[1::2] @ solver.coef_)**2)**0.5

        solver.fit(X[::2], Y[::2], X[1::2], Y[1::2], stopping_condition=stop_five_iter)
        rmse_five_iter = np.mean((Y[1::2] - X[1::2] @ solver.coef_)**2)**0.5

        solver.fit(X[::2], Y[::2], X[1::2], Y[1::2], stopping_condition=never_stop)
        rmse_full_solution = np.mean((Y[1::2] - X[1::2] @ solver.coef_)**2)**0.5

        print(rmse_two_iter)
        print(rmse_five_iter)
        print(rmse_full_solution)

        self.assertLess(rmse_five_iter, rmse_two_iter)
        self.assertLess(rmse_full_solution, rmse_five_iter)
        self.assertLess(rmse_full_solution, 1e-5)

    def test_ValidationEarlyStopping_ImprovesSolution(self):
        X, y = load_diabetes(return_X_y=True)

        H = np.tanh(X @ np.random.randn(X.shape[1], 99))  # make lots of extra features
        H = np.hstack((np.ones((H.shape[0], 1)), H))  # add bias column - Lanczos solver has no bias

        Ht, Hv, Hs = H[0::3], H[1::3], H[2::3]
        Yt, Yv, Ys = y[0::3], y[1::3], y[2::3]

        def never_stop(_):
            return False

        solver = LanczosSolver()
        solver.fit(Ht, Yt, Hv, Yv)
        rmse_validation = np.mean((Ys - Hs @ solver.coef_)**2)**0.5

        solver.fit(Ht, Yt, Hv, Yv, stopping_condition=never_stop)
        rmse_full_solution = np.mean((Ys - Hs @ solver.coef_)**2)**0.5

        self.assertLess(rmse_validation, rmse_full_solution)


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
