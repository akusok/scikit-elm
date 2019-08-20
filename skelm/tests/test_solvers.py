import pytest
from pytest import approx
import numpy as np

from skelm.solver_batch import BatchCholeskySolver


def test_SingleStepSolution():
    X = np.random.randn(100, 3)
    Y = X @ np.array([1, 2, 3]) - 2
    solver = BatchCholeskySolver().fit(X[::2], Y[::2])
    Yh = solver.predict(X[1::2])
    assert Y[1::2] == approx(Yh)
    assert solver.coef_ == approx(np.array([1, 2, 3]), rel=1e-3)
    assert solver.intercept_ == approx(-2)


def test_PartialFitSolution():
    X = np.random.randn(100, 3)
    Y = X @ np.array([1, 2, 3]) - 2
    solver = BatchCholeskySolver().partial_fit(X[::2], Y[::2])
    Yh = solver.predict(X[1::2])
    assert Y[1::2] == approx(Yh, rel=1e-3)
    assert solver.coef_ == approx(np.array([1, 2, 3]), rel=1e-3)
    assert solver.intercept_ == approx(-2)


def test_OutputShape_1d():
    X = np.random.randn(100, 3)
    Y = X @ np.array([1, 2, 3]) - 2
    assert len(Y.shape) == 1
    solver = BatchCholeskySolver().partial_fit(X[::2], Y[::2])
    Yh = solver.predict(X[1::2])
    assert len(Yh.shape) == 1


def test_OutputShape_2d():
    X = np.random.randn(100, 3)
    W = np.array([[1, 4],
                  [2, 5],
                  [3, 6]])
    Y = X @ W - 2
    assert len(Y.shape) == 2
    solver = BatchCholeskySolver().partial_fit(X[::2], Y[::2])
    Yh = solver.predict(X[1::2])
    assert len(Yh.shape) == 2


def test_PartialFit_SeveralParts():
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
    assert Y[1::2] == approx(Yh, rel=1e-3)
    assert solver.coef_ == approx(W, rel=1e-3)
    assert solver.intercept_ == approx(np.array([-2, -2]))

