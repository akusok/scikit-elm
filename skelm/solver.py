from __future__ import annotations

import numpy as np
import scipy as sp
import warnings
from typing import Protocol, Optional
from numpy.typing import ArrayLike
from sklearn.utils.extmath import safe_sparse_dot

from scipy.linalg import LinAlgWarning
warnings.simplefilter("ignore", LinAlgWarning)


class Solver(Protocol):
    """Stateful solver that updates and keeps its liner model coefficient and intercept."""

    coef_: Optional[ArrayLike]
    intercept_: Optional[ArrayLike]

    def fit(self, X: ArrayLike, y: ArrayLike) -> Solver:
        """Compute coefficient and intercept of a solver."""


class BatchSolver(Solver, Protocol):
    """Protocol for batch solvers."""

    def partial_fit(self, X: ArrayLike, y: ArrayLike, compute_output_weights=True, forget=False) -> BatchSolver:
        """Add or subtract new data, optionally update solution."""

    def compute_output_weights(self) -> None:
        """Update solution from currently stored data."""


class BasicSolver:
    coef_ = None
    intercept_ = np.array([0])

    def fit(self, X, y):
        self.coef_ = np.linalg.pinv(X) @ y


class RidgeSolver:
    coef_ = None
    intercept_ = np.array([0])

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        XX = X.T @ X + self.alpha * np.eye(X.shape[1])
        Xy = X.T @ y
        self.coef_ = np.linalg.lstsq(XX, Xy, rcond=-1)[0]


class BatchRidgeSolver:
    """Stateful batch solver, can add or remove training data."""

    coef_ = None
    intercept_ = np.array([0], ndmin=2)
    XX = None
    Xy = None

    def __init__(self, alpha=1e-3):
        self.alpha = alpha

    def fit(self, X, y):
        self.XX = X.T @ X + self.alpha * np.eye(X.shape[1])
        self.Xy = X.T @ y
        self.compute_output_weights()

    def compute_output_weights(self):
        self.coef_ = np.linalg.lstsq(self.XX, self.Xy, rcond=-1)[0]

    def partial_fit(self, X, y, compute_output_weights=True, forget=False):

        self.coef = None  # invalidate old solution
        XX_delta = X.T @ X
        Xy_delta = X.T @ y

        if self.XX is None:
            d = X.shape[1]
            d_out = y.shape[1]
            self.XX = self.alpha * np.eye(d)
            self.Xy = np.zeros((d, d_out))

        if forget:
            self.XX -= XX_delta
            self.Xy -= Xy_delta
        else:
            self.XX += XX_delta
            self.Xy += Xy_delta

        if compute_output_weights:
            self.compute_output_weights()


class CholeskySolver:

    def __init__(self, alpha=1e-7):
        self.alpha: float = alpha
        self.XtX = None
        self.XtY = None
        self.coef_ = None
        self.intercept_ = None

    def _reset_model(self):
        """Clean temporary data storage.
        """
        self.XtX = None
        self.XtY = None
        self.coef_ = None
        self.intercept_ = None

    def _init_model(self, X, y):
        """Initialize model storage.
        """
        d_in = X.shape[1]
        self.XtX = np.eye(d_in + 1) * self.alpha
        self.XtX[0, 0] = 0
        if len(y.shape) == 1:
            self.XtY = np.zeros((d_in + 1,))
        else:
            self.XtY = np.zeros((d_in + 1, y.shape[1]))

    def _validate_model(self, X, y):
        if self.XtX is None:
            raise RuntimeError("Model is not initialized")

        if X.shape[1] + 1 != self.XtX.shape[0]:
            n_new, n_old = X.shape[1], self.XtX.shape[0] - 1
            raise ValueError("Number of features %d does not match previous data %d." % (n_new, n_old))

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different number of samples.")

    def _update_model(self, X, y, forget=False):
        """Update model with new data; or remove already trained data from model.
        """
        X_sum = safe_sparse_dot(X.T, np.ones((X.shape[0],)))
        y_sum = safe_sparse_dot(y.T, np.ones((y.shape[0],)))

        if not forget:
            self.XtX[0, 0] += X.shape[0]
            self.XtX[1:, 0] += X_sum
            self.XtX[0, 1:] += X_sum
            self.XtX[1:, 1:] += X.T @ X

            self.XtY[0] += y_sum
            self.XtY[1:] += X.T @ y
        else:
            self.XtX[0, 0] -= X.shape[0]
            self.XtX[1:, 0] -= X_sum
            self.XtX[0, 1:] -= X_sum
            self.XtX[1:, 1:] -= X.T @ X

            self.XtY[0] -= y_sum
            self.XtY[1:] -= X.T @ y

        # invalidate previous solution
        self.coef_ = None
        self.intercept_ = None

    def compute_output_weights(self):
        """Compute solution from model with some data in it.

        Second stage of solution (X'X)B = X'Y that uses a fast Cholesky decomposition approach.
        """
        if self.XtX is None:
            raise RuntimeError("Attempting to solve uninitialized model")

        B = sp.linalg.solve(self.XtX, self.XtY, assume_a='sym', overwrite_a=False, overwrite_b=False)
        self.coef_ = B[1:]
        self.intercept_ = B[0]

    def fit(self, X, y):
        self._reset_model()
        self._init_model(X, y)
        self._update_model(X, y)
        self.compute_output_weights()

    def partial_fit(self, X, y, compute_output_weights=True, forget=False):
        if forget:
            self.batch_forget(X, y, compute_output_weights)
        else:
            self.batch_update(X, y, compute_output_weights)

    def batch_update(self, X, y, compute_output_weights=True):
        if self.XtX is None:
            self._init_model(X, y)
        else:
            self._validate_model(X, y)

        self._update_model(X, y)
        if compute_output_weights:
            self.compute_output_weights()

    def batch_forget(self, X, y, compute_output_weights=True):
        if self.XtX is None:
            raise RuntimeError("Attempting to subtract data from uninitialized model")
        else:
            self._validate_model(X, y)

        self._update_model(X, y, forget=True)
        if compute_output_weights:
            self.compute_output_weights()
