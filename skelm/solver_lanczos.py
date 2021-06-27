import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Callable, List

from .solver import Solver


def worst_of_five(errors: List[float]) -> bool:
    """Returns true when the last error is the worst of five last ones."""
    if len(errors) > 3 and (errors[-1] > max(errors[-5:-1])):
        return True
    return False


class LanczosSolver(Solver):

    coef_ = None
    intercept_ = np.zeros((1,))
    X_val: Optional[ArrayLike] = None
    y_val: Optional[ArrayLike] = None
    data_features: Optional[int] = None
    validation_features: Optional[int] = None

    def fit(self, X, y, X_val=None, y_val=None, stopping_condition: Callable[[List[float]], bool] = worst_of_five):
        """Solve system using validation set for automatic early stopping.

        :param X:
        :param y:
        :param X_val:
        :param y_val:
        :param stopping_condition:  func(validation_error_list) -> bool. Function returning True to stop solution early.
        :return:
        """

        # Lanczos solver works only with vector 'y'
        if len(y.shape) == 2:
            if y.shape[1] != 1:
                raise ValueError("Lanczos solver accepts only vector 'y'. "
                                 "Use separate model for each output in multi-output problem.")
            else:
                y = y.ravel()

        dim = X.shape[1]
        r = [np.nan] * (dim + 1)
        q = [np.nan] * (dim + 1)
        a = [np.nan] * (dim + 1)
        b = [np.nan] * (dim + 2)

        r[0] = X.T @ y
        q[0] = np.zeros(dim)
        b[1] = np.linalg.norm(r[0])

        e1 = np.zeros((dim,))
        e1[0] = 1

        rmse_v = []
        B = None

        for j in range(1, dim + 1):
            q[j] = r[j - 1] / b[j]
            M = X @ q[j]
            a[j] = M.T @ M
            r[j] = X.T @ M - a[j] * q[j] - b[j] * q[j - 1]
            b[j + 1] = np.linalg.norm(r[j])

            if j > 2:
                Z = np.diag(a[1:j + 1]) + np.diag(b[2:j + 1], -1) + np.diag(b[2:j + 1], 1)
                Q = np.vstack(q[1:j + 1]).T
                B = Q @ np.linalg.pinv(Z) @ e1[:j] * b[1]

                if X_val is None:  # early stopping disabled
                    continue

                rmse_v.append(np.mean((X_val @ B - y_val) ** 2) ** 0.5)
                if stopping_condition(rmse_v):
                    break

        self.coef_ = B
        return self
