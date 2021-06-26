import numpy as np
from numpy.typing import ArrayLike
from typing import Optional

from .solver import Solver


class LanczosSolver(Solver):

    coef_ = None
    intercept_ = np.zeros((1,))
    X_val: Optional[ArrayLike] = None
    y_val: Optional[ArrayLike] = None
    data_features: Optional[int] = None
    validation_features: Optional[int] = None

    def fit(self, X, y, X_val=None, y_val=None):
        """Solve system using validation set for automatic early stopping.

        :param X: training inputs
        :param y: training targets
        :param X_val: validation inputs
        :param y_val: validation targets
        """
        l = X.shape[1]
        r = [np.nan] * (l + 1)
        q = [np.nan] * (l + 1)
        a = [np.nan] * (l + 1)
        b = [np.nan] * (l + 2)

        r[0] = X.T @ y
        q[0] = np.zeros(l)
        b[1] = np.linalg.norm(r[0])

        e1 = np.zeros((l,))
        e1[0] = 1

        rmse_v = []
        B = None

        for j in range(1, l + 1):
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
                if len(rmse_v) > 3 and (rmse_v[-1] > max(rmse_v[-5:-1])):
                    break

        self.coef_ = B
        return self
