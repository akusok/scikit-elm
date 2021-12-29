from __future__ import annotations

from .solver import CholeskySolver

import warnings
from numpy.typing import ArrayLike

from sklearn.exceptions import DataConversionWarning
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_X_y, check_array

from scipy.linalg import LinAlgWarning
warnings.simplefilter("ignore", LinAlgWarning)


class BatchCholeskySolver(BaseEstimator, RegressorMixin):

    def __init__(self, alpha: float = 1e-7):
        self.alpha = alpha

    # interface to solver parameters, for save/load, and other usages
    @property
    def XtY_(self):
        return self.solver_.XtY

    @XtY_.setter
    def XtY_(self, value):
        self.solver_.XtY = value

    @property
    def XtX_(self):
        return self.solver_.XtX

    @XtX_.setter
    def XtX_(self, value):
        self.solver_.XtX = value

    @property
    def coef_(self):
        return self.solver_.coef_

    @coef_.setter
    def coef_(self, value):
        self.solver_.coef_ = value

    @property
    def intercept_(self):
        return self.solver_.intercept_

    @intercept_.setter
    def intercept_(self, value):
        self.solver_.intercept_ = value

    def fit(self, X, y):
        """Solves an L2-regularized linear system like Ridge regression, overwrites any previous solutions.
        """
        self.solver_ = CholeskySolver(self.alpha)  # reset solution
        self.partial_fit(X, y, compute_output_weights=True)
        return self

    def partial_fit(self, X, y, forget=False, compute_output_weights=True) -> BatchCholeskySolver:
        """Update model with a new batch of data.

        Output weight computation can be temporary turned off for faster processing. This will mark model as
        not fit. Enable `compute_output_weights` in the final call to `partial_fit`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            Training input samples

        y : array-like, shape=[n_samples, n_targets]
            Training targets

        forget : boolean, default False
            Performs a negative update, effectively removing the information given by training
            samples from the model. Output weights need to be re-computed after forgetting data.

        compute_output_weights : boolean, optional, default True
            Whether to compute new output weights (coef_, intercept_). Disable this in intermediate `partial_fit`
            steps to run computations faster, then enable in the last call to compute the new solution.

            .. Note::
                Solution can be updated without extra data by setting `X=None` and `y=None`.
        """

        if self.alpha < 0:
            raise ValueError("Regularization parameter alpha must be non-negative.")

        # solution only
        if X is None and y is None and compute_output_weights:
            self.solver_.compute_output_weights()
            self.is_fitted_ = True
            return self

        # validate parameters
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True, y_numeric=True, ensure_2d=True)
        if len(y.shape) > 1 and y.shape[1] == 1:
            msg = "A column-vector y was passed when a 1d array was expected.\
                   Please change the shape of y to (n_samples, ), for example using ravel()."
            warnings.warn(msg, DataConversionWarning)


        # do the model update + solution
        if forget:
            self.solver_.batch_forget(X, y, compute_output_weights=compute_output_weights)
        else:
            self.solver_.batch_update(X, y, compute_output_weights=compute_output_weights)
        self.n_features_in_ = X.shape[1]

        # reset "is_fitted" status if no solution requested
        if hasattr(self, 'is_fitted_') and not compute_output_weights:
            delattr(self, 'is_fitted_')
            return self

        self.is_fitted_ = True
        return self

    def compute_output_weights(self):
        self.solver_.compute_output_weights()

    def predict(self, X) -> ArrayLike:
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True)
        return safe_sparse_dot(X, self.solver_.coef_, dense_output=True) + self.solver_.intercept_
