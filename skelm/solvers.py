import numpy as np
import scipy as sp
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_X_y, check_array


class IterativeSolver(BaseEstimator, RegressorMixin):
    
    def __init__(self, alpha=1e-7):
        self.alpha = alpha
    
    #todo: XtX, XtY parameters with setter and getter
    
    def _init_XY(self, X, y):
        """Initialize covariance matrices, including a separate bias term.
        """
        d_in = X.shape[1]
        self.XtX_ = np.eye(d_in + 1) * self.alpha
        self.XtX_[0, 0] = 0
        if len(y.shape) == 1:
            self.XtY_ = np.zeros((d_in + 1, )) 
        else:
            self.XtY_ = np.zeros((d_in + 1, y.shape[1]))
    
    def fit(self, X, y):
        """Solves an L2-regularized linear system like Ridge regression, overwrites any previous solutions.
        """
        # reset old solution
        if hasattr(self, "XtX_"):
            del self.XtX_, self.XtY_, self.coef_, self.intercept_
            
        self.partial_fit(X, y, skip_solution=False)
        return self
    
    def partial_fit(self, X, y, skip_solution=False):
        """Update model with a new batch of data.
        
        Final solution can be temporary turned off for faster processing.

        :param skip_solution: Skips computing the new (coef_, intercept_) solution and erases the existing one.
                              Speed up computations by using this in intermediate partial_fits; then finish by
                              the last partial_fit with skip_soluton=False.
        """
        # solution only
        if X is None and y is None and skip_solution == True:
            B = sp.linalg.solve(self.XtX_, self.XtY_, assume_a='pos', overwrite_a=False, overwrite_b=False)
            self.coef_ = B[1:]
            self.intercept_ = B[0]
            return self

        # validate parameters
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], multi_output=True, y_numeric=True, ensure_2d=True)
        if len(y.shape) > 1 and y.shape[1] == 1:
            msg = "A column-vector y was passed when a 1d array was expected.\
                   Please change the shape of y to (n_samples, ), for example using ravel()."
            warnings.warn(msg, DataConversionWarning)
        
        # init temporary data storage
        if not hasattr(self, 'XtX_'):
            self._init_XY(X, y)
        else:
            if X.shape[1] + 1 != self.XtX_.shape[0]:
                raise ValueError("Number of features %d does not match previous "
                                 "data %d." % (X.shape[1], self.XtX_.shape[0] - 1))
                
        # compute temporary data
        X_sum = X.T @ np.ones((X.shape[0],))
        y_sum = y.T @ np.ones((y.shape[0],))
        self.XtX_[0, 0] += X.shape[0]
        self.XtX_[1:, 0] += X_sum
        self.XtX_[0, 1:] += X_sum
        self.XtX_[1:, 1:] += X.T @ X

        self.XtY_[0] += y_sum
        self.XtY_[1:] += X.T @ y
        
        # solve
        if skip_solution:
            if hasattr(self, 'coef_'):
                del self.coef_, self.intercept_
        else:
            B = sp.linalg.solve(self.XtX_, self.XtY_, assume_a='pos', overwrite_a=False, overwrite_b=False)
            self.coef_ = B[1:]
            self.intercept_ = B[0]

        return self
    
    def predict(self, X, **kwargs):
        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        return X @ self.coef_ + self.intercept_
