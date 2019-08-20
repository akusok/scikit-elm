import numpy as np
import scipy as sp
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_X_y, check_array

from dask import distributed
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import dask.array as da


class DaskCholeskySolver(BaseEstimator, RegressorMixin):
    """Out-of-core linear system solver with Dask back-end.

    Parameters
    ----------

    alpha : float, non-negative
        L2 regularization parameter, larger value means stronger effect. The value may be
        increased if the system fails to converge; actual used value stored in `alpha_` parameter.

    batch_size : int
        Batch size for **samples and features**. Computations proceed on square blocks of data.
        For optimal performance, use a number of features that is equal or a bit less than multiple
        of a batch size; e.g. 8912 features with 3000 batch size.

    swap_dir : str
        Directory for temporary storage of Dask data that does not fit in memory. A large and fast
        storage is advised, like a local SSD.


    Attributes
    ----------

    cluster_ : object
        An instance of `dask.distributed.LocalCluster`.

    client_ : object
        Dask client for running computations.
    """

    def __init__(self, alpha=1e-7, batch_size=2000, swap_dir=None):
        self.alpha = alpha
        self.batch_size = batch_size
        self.swap_dir = swap_dir

    def _init_dask(self):
        self.cluster_ = LocalCluster( n_workers=2, local_dir=self.swap_dir)
        self.client_ = Client(self.cluster_)
        print("Running on:")
        print(self.client_)

    def fit(self, X, y):


        self.W_ = da.random.normal

        return self

    def predict(self, X):
        return None


class BBvdsnjvlsdnjhbgfndjvksdjkvlndsf(BaseEstimator, RegressorMixin):

    def __init__(self, alpha=1e-7):
        self.alpha = alpha

    def _init_XY(self, X, y):
        """Initialize covariance matrices, including a separate bias term.
        """
        d_in = X.shape[1]
        self._XtX = np.eye(d_in + 1) * self.alpha
        self._XtX[0, 0] = 0
        if len(y.shape) == 1:
            self._XtY = np.zeros((d_in + 1,))
        else:
            self._XtY = np.zeros((d_in + 1, y.shape[1]))

    @property
    def XtY_(self):
        return self._XtY

    @property
    def XtX_(self):
        return self._XtX

    @XtY_.setter
    def XtY_(self, value):
        self._XtY = value

    @XtX_.setter
    def XtX_(self, value):
        self._XtX = value

    def _solve(self):
        """Second stage of solution (X'X)B = X'Y using Cholesky decomposition.

        Sets `is_fitted_` to True.
        """
        B = sp.linalg.solve(self._XtX, self._XtY, assume_a='pos', overwrite_a=False, overwrite_b=False)
        self.coef_ = B[1:]
        self.intercept_ = B[0]
        self.is_fitted_ = True

    def _reset(self):
        """Erase solution and data matrices.
        """
        [delattr(self, attr) for attr in ('_XtX', '_XtY', 'coef_', 'intercept_', 'is_fitted_') if hasattr(self, attr)]

    def fit(self, X, y):
        """Solves an L2-regularized linear system like Ridge regression, overwrites any previous solutions.
        """
        self._reset()  # remove old solution
        self.partial_fit(X, y, compute_output_weights=True)
        return self

    def partial_fit(self, X, y, compute_output_weights=True):
        """Update model with a new batch of data.

        Output weight computation can be temporary turned off for faster processing. This will mark model as
        not fit. Enable `compute_output_weights` in the final call to `partial_fit`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            Training input samples

        y : array-like, shape=[n_samples, n_targets]
            Training targets

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
            self._solve()
            return self

        # validate parameters
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True, y_numeric=True, ensure_2d=True)
        if len(y.shape) > 1 and y.shape[1] == 1:
            msg = "A column-vector y was passed when a 1d array was expected.\
                   Please change the shape of y to (n_samples, ), for example using ravel()."
            warnings.warn(msg, DataConversionWarning)

        # init temporary data storage
        if not hasattr(self, '_XtX'):
            self._init_XY(X, y)
        else:
            if X.shape[1] + 1 != self._XtX.shape[0]:
                n_new, n_old = X.shape[1], self._XtX.shape[0] - 1
                raise ValueError("Number of features %d does not match previous data %d." % (n_new, n_old))

        # compute temporary data
        X_sum = safe_sparse_dot(X.T, np.ones((X.shape[0],)))
        y_sum = safe_sparse_dot(y.T, np.ones((y.shape[0],)))
        self._XtX[0, 0] += X.shape[0]
        self._XtX[1:, 0] += X_sum
        self._XtX[0, 1:] += X_sum
        self._XtX[1:, 1:] += X.T @ X

        self._XtY[0] += y_sum
        self._XtY[1:] += X.T @ y

        # solve
        if not compute_output_weights:
            # mark as not fitted
            [delattr(self, attr) for attr in ('coef_', 'intercept_', 'is_fitted_') if hasattr(self, attr)]
        else:
            self._solve()
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True)
        return safe_sparse_dot(X, self.coef_, dense_output=True) + self.intercept_
