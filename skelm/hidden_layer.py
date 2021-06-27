import numpy as np
from scipy.special import expit
from scipy.spatial.distance import cdist
from typing import Protocol
from abc import abstractmethod
from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from .utils import PairwiseRandomProjection

# suppress annoying warning of random projection into a higher-dimensional space
import warnings
warnings.filterwarnings("ignore", message="DataDimensionalityWarning")


def auto_neuron_count(n, d):
    # computes default number of neurons for `n` data samples with `d` features
    return min(int(250 * np.log(1 + d/10) - 15), n//3 + 1)


ufuncs = {"tanh": np.tanh,
          "sigm": expit,
          "relu": lambda x: np.maximum(x, 0),
          "lin": lambda x: x,
          None: lambda x: x}


class SLFN(Protocol):
    """Single hidden Layer Feed-forward neural Network, general ELM hidden neuron layer."""

    n_neurons: int  # number of neurons

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Hidden layer projects data into another format."""


class DenseSLFN:
    def __init__(self, W, ufunc):
        self.W = W
        self.ufunc = ufunc
        self.n_neurons = W.shape[1]

    def transform(self, X):
        H = self.ufunc(X @ self.W)
        return H


class PairwiseSLFN:
    def __init__(self, X, k):
        self.basis = X[:k]
        self.n_neurons = self.basis.shape[0]

    def transform(self, X):
        H = cdist(X, self.basis, metric="euclidean")
        return H


class CopyInputsSLFN:
    def __init__(self, X):
        self.n_neurons = X.shape[1]

    def transform(self, X):
        return X


class RandomProjectionSLFN(SLFN):
    def __init__(self, X, n_neurons, ufunc=np.tanh, random_state=None):
        self.n_neurons = n_neurons
        self.ufunc = ufunc
        self.projection = GaussianRandomProjection(
            n_components=n_neurons, random_state=random_state
        )
        self.projection.fit(X)

    def transform(self, X):
        return self.ufunc(self.projection.transform(X))


class SparseRandomProjectionSLFN(SLFN):
    def __init__(self, X, n_neurons, density=0.1, ufunc=np.tanh, random_state=None):
        self.n_neurons = n_neurons
        self.ufunc = ufunc
        self.projection = SparseRandomProjection(
            n_components=n_neurons, density=density, dense_output=True, random_state=random_state
        )
        self.projection.fit(X)

    def transform(self, X):
        return self.ufunc(self.projection.transform(X))


class PairwiseRandomProjectionSLFN(SLFN):
    def __init__(self, X, n_neurons, pairwise_metric="euclidean", random_state=None):
        self.n_neurons = n_neurons
        self.projection = PairwiseRandomProjection(
            n_components=n_neurons, pairwise_metric=pairwise_metric, random_state=random_state
        )
        self.projection.fit(X)

    def transform(self, X):
        return self.projection.transform(X)


class HiddenLayer(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible interface for SLFN.

    Handles parameter transformation and input checks.
    Not a part of ELM; for stand-alone usage.
    """

    def __init__(self, n_neurons=None, density=None, ufunc="tanh", pairwise_metric=None, random_state=None):
        self.n_neurons = n_neurons
        self.density = density
        self.ufunc = ufunc
        self.pairwise_metric = pairwise_metric
        self.random_state = random_state

    def fit(self, X, y=None):
        # basic checks
        X: ArrayLike = check_array(X, accept_sparse=True)
        _ = y  # suppress warning for sklearn compatibility

        # handle random state
        self.random_state_ = check_random_state(self.random_state)

        # get number of neurons
        n, d = X.shape
        self.n_neurons_ = int(self.n_neurons) if self.n_neurons is not None else auto_neuron_count(n, d)
        self.n_features_in_ = d

        if self.ufunc in ufuncs.keys():
            self.ufunc_ = ufuncs[self.ufunc]
        elif callable(self.ufunc):
            self.ufunc_ = self.ufunc
        else:
            raise ValueError("Ufunc transformation function not understood: ", self.ufunc)

        # choose a suitable projection
        if self.pairwise_metric is not None:
            self.SLFN_ = PairwiseRandomProjectionSLFN(X, self.n_neurons_, self.pairwise_metric, self.random_state_)
        elif self.density is not None:
            self.SLFN_ = SparseRandomProjectionSLFN(X, self.n_neurons_, self.density, self.ufunc_, self.random_state_)
        else:
            self.SLFN_ = RandomProjectionSLFN(X, self.n_neurons_, self.ufunc_, self.random_state_)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "is_fitted_")

        clean_X: ArrayLike = check_array(X, accept_sparse=True)
        if clean_X.shape[1] != self.n_features_in_:
            raise ValueError("X has %d features per sample; expecting %d" % (clean_X.shape[1], self.n_features_in_))

        return self.SLFN_.transform(clean_X)
