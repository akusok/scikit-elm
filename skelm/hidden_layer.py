import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from .utils import PairwiseRandomProjection, HiddenLayerType, dummy

# suppress annoying warning of random projection into a higher-dimensional space
import warnings
warnings.filterwarnings("ignore", message="DataDimensionalityWarning")


def auto_neuron_count(n, d):
    # computes default number of neurons for `n` data samples with `d` features
    return min(int(250 * np.log(1 + d/10) - 15), n//3 + 1)


ufuncs = {"tanh": np.tanh,
          "sigm": sp.special.expit,
          "relu": lambda x: np.maximum(x, 0),
          "lin": dummy,
          None: dummy}


class HiddenLayer(BaseEstimator, TransformerMixin):

    def __init__(self, n_neurons=None, density=None, ufunc="tanh", pairwise_metric=None, random_state=None):
        self.n_neurons = n_neurons
        self.density = density
        self.ufunc = ufunc
        self.pairwise_metric = pairwise_metric
        self.random_state = random_state

    def _fit_random_projection(self, X):
        self.hidden_layer_ = HiddenLayerType.RANDOM
        self.projection_ = GaussianRandomProjection(n_components=self.n_neurons_, random_state=self.random_state_)
        self.projection_.fit(X)

    def _fit_sparse_projection(self, X):
        self.hidden_layer_ = HiddenLayerType.SPARSE
        self.projection_ = SparseRandomProjection(n_components=self.n_neurons_, density=self.density,
                                                  dense_output=True, random_state=self.random_state_)
        self.projection_.fit(X)

    def _fit_pairwise_projection(self, X):
        self.hidden_layer_ = HiddenLayerType.PAIRWISE
        self.projection_ = PairwiseRandomProjection(n_components=self.n_neurons_,
                                                    pairwise_metric=self.pairwise_metric,
                                                    random_state=self.random_state_)
        self.projection_.fit(X)

    def fit(self, X, y=None):
        # basic checks
        X = check_array(X, accept_sparse=True)

        # handle random state
        self.random_state_ = check_random_state(self.random_state)

        # get number of neurons
        n, d = X.shape
        self.n_neurons_ = int(self.n_neurons) if self.n_neurons is not None else auto_neuron_count(n, d)

        # fit a projection
        if self.pairwise_metric is not None:
            self._fit_pairwise_projection(X)
        elif self.density is not None:
            self._fit_sparse_projection(X)
        else:
            self._fit_random_projection(X)

        if self.ufunc in ufuncs.keys():
            self.ufunc_ = ufuncs[self.ufunc]
        elif callable(self.ufunc):
            self.ufunc_ = self.ufunc
        else:
            raise ValueError("Ufunc transformation function not understood: ", self.ufunc)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "is_fitted_")

        X = check_array(X, accept_sparse=True)
        n_features = self.projection_.components_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d" % (X.shape[1], n_features))

        if self.hidden_layer_ == HiddenLayerType.PAIRWISE:
            return self.projection_.transform(X)  # pairwise projection ignores ufunc

        return self.ufunc_(self.projection_.transform(X))
