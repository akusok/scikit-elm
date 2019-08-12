import numpy as np
import scipy as sp
from enum import Enum

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import QuantileTransformer
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from .utils import _dense

# suppress annoying warning of random projection into a higher-dimensional space
import warnings
warnings.filterwarnings("ignore", message="DataDimensionalityWarning")


def auto_neuron_count(n, d):
    # computes default number of neurons for `n` data samples with `d` features
    return min(int(250 * np.log(1 + d/10) - 15), n//3 + 1)


def dummy(x):
    return x


class HiddenLayerType(Enum):
    RANDOM = 1    # Gaussian random projection
    SPARSE = 2    # Sparse Random Projection
    PAIRWISE = 3  # Pairwise kernel with a number of centroids
    

ufuncs = {"tanh": np.tanh,
          "sigm": sp.special.expit,
          "relu": lambda x: np.maximum(x, 0),
          "lin": dummy,
          None: dummy}


class HiddenLayer(BaseEstimator, TransformerMixin):
    #todo: use random state properly
    
    def __init__(self, n_neurons=None, density=None, ufunc="tanh", pairwise_metric=None,
                 include_original_features=False, random_state=None):
        self.n_neurons = n_neurons
        self.density = density
        self.ufunc = ufunc
        self.pairwise_metric = pairwise_metric
        self.include_original_features = include_original_features
        self.random_state = random_state
        
    def _fit_random_projection(self, X):
        self.hidden_layer_ = HiddenLayerType.RANDOM
        self.projection_ = GaussianRandomProjection(n_components=self.n_neurons_, random_state=self.random_state_)
        self.projection_.fit(X)
            
    def _fit_sparse_projection(self, X):
        self.hidden_layer_ = HiddenLayerType.SPARSE
        self.projection_ = SparseRandomProjection(
            n_components=self.n_neurons_, density=self.density, dense_output=True, random_state=self.random_state_)
        self.projection_.fit(X)

    def _fit_pairwise_projection(self, X):
        # use quantile transformer to fit random centroids to data distribution
        self.hidden_layer_ = HiddenLayerType.PAIRWISE
        transformer = QuantileTransformer(n_quantiles=min(100, X.shape[0]),
                                          ignore_implicit_zeros=True)
        transformer.fit(X)
        random_centroids = np.random.rand(self.n_neurons_, X.shape[1])
        self.centroids_ = transformer.inverse_transform(random_centroids)

    def _project(self, X):
        # validate X dimension
        if self.hidden_layer_ is HiddenLayerType.PAIRWISE:
            n_features = self.centroids_.shape[1]
        else:
            n_features = self.projection_.components_.shape[1]

        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d" % (X.shape[1], n_features))

        if self.hidden_layer_ is HiddenLayerType.PAIRWISE:
            try:
                H = pairwise_distances(X, self.centroids_, metric=self.pairwise_metric)
            except TypeError:
                # scipy distances that don't support sparse matrices
                H = pairwise_distances(_dense(X), _dense(self.centroids_), metric=self.pairwise_metric)
        else:
            H = self.ufunc_(self.projection_.transform(X))

        if self.include_original_features:
            H = np.hstack((X if isinstance(X, np.ndarray) else np.array(X.todense()), H))

        return H
    
    def fit(self, X, y=None):
        # basic checks
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        # handle random state
        self.random_state_ = check_random_state(self.random_state)
        
        # get number of neurons
        n, d = X.shape
        self.n_neurons_ = self.n_neurons if self.n_neurons is not None else auto_neuron_count(n, d)
        
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
        
        return self
    
    def transform(self, X):
        check_is_fitted(self, "hidden_layer_")
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        return self._project(X)
