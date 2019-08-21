import scipy as sp
from enum import Enum
from sklearn.metrics import pairwise_distances
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state


class HiddenLayerType(Enum):
    RANDOM = 1    # Gaussian random projection
    SPARSE = 2    # Sparse Random Projection
    PAIRWISE = 3  # Pairwise kernel with a number of centroids


def dummy(x):
    return x


def flatten(items):
    """Yield items from any nested iterable."""
    for x in items:
        # don't break strings into characters
        if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def _is_list_of_strings(obj):
    return obj is not None and all(isinstance(elem, str) for elem in obj)


def _dense(X):
    if sp.sparse.issparse(X):
        return X.todense()
    else:
        return X


class PairwiseRandomProjection(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=100, pairwise_metric='l2', n_jobs=None, random_state=None):
        """Pairwise distances projection with random centroids.

        Parameters
        ----------
        n_components : int
            Number of components (centroids) in the projection. Creates the same number of output features.

        pairwise_metric : str
            A valid pairwise distance metric, see pairwise-distances_.
            .. _pairwise-distances: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances

        n_jobs : int or None, optional, default=None
            Number of jobs to use in distance computations, or `None` for no parallelism.
            Passed to _pairwise-distances function.

        random_state
            Used for random generation of centroids.
        """
        self.n_components = n_components
        self.pairwise_metric = pairwise_metric
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y=None):
        """Generate artificial centroids.

        Centroids are sampled from a normal distribution. They work best if the data is normalized.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            Input data
        """
        X = check_array(X, accept_sparse=True)
        self.random_state_ = check_random_state(self.random_state)

        if self.n_components <= 0:
            raise ValueError("n_components must be greater than 0, got %s" % self.n_components)

        self.components_ = self.random_state_.randn(self.n_components, X.shape[1])
        self.n_jobs_ = 1 if self.n_jobs is None else self.n_jobs
        return self

    def transform(self, X):
        """Compute distance matrix between input data and the centroids.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            Input data samples.

        Returns
        -------
        X_dist : numpy array
            Distance matrix between input data samples and centroids.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'components_')

        if X.shape[1] != self.components_.shape[1]:
            raise ValueError(
                'Impossible to perform projection: X at fit stage had a different number of features. '
                '(%s != %s)' % (X.shape[1], self.components_.shape[1]))

        try:
            X_dist = pairwise_distances(X, self.components_, n_jobs=self.n_jobs_, metric=self.pairwise_metric)
        except TypeError:
            # scipy distances that don't support sparse matrices
            X_dist = pairwise_distances(_dense(X), _dense(self.components_), n_jobs=self.n_jobs_, metric=self.pairwise_metric)

        return X_dist