import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from typing import Iterable, Callable, List

from .hidden_layer import SLFN, CopyInputsSLFN, HiddenLayer
from .solver_lanczos import LanczosSolver, worst_of_five


class LanczosELM(BaseEstimator, RegressorMixin):

    def __init__(self, include_original_features=False, n_neurons=None,
                 ufunc="tanh", density=None, pairwise_metric=None, random_state=None):
        """Scikit-ELM's version of __init__, that only saves input parameters and does nothing else.
        """
        self.n_neurons = n_neurons
        self.ufunc = ufunc
        self.include_original_features = include_original_features
        self.density = density
        self.pairwise_metric = pairwise_metric
        self.random_state = random_state

    def _make_slfns(self, X) -> Iterable[SLFN]:
        # only one type of neurons
        SLFNs = []
        if not hasattr(self.n_neurons, '__iter__'):
            slfn = HiddenLayer(n_neurons=self.n_neurons, density=self.density, ufunc=self.ufunc,
                               pairwise_metric=self.pairwise_metric, random_state=self.random_state)
            slfn.fit(X)
            SLFNs.append(slfn)

        # several different types of neurons
        else:
            k = len(self.n_neurons)

            # fix default values
            ufuncs = self.ufunc
            if isinstance(ufuncs, str) or not hasattr(ufuncs, "__iter__"):
                ufuncs = [ufuncs] * k

            densities = self.density
            if densities is None or not hasattr(densities, "__iter__"):
                densities = [densities] * k

            pw_metrics = self.pairwise_metric
            if pw_metrics is None or isinstance(pw_metrics, str):
                pw_metrics = [pw_metrics] * k

            if not k == len(ufuncs) == len(densities) == len(pw_metrics):
                raise ValueError("Inconsistent parameter lengths for model with {} different types of neurons.\n"
                                 "Set 'ufunc', 'density' and 'pairwise_distances' by lists "
                                 "with {} elements, or leave the default values.".format(k, k))

            for n_neurons, ufunc, density, metric in zip(self.n_neurons, ufuncs, densities, pw_metrics):
                slfn = HiddenLayer(n_neurons=n_neurons, density=density, ufunc=ufunc,
                                   pairwise_metric=metric, random_state=self.random_state)
                slfn.fit(X)
                SLFNs.append(slfn)

        if self.include_original_features:
            SLFNs.append(CopyInputsSLFN(X))

        return SLFNs

    def _init_model(self, X):
        self.SLFNs_ = self._make_slfns(X)
        self.solver_ = LanczosSolver()

    @classmethod
    def _add_bias(cls, H):
        return np.hstack((np.ones((H.shape[0], 1)), H))

    def fit(self, X, y, X_val=None, y_val=None, stopping_condition: Callable[[List[float]], bool] = worst_of_five):
        X, y = check_X_y(X, y, multi_output=False, accept_sparse=False)

        if not hasattr(self, "solver_"):
            self._init_model(X)

        H = np.hstack([slfn.transform(X) for slfn in self.SLFNs_])
        H_bias = LanczosELM._add_bias(H)

        if X_val is None:
            H_val_bias = None
        else:
            H_val = np.hstack([slfn.transform(X_val) for slfn in self.SLFNs_])
            H_val_bias = LanczosELM._add_bias(H_val)

        self.solver_.fit(H_bias, y, H_val_bias, y_val, stopping_condition)
        return self

    def predict(self, X):
        check_is_fitted(self, "SLFNs_")
        X = check_array(X)
        H = np.hstack([slfn.transform(X) for slfn in self.SLFNs_])
        H_bias = LanczosELM._add_bias(H)
        yh = H_bias @ self.solver_.coef_
        return yh
