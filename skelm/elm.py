"""
High-level Extreme Learning Machine modules
"""

from __future__ import annotations

import numpy as np
import warnings
from scipy.special import expit
from typing import Protocol, Iterable, cast, Optional
from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.exceptions import DataConversionWarning, DataDimensionalityWarning

from .hidden_layer import HiddenLayer, SLFN, CopyInputsSLFN
from .solver_batch import BatchCholeskySolver
from .solver import Solver, BatchSolver

warnings.simplefilter("ignore", DataDimensionalityWarning)


class ELMProtocol(Protocol):
    """Extreme Learning Machine very basic functionality.

    Basic operation is to transform data using each SLFN, stack those features together,
    then compute weights/intercepts of an output linear model with a solver.
    """
    SLFNs: Iterable[SLFN]  # ELM has one or several types of hidden neurons
    solver: Solver  # ELM has an output layer solver
    is_fitted: False  # whether an ELM model is ready to predict

    @property
    def n_neurons(self) -> int:
        """Number of neurons in ELM model"""
        return 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> ELMProtocol:
        """Fit an ELM, return self for command chaining."""

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict outputs for new inputs."""


class BatchELMProtocol(ELMProtocol, Protocol):
    """ELM that supports incremental solution.
    """
    solver: BatchSolver  # batch ELM needs a batch solver

    def partial_fit(self, X: ArrayLike, y: ArrayLike, solve: bool, forget: bool) -> BatchELMProtocol:
        """Update ELM model by adding or removing training data samples.

        Solving can be temporary disabled to speed up processing of multiple data batches.
        """

    def compute_output_weights(self) -> None:
        """Compute solution from internally stored data."""


class BasicELM(ELMProtocol):
    """Minimal ELM implementation."""

    def __init__(self, SLFNs: Iterable[SLFN], solver: Solver):
        self.SLFNs = SLFNs
        self.solver = solver
        self.is_fitted = False

    @property
    def n_neurons(self):
        return sum([slfn.n_neurons for slfn in self.SLFNs])

    def fit(self, X, y):
        H = np.hstack([slfn.transform(X) for slfn in self.SLFNs])
        self.solver.fit(H, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model is not fit")

        H = np.hstack([slfn.transform(X) for slfn in self.SLFNs])
        yh = H @ self.solver.coef_ + self.solver.intercept_
        return yh


class BatchELM(BasicELM, BatchELMProtocol):
    """Minimal incremental ELM implementation."""

    def __init__(self, SLFNs: Iterable[SLFN], solver: BatchSolver):
        super().__init__(SLFNs, solver)
        self.solver = solver  # using batch solver instead of BasicELM's simple solver

    def partial_fit(self, X, y, compute_output_weights=True, forget=False) -> BatchELM:
        H = np.hstack([slfn.transform(X) for slfn in self.SLFNs])
        self.solver.partial_fit(H, y, forget=forget, compute_output_weights=True)
        self.is_fitted = True
        return self

    def compute_output_weights(self):
        self.solver.compute_output_weights()
        self.is_fitted = True


class ScikitELM(BaseEstimator, RegressorMixin):
    """Incremental ELM compatible with Scikit-Learn parametrization.
    """

    def __init__(self, alpha=1e-7, batch_size=None, include_original_features=False,
                 n_neurons=None, ufunc="tanh", density=None, pairwise_metric=None,
                 random_state=None):
        """Scikit-ELM's version of __init__, that only saves input parameters and does nothing else.
        """
        self.alpha = alpha
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.ufunc = ufunc
        self.include_original_features = include_original_features
        self.density = density
        self.pairwise_metric = pairwise_metric
        self.random_state = random_state

    @property
    def n_neurons_(self):
        if not hasattr(self, "model_"):
            return None

        return self.model_.n_neurons

    @property
    def SLFNs_(self):
        if not hasattr(self, "model_"):
            return None
        return self.model_.SLFNs

    @property
    def solver_(self):
        if not hasattr(self, "model_"):
            return None
        return self.model_.solver

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
        """Create composition objects and ELM model.
        """
        SLFNs = self._make_slfns(X)
        solver = BatchCholeskySolver(self.alpha)
        self.model_ = BatchELM(SLFNs, solver)

    def _reset(self):
        runtime_attributes = ('n_features_', 'model_', 'is_fitted_')
        [delattr(self, attr) for attr in runtime_attributes if hasattr(self, attr)]

    def predict(self, X) -> ArrayLike:
        """Predict real valued outputs for new inputs X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data samples.

        Returns
        -------
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            Predicted outputs for inputs X.

            .. attention::

                :mod:`predict` always returns a dense matrix of predicted outputs -- unlike
                in :meth:`fit`, this may cause memory issues at high number of outputs
                and very high number of samples. Feed data by smaller batches in such case.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")
        return self.model_.predict(X)

    def fit(self, X, y) -> ScikitELM:
        """Reset model and fit on the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values used as real numbers.

        Returns
        -------
        self : object
            Returns self.
        """
        self._reset()
        self.partial_fit(X, y, compute_output_weights=True)
        return self

    def partial_fit(self, X, y=None, forget=False, compute_output_weights=True) -> ScikitELM:
        """Update model with a new batch of data.

                |method_partial_fit|

                .. |method_partial_fit| replace:: Output weight computation can be temporary turned off
                    for faster processing. This will mark model as not fit. Enable `compute_output_weights`
                    in the final call to `partial_fit`.

                .. |param_forget| replace:: Performs a negative update, effectively removing the information
                    given by training samples from the model. Output weights need to be re-computed after forgetting
                    data. Forgetting data that have not been learned before leads to unpredictable results.

                .. |param_compute_output_weights| replace::  Whether to compute new output weights
                    (coef_, intercept_). Disable this in intermediate `partial_fit`
                    steps to run computations faster, then enable in the last call to compute the new solution.


                Parameters
                ----------
                X : {array-like, sparse matrix}, shape=[n_samples, n_features]
                    Training input samples

                y : array-like, shape=[n_samples, n_targets]
                    Training targets

                forget : boolean, default False
                    |param_forget|

                compute_output_weights : boolean, optional, default True
                    |param_compute_output_weights|

                    .. Note::
                        Solution can be updated without extra data by setting `X=None` and `y=None`.

                    Example:
                        >>> model.partial_fit(X_1, y_1)
                        ... model.partial_fit(X_2, y_2)
                        ... model.partial_fit(X_3, y_3)    # doctest: +SKIP

                    Faster:
                        >>> model.partial_fit(X_1, y_1, compute_output_weights=False)
                        ... model.partial_fit(X_2, y_2, compute_output_weights=False)
                        ... model.partial_fit(X_3, y_3)    # doctest: +SKIP
        """

        # run late init
        if not hasattr(self, 'model_'):
            self._init_model(X)

        # compute output weights only
        if X is None and y is None and compute_output_weights:
            self.model_.compute_output_weights()
            self.is_fitted_ = True
            return self

        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)
        if len(y.shape) > 1 and y.shape[1] == 1:
            msg = ("A column-vector y was passed when a 1d array was expected. "
                   "Please change the shape of y to (n_samples, ), for example using ravel().")
            warnings.warn(msg, DataConversionWarning)

        n_samples, n_features = X.shape
        if hasattr(self, 'n_features_') and self.n_features_ != n_features:
            raise ValueError('Shape of input is different from what was seen in `fit`')

        # set batch size, default is bsize=2000 or all-at-once with less than 10_000 samples
        self.bsize_ = self.batch_size
        if self.bsize_ is None:
            self.bsize_ = n_samples if n_samples < 10 * 1000 else 2000

        # init model if not fit yet
        if not hasattr(self, 'model_'):
            self.n_features_ = n_features
            self._init_model(X)

        # special case of one-shot processing
        if self.bsize_ >= n_samples:
            self.model_.partial_fit(X, y, compute_output_weights=compute_output_weights, forget=forget)
        # batch processing
        else:
            for b_start in range(0, n_samples, self.bsize_):
                b_end = min(b_start + self.bsize_, n_samples)
                b_X = X[b_start:b_end]
                b_y = y[b_start:b_end]
                self.model_.partial_fit(b_X, b_y, compute_output_weights=False, forget=forget)

        # validate/invalidate current solution
        if compute_output_weights:
            self.model_.compute_output_weights()
            self.is_fitted_ = True
        else:
            if hasattr(self, 'is_fitted_'):
                del self.is_fitted_

        return self


class ELMRegressor(ScikitELM):
    """Extreme Learning Machine for regression problems.

    This model solves a regression problem, that is a problem of predicting continuous outputs.
    It supports multi-variate regression (when ``y`` is a 2d array of shape [n_samples, n_targets].)
    ELM uses ``L2`` regularization, and optionally includes the original data features to
    capture linear dependencies in the data natively.

    Parameters
    ----------
    alpha : float
        Regularization strength; must be a positive float. Larger values specify stronger effect.
        Regularization improves model stability and reduces over-fitting at the cost of some learning
        capacity. The same value is used for all targets in multi-variate regression.

        The optimal regularization strength is suggested to select from a large range of logarithmically
        distributed values, e.g. :math:`[10^{-5}, 10^{-4}, 10^{-3}, ..., 10^4, 10^5]`. A small default
        regularization value of :math:`10^{-7}` should always be present to counter numerical instabilities
        in the solution; it does not affect overall model performance.

        .. attention::
            The model may automatically increase the regularization value if the solution
            becomes unfeasible otherwise. The actual used value contains in ``alpha_`` attribute.

    batch_size : int, optional
        Actual computations will proceed in batches of this size, except the last batch that may be smaller.
        Default behavior is to process all data at once with <10,000 samples, otherwise use batches
        of size 2000.

    include_original_features : boolean, default=False
        Adds extra hidden layer neurons that simpy copy the input data features, adding a linear part
        to the final model solution that can directly capture linear relations between data and
        outputs. Effectively increases `n_neurons` by `n_inputs` leading to a larger model.
        Including original features is generally a good thing if the number of data features is low.

    n_neurons : int or [int], optional
        Number of hidden layer neurons in ELM model, controls model size and learning capacity.
        Generally number of neurons should be less than the number of training data samples, as
        otherwise the model will learn the training set perfectly resulting in overfitting.

        Several different kinds of neurons can be used in the same model by specifying a list of
        neuron counts. ELM will create a separate neuron type for each element in the list.
        In that case, the following attributes ``ufunc``, ``density`` and ``pairwise_metric``
        should be lists of the same length; default values will be automatically expanded into a list.

        .. note::
            Models with <1,000 neurons are very fast to compute, while GPU acceleration is efficient
            starting from 1,000-2,000 neurons. A standard computer should handle up to 10,000 neurons.
            Very large models will not fit in memory but can still be trained by an out-of-core solver.

    ufunc : {'tanh', 'sigm', 'relu', 'lin' or callable}, or a list of those (see n_neurons)
        Transformation function of hidden layer neurons. Includes the following options:
            - 'tanh' for hyperbolic tangent
            - 'sigm' for sigmoid
            - 'relu' for rectified linear unit (clamps negative values to zero)
            - 'lin' for linear neurons, transformation function does nothing
            - any custom callable function like members of ``Numpu.ufunc``

    density : float in range (0, 1], or a list of those (see n_neurons), optional
        Specifying density replaces dense projection layer by a sparse one with the specified
        density of the connections. For instance, ``density=0.1`` means each hidden neuron will
        be connected to a random 10% of input features. Useful for working on very high-dimensional
        data, or for large numbers of neurons.

    pairwise_metric : {'euclidean', 'cityblock', 'cosine' or other}, or a list of those (see n_neurons), optional
        Specifying pairwise metric replaces multiplicative hidden neurons by distance-based hidden
        neurons. This ELM model is known as Radial Basis Function ELM (RBF-ELM).

        .. note::
            Pairwise function neurons ignore ufunc and density.

        Typical metrics are `euclidean`, `cityblock` and `cosine`. For a full list of metrics check
        the `webpage <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html>`_
        of :mod:`sklearn.metrics.pairwise_distances`.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when generating random numbers e.g.
        for hidden neuron parameters. Random state instance is passed to lower level objects and routines.
        Use it for repeatable experiments.

    Attributes
    ----------
    n_neurons_ : int
        Number of automatically generated neurons.

    ufunc_ : function
        Tranformation function of hidden neurons.

    projection_ : object
        Hidden layer projection function.

    solver_ : object
        Solver instance, read solution from there.


    Examples
    --------

    Combining ten sigmoid and twenty RBF neurons in one model:

    >>> model = ELMRegressor(n_neurons=(10, 20),
    ...                      ufunc=('sigm', None),
    ...                      density=(None, None),
    ...                      pairwise_metric=(None, 'euclidean'))   # doctest: +SKIP

    Default values in multi-neuron ELM are automatically expanded to a list

    >>>  model = ELMRegressor(n_neurons=(10, 20),
    ...                       ufunc=('sigm', None),
    ...                       pairwise_metric=(None, 'euclidean'))   # doctest: +SKIP

    >>>  model = ELMRegressor(n_neurons=(30, 30),
    ...                       pairwise_metric=('cityblock', 'cosine'))   # doctest: +SKIP
    """
    pass


class ELMClassifier(ScikitELM, ClassifierMixin):
    """ELM classifier, modified for multi-label classification support.

    :param classes: Set of classes to consider in the model; can be expanded at runtime.
                    Samples of other classes will have their output set to zero.
    :param solver: Solver to use, "default" for build-in Least Squares or "ridge" for Ridge regression


    Example descr...

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, classes=None, alpha=1e-7, batch_size=None, include_original_features=False, n_neurons=None,
                 ufunc="tanh", density=None, pairwise_metric=None, random_state=None):
        super().__init__(alpha, batch_size, include_original_features, n_neurons, ufunc, density, pairwise_metric,
                         random_state)
        self.classes = classes

    @property
    def classes_(self):
        return self.label_binarizer_.classes_

    def _get_tags(self):
        return {"multioutput": True, "multilabel": True}

    def _reset(self):
        if hasattr(self, 'label_binarizer_'):
            delattr(self, 'label_binarizer_')
        super()._reset()

    def _update_classes(self, y):
        if not hasattr(self.model_.solver, "partial_fit"):
            raise RuntimeError("Current solver does not support partial fit: {}".format(self.model_.solver))

        old_classes = self.label_binarizer_.classes_
        partial_classes = clone(self.label_binarizer_).fit(y).classes_

        # no new classes detected
        if set(partial_classes) <= set(old_classes):
            return

        if len(old_classes) < 3:
            raise ValueError("Dynamic class update has to start with at least 3 classes to function correctly; "
                             "provide 3 or more 'classes=[...]' during initialization.")

        # get new classes sorted by LabelBinarizer
        self.label_binarizer_.fit(np.hstack((old_classes, partial_classes)))
        new_classes = self.label_binarizer_.classes_

        # convert existing XtY matrix to new classes
        if hasattr(self.model_.solver, 'XtY_'):
            XtY_old = self.model_.solver.XtY_
            XtY_new = np.zeros((XtY_old.shape[0], new_classes.shape[0]))
            for i, c in enumerate(old_classes):
                j = np.where(new_classes == c)[0][0]
                XtY_new[:, j] = XtY_old[:, i]
            self.model_.solver.XtY_ = XtY_new

        # reset the solution
        self.model_.is_fitted = False
        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

    def partial_fit(self, X, y=None, forget=False, update_classes=False, compute_output_weights=True) -> ELMClassifier:
        """Update classifier with a new batch of data.

        |method_partial_fit|

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            Training input samples

        y : array-like, shape=[n_samples, n_targets]
            Training targets

        forget : boolean, default False
            |param_forget|

        update_classes : boolean, default False
            Include new classes from `y` into the model, assuming they were 0 in all previous samples.

        compute_output_weights : boolean, optional, default True
            |param_compute_output_weights|
        """

        #todo: Warning on strongly non-normalized data

        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)

        # init label binarizer if needed
        if not hasattr(self, 'label_binarizer_'):
            self.label_binarizer_ = LabelBinarizer()
            if type_of_target(y).endswith("-multioutput"):
                self.label_binarizer_ = MultiLabelBinarizer()
            self.label_binarizer_.fit(self.classes if self.classes is not None else y)

        if update_classes:
            self._update_classes(y)

        y_numeric = self.label_binarizer_.transform(y)
        if len(y_numeric.shape) > 1 and y_numeric.shape[1] == 1:
            y_numeric = y_numeric[:, 0]

        super().partial_fit(X, y_numeric, forget=forget, compute_output_weights=compute_output_weights)
        return self

    def fit(self, X, y=None) -> ELMClassifier:
        """Fit a classifier erasing any previously trained model.

        Returns
        -------
        self : object
            Returns self.
        """

        self._reset()
        self.partial_fit(X, y)
        return self

    def predict(self, X) -> ArrayLike:
        """Predict classes of new inputs X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            Returns one most probable class for multi-class problem, or
            a binary vector of all relevant classes for multi-label problem.
        """
        check_is_fitted(self, "is_fitted_")
        scores = super().predict(X)
        return self.label_binarizer_.inverse_transform(scores)

    def predict_proba(self, X) -> ArrayLike:
        """Probability estimation for all classes.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        check_is_fitted(self, "is_fitted_")
        prob = super().predict(X)
        expit(prob, out=prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob