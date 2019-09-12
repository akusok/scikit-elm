"""
High-level Extreme Learning Machine modules
"""

import numpy as np
import warnings
from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels, type_of_target

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.exceptions import DataConversionWarning

from .hidden_layer import HiddenLayer
from .solver_batch import BatchCholeskySolver
from .utils import _dense



class _BaseELM(BaseEstimator):

    def __init__(self, alpha=1e-7, batch_size=None, include_original_features=False,
                 n_neurons=None, ufunc="tanh", density=None, pairwise_metric=None,
                 random_state=None):
        self.alpha = alpha
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.ufunc = ufunc
        self.include_original_features = include_original_features
        self.density = density
        self.pairwise_metric = pairwise_metric
        self.random_state = random_state

    def _init_hidden_layers(self, X):
        """Init an empty model, creating objects for hidden layers and solver.

        Also validates inputs for several hidden layers.
        """
        # only one type of neurons
        if not hasattr(self.n_neurons, '__iter__'):
            hl = HiddenLayer(n_neurons=self.n_neurons, density=self.density, ufunc=self.ufunc,
                             pairwise_metric=self.pairwise_metric, random_state=self.random_state)
            hl.fit(X)
            self.hidden_layers_ = (hl, )

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

            self.hidden_layers_ = []
            for n_neurons, ufunc, density, metric in zip(self.n_neurons, ufuncs, densities, pw_metrics):
                hl = HiddenLayer(n_neurons=n_neurons, density=density, ufunc=ufunc,
                                 pairwise_metric=metric, random_state=self.random_state)
                hl.fit(X)
                self.hidden_layers_.append(hl)

    def _reset(self):
        [delattr(self, attr) for attr in ('n_features_', 'solver_', 'hidden_layers_', 'is_fitted_') if hasattr(self, attr)]

    @property
    def coef_(self):
        return self.solver_.coef_

    @property
    def intercept_(self):
        return self.solver_.intercept_

    def partial_fit(self, X, y=None, compute_output_weights=True):
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

            Example:
                >>> model.partial_fit(X_1, y_1)
                ... model.partial_fit(X_2, y_2)
                ... model.partial_fit(X_3, y_3)    # doctest: +SKIP

            Faster, option 1:
                >>> model.partial_fit(X_1, y_1, compute_output_weights=False)
                ... model.partial_fit(X_2, y_2, compute_output_weights=False)
                ... model.partial_fit(X_3, y_3)    # doctest: +SKIP

            Faster, option 2:
                >>> model.partial_fit(X_1, y_1, compute_output_weights=False)
                ... model.partial_fit(X_2, y_2, compute_output_weights=False)
                ... model.partial_fit(X_3, y_3, compute_output_weights=False)
                ... model.partial_fit(X=None, y=None)    # doctest: +SKIP
        """
        # compute output weights only
        if X is None and y is None and compute_output_weights:
            self.solver_.partial_fit(None, None, compute_output_weights=True)
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
        if not hasattr(self, 'hidden_layers_'):
            self.n_features_ = n_features
            self.solver_ = BatchCholeskySolver(alpha=self.alpha)
            self._init_hidden_layers(X)

        # special case of one-shot processing
        if self.bsize_ >= n_samples:
            H = [hl.transform(X) for hl in self.hidden_layers_]
            H = np.hstack(H if not self.include_original_features else [_dense(X)] + H)
            self.solver_.partial_fit(H, y, compute_output_weights=False)

        else:  # batch processing
            for b_start in range(0, n_samples, self.bsize_):
                b_end = min(b_start + self.bsize_, n_samples)
                b_X = X[b_start:b_end]
                b_y = y[b_start:b_end]

                b_H = [hl.transform(b_X) for hl in self.hidden_layers_]
                b_H = np.hstack(b_H if not self.include_original_features else [_dense(b_X)] + b_H)
                self.solver_.partial_fit(b_H, b_y, compute_output_weights=False)

        # output weights if needed
        if compute_output_weights:
            self.solver_.partial_fit(None, None, compute_output_weights=True)
            self.is_fitted_ = True

        # mark as needing a solution
        elif hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        return self

    def fit(self, X, y=None):
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
        
        #todo: add X as bunch of files support
        
        self._reset()
        self.partial_fit(X, y)
        return self

    def predict(self, X):
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

        H = [hl.transform(X) for hl in self.hidden_layers_]
        if self.include_original_features:
            H = [_dense(X)] + H
        H = np.hstack(H)

        return self.solver_.predict(H)


class ELMRegressor(_BaseELM, RegressorMixin):
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

class ELMClassifier(_BaseELM, ClassifierMixin):
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

    def _update_classes(self, y):
        if not isinstance(self.solver_, BatchCholeskySolver):
            raise ValueError("Only iterative solver supports dynamic class update")

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
        if hasattr(self.solver_, 'XtY_'):
            XtY_old = self.solver_.XtY_
            XtY_new = np.zeros((XtY_old.shape[0], new_classes.shape[0]))
            for i, c in enumerate(old_classes):
                j = np.where(new_classes == c)[0][0]
                XtY_new[:, j] = XtY_old[:, i]
            self.solver_.XtY_ = XtY_new

        # reset the solution
        if hasattr(self.solver_, 'is_fitted_'):
            del self.solver_.is_fitted_

    def partial_fit(self, X, y=None, classes=None, update_classes=False, compute_output_weights=True):
        """Update classifier with new data.

        :param classes: ignored
        :param update_classes: Includes new classes from 'y' into the model;
                               assumes they are set to 0 in all previous targets.
        """

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
        super().partial_fit(X, y_numeric, compute_output_weights=compute_output_weights)
        return self

    def fit(self, X, y=None):
        """Fit a classifier erasing any previously trained model.

        Returns
        -------
        self : object
            Returns self.
        """
        if hasattr(self, "label_binarizer_"):
            del self.label_binarizer_
        self.partial_fit(X, y, compute_output_weights=True)
        return self

    def predict(self, X):
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

    def predict_proba(self, X):
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