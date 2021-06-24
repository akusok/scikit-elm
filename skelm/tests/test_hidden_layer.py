import unittest
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, lil_matrix

import warnings
from sklearn.exceptions import DataDimensionalityWarning

from sklearn.datasets import load_boston, load_iris
from sklearn.preprocessing import RobustScaler
from sklearn.utils.estimator_checks import check_estimator

from skelm.hidden_layer import (
    RandomProjectionSLFN,
    SparseRandomProjectionSLFN,
    PairwiseRandomProjectionSLFN,
    CopyInputsSLFN,
    HiddenLayer
)


c_X, c_y = load_iris(return_X_y=True)
r_X, r_y = load_boston(return_X_y=True)
r_X = RobustScaler().fit_transform(r_X)

data_formats = (
    (r_X, r_y),
    (c_X, c_y),
    (csc_matrix(r_X), r_y),
    (csc_matrix(c_X), c_y),
    (csr_matrix(r_X), r_y),
    (csr_matrix(c_X), c_y),
    (coo_matrix(r_X), r_y),
    (coo_matrix(c_X), c_y),
    (lil_matrix(r_X), r_y),
    (lil_matrix(c_X), c_y)
)


class TestNativeSLFNs(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", DataDimensionalityWarning)

    def test_RandomProjection_SetNumberOfNeurons(self):
        for X, y in data_formats:
            with self.subTest(matrix_type=type(X)):
                rp = RandomProjectionSLFN(X, 5, ufunc=np.tanh, random_state=0)
                H_rp = rp.transform(X)
                self.assertEqual(H_rp.shape[1], 5)

    def test_SparseRandomProjection_SetNumberOfNeurons(self):
        for X, y in data_formats:
            with self.subTest(matrix_type=type(X)):
                srp = SparseRandomProjectionSLFN(X, 5, density=0.5, ufunc=np.tanh, random_state=0)
                H_srp = srp.transform(X)
                self.assertEqual(H_srp.shape[1], 5)

    def test_PairwiseRandomProjection_SetNumberOfNeurons(self):
        for X, y in data_formats:
            with self.subTest(matrix_type=type(X)):
                prp = PairwiseRandomProjectionSLFN(X, 5, pairwise_metric="cosine", random_state=0)
                H_prp = prp.transform(X)
                self.assertEqual(H_prp.shape[1], 5)

    def test_SLFN_LinearPart(self):
        for X, y in data_formats:
            with self.subTest(matrix_type=type(X)):
                dummy_projection = CopyInputsSLFN(X)
                H_copy = dummy_projection.transform(X)
                # "allclose" for sparse matrices
                self.assertEqual(H_copy.shape, X.shape)
                self.assertLess(np.abs(H_copy - X).max(), 1e-5, msg="Arrays are not equal")



###########################
## Scikit-Learn's wrapper

class TestScikitLearnCompatibleInterface(unittest.TestCase):

    def test_HiddenLayer_IsScikitLearnEstimator(self):
        model_rp = HiddenLayer(5)
        model_srp = HiddenLayer(5, density=0.5)
        model_pairwise = HiddenLayer(5, pairwise_metric="cosine")
        check_estimator(model_rp)
        check_estimator(model_srp)
        check_estimator(model_pairwise)

    def test_Ufuncs_Sigmoid(self):
        for X, y in data_formats:
            for ufunc in ["tanh", "sigm", "relu", "lin", None]:
                with self.subTest(ufunc=ufunc, matrix_type=type(X)):
                    model = HiddenLayer(5, ufunc=ufunc)
                    model.fit(X)
                    H = model.transform(X)
                    self.assertEqual(H.shape[1], 5)

    def test_DefaultHiddenLayer_UseGaussianRandomProjection(self):
        for X, y in data_formats:
            with self.subTest(matrix_type=type(X)):
                model = HiddenLayer(5)
                model.fit(X)
                self.assertIsInstance(model.SLFN_, RandomProjectionSLFN)

    def test_SparseHiddenLayer_UseSparseRandomProjection(self):
        for X, y in data_formats:
            with self.subTest(matrix_type=type(X)):
                model = HiddenLayer(5, density=0.99)
                model.fit(X)
                self.assertIsInstance(model.SLFN_, SparseRandomProjectionSLFN)

    def test_PairwiseMetric_UsePairwiseProjection(self):
        for X, y in data_formats:
            with self.subTest(matrix_type=type(X)):
                model = HiddenLayer(5, pairwise_metric="cosine")
                model.fit(X)
                self.assertIsInstance(model.SLFN_, PairwiseRandomProjectionSLFN)

    def test_PairwiseKernel_TooManyNeurons_StillWorks(self):
        """More neurons than original data features available; model still works.
        """
        for X, y in data_formats:
            with self.subTest(matrix_type=type(X)):
                model = HiddenLayer(n_neurons=3 * X.shape[0], pairwise_metric="cosine")
                model.fit(X)

    def test_Ufunc_WrongName_ReturnsValueError(self):
        model = HiddenLayer(5, ufunc="UnIcOrN")
        with self.assertRaises(ValueError):
            model.fit(r_X)

    def test_Ufunc_OtherUfunc_Works(self):
        for X, y in data_formats:
            with self.subTest(matrix_type=type(X)):
                model = HiddenLayer(5, ufunc=np.mod)
                model.fit(X)

    def test_Ufunc_CustomCreatedUfunc_Works(self):
        def foo(x):
            return x + 1
        my_ufunc = np.frompyfunc(foo, 1, 1)
        for X, y in data_formats:
            with self.subTest(matrix_type=type(X)):
                model = HiddenLayer(5, ufunc=my_ufunc)
                model.fit(X)

    def test_Ufunc_CustomLambdaFunction_Works(self):
        embiggen = lambda x: x + 1
        for X, y in data_formats:
            with self.subTest(matrix_type=type(X)):
                model = HiddenLayer(5, ufunc=embiggen)
                model.fit(X)

    def test_PairwiseDistances_AllKinds_FromScikitLearn(self):
        for pm in ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]:
            with self.subTest(pairwise_metric=pm):
                model = HiddenLayer(5, pairwise_metric=pm)
                model.fit(r_X)

    def test_PairwiseDistances_AllKinds_FromScipy(self):
        for pm in ["braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming",
                   "jaccard", "kulsinski", "mahalanobis", "minkowski", "rogerstanimoto",
                   "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean"]:
            with self.subTest(pairwise_metric=pm):
                model = HiddenLayer(5, pairwise_metric=pm)
                model.fit(r_X)

