import pytest
from sklearn.utils.estimator_checks import check_estimator
from skelm import ELMRegressor, ELMClassifier
from skelm.solvers import BatchCholeskySolver


@pytest.mark.parametrize("method", [ELMRegressor, ELMClassifier, BatchCholeskySolver])
def test_DenseData_ScikitLearnCompatibility(method):
    check_estimator(method)

