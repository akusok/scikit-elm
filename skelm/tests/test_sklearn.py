import pytest
from sklearn.utils.estimator_checks import check_estimator
from skelm import ELMRegressor, ELMClassifier, BatchCholeskySolver, PairwiseRandomProjection


@pytest.mark.parametrize("method", [ELMRegressor, ELMClassifier, BatchCholeskySolver, PairwiseRandomProjection])
def test_DenseData_ScikitLearnCompatibility(method):
    check_estimator(method)

