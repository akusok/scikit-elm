import pytest
from sklearn.utils.estimator_checks import check_estimator
from skelm import ELMRegressor, ELMClassifier, BatchCholeskySolver, PairwiseRandomProjection


class MultiTaskELMRegressor(ELMRegressor):
    pass

class MultiTaskELMClassifier(ELMClassifier):
    pass


@pytest.mark.parametrize("method", [MultiTaskELMRegressor, MultiTaskELMClassifier, BatchCholeskySolver, PairwiseRandomProjection])
def test_DenseData_ScikitLearnCompatibility(method):
    check_estimator(method)

