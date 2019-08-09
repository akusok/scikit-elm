import pytest

from sklearn.utils.estimator_checks import check_estimator

from skelm import TemplateEstimator
from skelm import TemplateClassifier
from skelm import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
