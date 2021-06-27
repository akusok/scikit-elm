import unittest
from sklearn.datasets import load_iris

from skelm import ELMClassifier, ELMRegressor, LanczosELM
from skelm.elm import ScikitELM

ELMs = (ScikitELM, ELMRegressor, ELMClassifier, LanczosELM)
BatchELMs = (ScikitELM, ELMRegressor, ELMClassifier)


class TestInterface(unittest.TestCase):

    def setUp(self) -> None:
        self.X, self.y = load_iris(return_X_y=True)

    def test_SLFNs(self):
        for ELM_model in ELMs:
            elm = ELM_model().fit(self.X, self.y)
            SLFNs_ = elm.SLFNs_
            self.assertIsNotNone(SLFNs_)

    def test_Solver(self):
        for ELM_model in ELMs:
            elm = ELM_model().fit(self.X, self.y)
            solver_ = elm.solver_
            self.assertIsNotNone(solver_)

    def test_Solver_CoefAndIntercept(self):
        for ELM_model in ELMs:
            elm = ELM_model().fit(self.X, self.y)
            coef_ = elm.solver_.coef_
            intercept_ = elm.solver_.intercept_
            self.assertIsNotNone(coef_)
            self.assertIsNotNone(intercept_)

    def test_BatchSolver_XtX_XtY(self):
        for ELM_model in BatchELMs:
            batch_elm = ELM_model().fit(self.X, self.y)
            XtX_ = batch_elm.solver_.XtX_
            XtY_ = batch_elm.solver_.XtY_
            self.assertIsNotNone(XtX_)
            self.assertIsNotNone(XtY_)
