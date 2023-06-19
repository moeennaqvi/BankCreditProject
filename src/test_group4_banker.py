import unittest
from unittest.mock import MagicMock
from sklearn.utils.validation import check_is_fitted
from group4_banker import Group4Banker
import pandas as pd
import numpy as np

class TestGroup4Banker(unittest.TestCase):
    """
    Testcase for Group4Banker containing uninttests for all its methods

    fields:
        r: default sample interest rate
        never_returns: instance of Group4Banker that always assumes loan will not be returned
        always_returns: instance of Group4Banker that always assumes loan will be returned

    """
    r = 0.05

    # Agent setup
    never_returns, always_returns = Group4Banker(), Group4Banker()
    for i, a in (0, never_returns), (1, always_returns):
            a.set_interest_rate(r)
            a.predict_proba = MagicMock(return_value=i)


    def test_set_interest_rate(self):
        banker = Group4Banker()
        assert not hasattr(banker, "rate")
        banker.set_interest_rate(self.r)
        assert hasattr(banker, "rate")
        assert banker.rate == self.r


    def test_predict_proba(self):
        banker = Group4Banker()
        banker.fit([[100, 0],
                    [0, 100]], [1, 2])
        p1 = banker.predict_proba(pd.Series([100, 0]))
        assert p1 >= 0.5
        p2 = banker.predict_proba(pd.Series([0, 100]))
        assert p2 <= 0.5

    def test_fit(self):
        decision_maker = Group4Banker()
        decision_maker.fit([[0, 0], [0, 0]], [0, 0])
        assert "classifier" in decision_maker.__dict__, \
            "Group4Banker should have attribute 'classifier' after calling fit"

        check_is_fitted(decision_maker.classifier, "estimators_")

    def test_expected_utility(self):
        x = pd.Series({"duration": 10, "amount": 100})
        assert self.never_returns.expected_utility(x, 1) < 0, \
            "Utility must be negative if person does not return loan"
        assert self.always_returns.expected_utility(x, 1) > 0, \
            "Utility must be positive if person does not return loan"

        proba = 0.7
        decision_maker = Group4Banker()
        decision_maker.set_interest_rate(self.r)
        decision_maker.predict_proba = MagicMock(return_value=proba)

        estimate = decision_maker.expected_utility(x, 1)
        ground_truth = 14.022623874420933
        assert np.isclose(estimate, ground_truth), \
            f"Estimate should be close to {ground_truth}, was {estimate}"


    def test_get_best_action(self):
        for d in range(1, 1000, 10):
            for amt in range(1, 50000, 1500):
                x = pd.Series({"duration": d, "amount": amt})
                assert not self.never_returns.get_best_action(x), \
                    "When probability of return is 0, best action should always be 0"
                assert self.always_returns.get_best_action(x), \
                    "When probability of return is 1, best action should always be 1"
