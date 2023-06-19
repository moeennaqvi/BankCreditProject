import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class Group4Banker:
    """
    A decision rule for giving loans to individuals.
    Policy: select action which maximises utility


    Arguments:
    ----------
        optimize: whether to optimize the classifier's hyperparameters using 5-fold cv grid search
            default: false
        random_state: seed for the pseduo random number generator (of the classifier)
            default: 1234
    """


    def __init__(self, optimize: bool=False, random_state: int=1234):
        self.optimize = optimize
        self.random_state = random_state


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fits a model for calculating the probability of credit- worthiness

        Arguments:
        ----------
            X: Feature set of individuals
            y: Target labels against the individuals
        """

        if self.optimize:
            #Finding optimal paramters
            param_grid = [{
                'bootstrap' : [True],
                'max_features' : list(range(10,20,1)),
                'max_depth' : list(range(10,100,10)),
                'n_estimators' : list(range(25,150,25))
            }]

            grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
            grid_search.fit(X, y)
            self.classifier = RandomForestClassifier(random_state=self.random_state, **grid_search.best_params_)

        else:
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight="balanced"
            )

        # NOTE:
        # classes = (1, 2)
        self.classifier.fit(X,y)


    def set_interest_rate(self, rate: float) -> None:
        self.rate = rate


    def predict_proba(self, x: pd.Series) -> float:
        """Returns the probability that a person will return the loan.

        Arguments:
        ----------
            x: Features describing a selected individual

        Returns:
        --------
            probability:
                real number between 0 and 1 denoting probability of the individual to be credit-worthy

        """

        if not hasattr(self, "classifier"):
            raise ValueError("This Group4Banker instance is not fitted yet. Call 'fit' "
                             "with appropriate arguments before using this method.")

        x_reshaped = np.reshape(x.to_numpy(), (1,-1))

        return self.classifier.predict_proba(x_reshaped)[0][0]


    def expected_utility(self, x: pd.Series, action: int) -> float:
        """Calculate the expected utility of a particular action for a given individual.

        Arguments:
        ----------
            x: Features describing a selected individual
            action: whether or not to give loan.

        Returns:
        --------
            expected_utility:
                the result of our given utility formula m[(1+r)^n] multiplied by probability of loan return
        """

        if action:
            # Probability of being credit worthy.
            pi = self.predict_proba(x)
            return x["amount"] * ((1 + self.rate) ** x["duration"] - 1) * pi - x["amount"] * (1 - pi)

        return 0.0


    def get_best_action(self, x: pd.Series) -> int:
        """Returns the action maximising expected utility.

        Arguemnts:
        ----------
            x: Feature "vector" describing a selected individual
        Returns:
        --------
            action: 0 or 1 regarding wether or not to give loan
        """
        return int(self.expected_utility(x, 1) > 0)
