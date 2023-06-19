import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from random_banker import RandomBanker
from group4_banker import Group4Banker

# Script extension:
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union

# For simplicity. Could be replaced with something more generic
Banker = Union[Group4Banker, RandomBanker]

def setup_data(data_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    This function takes in the path for a given dataset, and retuns a data frame, and a bundle of categorized datacolumns
    Code originates from https://github.com/olethrosdc/ml-society-science/blob/master/src/project-1/TestLending.py
    but has been adapted to fit the script better.

    Arguments:
    ----------
        data_path: relative or absolute path to the space-separated csv

    Returns:
    --------
        X: a dataframe of the dataset
        feature_data: dict of categorized dataframe columns
    """

    features = [
        'checking account balance', 'duration', 'credit history',
        'purpose', 'amount', 'savings', 'employment', 'installment',
        'marital status', 'other debtors', 'residence time',
        'property', 'age', 'other installments', 'housing', 'credits',
        'job', 'persons', 'phone', 'foreign'
    ]

    target = 'repaid'
    numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'persons', 'credits']

    df = pd.read_csv(data_path, sep=' ', names=features+[target])
    quantitative_features = list(filter(lambda x: x not in numerical_features, features))

    X = pd.get_dummies(df, columns=quantitative_features, drop_first=True)
    encoded_features = list(filter(lambda x: x != target, X.columns))

    encoded_categorical_features = list(filter(lambda x: x not in numerical_features, encoded_features))

    feature_data = {
        "features": features,
        "categorical_features": quantitative_features,
        "encoded_features": encoded_features,
        "encoded_categorical_features": encoded_categorical_features,
        "numerical_features": numerical_features,
        "target": target
    }

    return X, feature_data


def randomize_data(data: Tuple[pd.DataFrame, Dict[str, Any]],
                   probability: float, laplace_delta: float) -> pd.DataFrame:
    """
    randomizes features in the daset for privacy measures

    Arguments:
    ----------
        X: a dataframe of the entire dataset with encoded categorical features
        probability: probability of feature mutation
        laplace_delta: extent of noise to apply

    Returns:
    --------
        X_random: randomized dataset
    """

    X, feature_data = data
    categorical_features = feature_data["encoded_categorical_features"]
    numerical_features = feature_data["numerical_features"]
    X_random = X.copy()

    for column_name in categorical_features:
        temp_col = X_random[column_name]
        random_index = np.random.choice([True,False], temp_col.size,
                                        p=[probability, 1-probability])

        new_datapoints = np.random.choice(np.unique(temp_col), random_index.sum())
        temp_col[random_index] = new_datapoints
        X_random[column_name] = temp_col


    for column_name in numerical_features:
        temp_col = X_random[column_name]
        noise = np.random.laplace(0, laplace_delta, size=temp_col.size)
        noise *= temp_col.std()
        X_random[column_name] = temp_col + noise

    return X_random


def test_decision_maker(X_test: pd.DataFrame, y_test: pd.DataFrame, interest_rate: float, decision_maker: Banker):
    """
    This function tests the utilities and returns of the banker models
    """
    n_test_examples = len(X_test)
    utility = 0

    ## Example test function - this is only an unbiased test if the data has not been seen in training
    total_amount = 0
    total_utility = 0

    for t in range(n_test_examples):
        action = decision_maker.get_best_action(X_test.iloc[t])
        good_loan = y_test.iloc[t] # assume the labels are correct
        duration = X_test['duration'].iloc[t]
        amount = X_test['amount'].iloc[t]
        # If we don't grant the loan then nothing happens
        if action == 1:
            if good_loan != 1:
                utility -= amount
            else:
                utility += amount * (pow(1 + interest_rate, duration) - 1)
        total_utility += utility
        total_amount += amount
    return utility, total_utility/total_amount


def measure_bankers(bankers: List[Banker],
                    data: Tuple[pd.DataFrame, Dict[str, Any]],
                    interest_rate: float,
                    n_tests: int,
                    prints: bool=True,
                    target_data=None) -> Dict[str, Tuple[float, float]]:
    """
    Arguments:
    ----------
        bankers: an iterable of bankers to be measured
        interest: interest rate for the bankers
        n_tests: number of tests to run
        randomized: whether or not to apply randomization for privacy
            default: False
        prints: prints results to terminal
            default: True
        target_data: if target_data is none we assume that data is the target_data

    Returns:
    --------
        results: average utilities and investment returns
    """
    X, feature_data = data
    encoded_features = feature_data["encoded_features"]
    target = feature_data["target"]

    results = {}
    duplicate = {}

    for decision_maker in bankers:
        banker_name = type(decision_maker).__name__

        utility = 0
        investment_return = 0

        if prints: print("\nTesting on class:", banker_name, "...")

        range_iterator = range(n_tests)
        if prints: range_iterator = tqdm(range_iterator)

        for i in range_iterator:
            if target_data is not None:
                X_train, _, _, _ = train_test_split(X[encoded_features], X[target], test_size=0.2)
                _, X_test, y_train, y_test = train_test_split(target_data[encoded_features], target_data[target], test_size=0.2)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)

            decision_maker.set_interest_rate(interest_rate)
            decision_maker.fit(X_train, y_train)
            Ui, Ri = test_decision_maker(X_test, y_test, interest_rate, decision_maker)
            utility += Ui
            investment_return += Ri

        if prints:
            print("Results:")
            print("\tAverage utility:", utility / n_tests)
            print("\tAverage return on investment:", investment_return / n_tests)

        if banker_name in duplicate:
            banker_name += str(duplicate[banker_name])
            duplicate[banker_name] += 1
        else:
            duplicate[banker_name] = 1

        results[banker_name] = (utility/n_tests, investment_return/n_tests)


    return results


def parse_args():
    ap = ArgumentParser()

    ap.add_argument("data_path", nargs="?", default='../../data/credit/D_valid.csv',
                    help="Path to the space-separated csv file containg the data (Warning: this program uses relative path, so make sure to be in the same directory as test_lending.py)")
    ap.add_argument("--n-tests", type=int, default=10)
    ap.add_argument("-r", "--interest-rate", type=float, default=0.017)
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--randomized", nargs=2, type=float)
    ap.add_argument("--random_banker", action="store_true")

    return ap.parse_args()


def main():

    args = parse_args()

    np.random.seed(args.seed)
    print("Running with random seed: ", args.seed)

    interest_rate = args.interest_rate
    n_tests = args.n_tests

    X, features = data = setup_data(args.data_path)

    print(f"r={interest_rate}, n_tests={n_tests}, seed={args.seed}")

    decision_makers = [Group4Banker(optimize=args.optimize, random_state=args.seed)]

    if args.random_banker: decision_makers.insert(0, RandomBanker())
    if args.randomized:
        # Suppress warning in randomize data
        pd.options.mode.chained_assignment = None

        probability, laplace_delta = args.randomized
        print(f"probability: {probability}, laplace delta: {laplace_delta}")
        X_random = randomize_data(data, probability, laplace_delta)
        data_random = X_random, features

    results = measure_bankers(decision_makers, data, interest_rate, n_tests=args.n_tests)

    if args.randomized:
        results_random = measure_bankers(decision_makers, data_random, interest_rate, n_tests=args.n_tests, target_data=X_random)

        print("\nPrivacy cost (avg_return_normal - avg_return_privacy)")
        print(results["Group4Banker"][1] - results_random["Group4Banker"][1])

if __name__ == "__main__":
    main()
