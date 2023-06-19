import numpy as np
import pandas as pd
from group4_banker import Group4Banker
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from test_lending import setup_data
from functools import partial

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("optinal package seaborn not found: proceeding with default theme")

feature_description = {
    "marital status_1":"divorced/separated male",
    "marital status_2":"divorced/separated/married female",
    "marital status_3":"single male",
    "marital status_4":"married/widowed male",
    "marital status_5":"single female"
}

description_feature = {v:k for k, v in feature_description.items()}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n-tests", type=int, default=3)
    parser.add_argument("-r", "--interest-rate", type=float, default=0.05)
    parser.add_argument("-s", "--seed", type=int, default=42)

    return parser.parse_args()


def get_returns_on_feature(feature, data, target, feature_value=1):
    data = data[data[feature] == 1]
    positives, negatives = data[target] == 1, data[target] == 2
    return positives.sum(), negatives.sum()


def get_trained_model(interest_rate):
    X_train, feature_data = setup_data("../../data/credit/D_train.csv")
    decision_maker = Group4Banker()
    decision_maker.set_interest_rate(interest_rate)
    decision_maker.fit(X_train[feature_data["encoded_features"]],
                       X_train[feature_data["target"]])
    return decision_maker

def get_preprocessed_german_data():
    features = ['checking account balance', 'duration', 'credit history',
                'purpose', 'amount', 'savings', 'employment', 'installment',
                'marital status', 'other debtors', 'residence time',
                'property', 'age', 'other installments', 'housing', 'credits',
                'job', 'persons', 'phone', 'foreign']
    target = ['repaid']
    dataframe = pd.read_csv('../../data/credit/german.data', sep=' ',
                         names=features+target)
    numeric_features = dataframe[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [f for f in features if f not in numeric_features]
    dataframe['gender'] = dataframe.apply(lambda row: 'Female' if row['marital status'] in ('A92', 'A95') else 'Male', axis=1)

    dataframe = pd.get_dummies(dataframe, columns=categorical_features, drop_first=True)
    features = dataframe.drop(target, axis=1).select_dtypes(include=[np.number]).columns.tolist()

    return dataframe, features, target


def get_gender_outcome_action_from_german_data():
    pd.options.mode.chained_assignment = None
    dataset, feature_data = setup_data("../../data/credit/D_train.csv")
    dataframe, features, target = get_preprocessed_german_data()
    from sklearn.model_selection import KFold
    dataframe['action'] = pd.Series(np.zeros(dataframe.shape[0]))
    decision_maker = Group4Banker()
    decision_maker.set_interest_rate(0.05)
    for train, test in KFold(n_splits=5).split(dataframe):
        decision_maker.fit(dataset[feature_data["encoded_features"]], dataset[feature_data["target"]])
        for k in test:
            dataframe['action'].iloc[k] = decision_maker.get_best_action(dataset[feature_data["encoded_features"]].iloc[k])

    Z = np.array([1 if z=='Female' else -1 for z in dataframe['gender'].values])
    Y = np.array([1 if y==1 else -1 for y in dataframe['repaid'].values])
    A = np.array([1 if a==1 else -1 for a in dataframe['action'].values])

    return Z, Y, A


def marginal_probability(data, alpha, beta):
        total_probability = 1
        log_probability = 0
        for t in range(len(data)):
            p = alpha / (alpha + beta)
            if (data[t] > 0):
                log_probability += np.log(p)
                alpha += 1
            else:
                log_probability += np.log(1 - p)
                beta +=1
        return np.exp(log_probability)


def conditional_independence(A,Y,Z):
    P_D_mu0 = 0
    P_D_mu1 = 0

    table = np.zeros((2, 7))

    for i, y in enumerate([-1, 1]):
        table[i][0] = y
        ## P(A | Y, Z = 1)
        positive = (Y==y) & (Z==1)
        positive_alpha = sum(A[positive]==1)
        positive_beta = sum(A[positive]==-1)
        positive_ratio = positive_alpha / (positive_alpha + positive_beta)

        ## P(A | Y, Z = - 1)
        negative = (Y==y) & (Z==-1)
        negative_alpha = sum(A[negative]==1)
        negative_beta = sum(A[negative]==-1)
        negative_ratio = negative_alpha / (negative_alpha + negative_beta)

        diff = abs(positive_ratio - negative_ratio)
        table[i][1] = diff
        #  print("y: ", y, "Difference: ", diff)

        ##Calculate the marginals for each model
        P_D_positive = marginal_probability(A[positive], 1, 1)
        P_D_negative = marginal_probability(A[negative], 1, 1)
        P_D = marginal_probability(A[(Y==y)], 1, 1)

        #         print("Marginal likelihoods: ", P_D, P_D_negative, P_D_positive)
        table[i][2] = P_D
        table[i][3] = P_D_negative
        table[i][4] = P_D_positive

        ##combining all the marginal probabilities
        P_D_mu0 += np.log(P_D)
        P_D_mu1 += np.log(P_D_positive) + np.log(P_D_negative)
        #  print(f"P_D_mu0={P_D_mu0},P_D_mu1 ={P_D_mu1}")
        table[i][5] = P_D_mu0
        table[i][6] = P_D_mu1

    return pd.DataFrame(dict(zip("y, diff, P_D, P_D_negative, P_D_positive, P_D_mu0, P_D_mu1".split(", "), table.T)))


def get_encoded_features():
    return setup_data("../../data/credit/D_train.csv")[1]["encoded_features"]


def measure_probability_difference(args):
    np.random.seed(args.seed)

    single_male = description_feature["single male"]
    single_female = description_feature["single female"]

    X_train, feature_data = setup_data("../../data/credit/D_train.csv")
    X_val, *_ = setup_data("../../data/credit/D_valid.csv")

    encoded_features = feature_data["encoded_features"]
    target = feature_data["target"]

    decision_maker = Group4Banker()
    decision_maker.set_interest_rate(args.interest_rate)
    decision_maker.fit(X_train[encoded_features], X_train[target])

    df_single_male = (X_train[single_male] == 1)
    single_male_and_return = (X_train[single_male] == 1) & (X_train[target] == 1)

    df_single_female = (X_train[single_female] == 1)
    single_female_and_return = (X_train[single_female] == 1) & (X_train[target] == 1)

    samples = X_val.sample(n=args.n_tests)
    n_tests = args.n_tests

    avg_diff = 0
    max_diff_male = 0
    max_diff_female = 0

    for i, row in samples.iterrows():
        for i in range(1, 6):
            row["martial status_"+str(i)] = 0

        row[single_male] = 1

        proba_on_m = decision_maker.predict_proba(row[encoded_features])
        utility_m = decision_maker.expected_utility(row[encoded_features], 1)

        row[single_male] = 0
        row[single_female] = 1

        proba_on_f = decision_maker.predict_proba(row[encoded_features])
        utility_f = decision_maker.expected_utility(row[encoded_features], 1)
        row[single_female] = 0

        diff = proba_on_m - proba_on_f
        if diff > max_diff_male: max_diff_male = diff
        if diff < max_diff_female: max_diff_female = diff
        absdiff = abs(diff)
        if n_tests < 10:
            print("Estimated probability for single male:", proba_on_m,
                  "\nEstimated probabiltiy for single female:", proba_on_f,
                  "\nAbsolute difference", absdiff, "\n")

    if n_tests >= 10:
        print("Average probability difference:", absdiff/n_tests)
        print("max diff benefitting female:", -max_diff_female)
        print("max diff benefitting male:", max_diff_male)

def create_histogram(args):
    dataset, feature_data = setup_data("../../data/credit/D_train.csv")
    encoded_features, target = feature_data["encoded_features"], feature_data["target"]
    X = dataset[encoded_features]
    y = dataset[target]
    decision_maker = Group4Banker()
    decision_maker.set_interest_rate(0.05)
    decision_maker.fit(X, y)
    # plt.subplots(1, 2)

    # dataset, feature_data = setup_data("../../data/credit/D_train.csv")
    # ax_1 = generate_barchart(decision_maker, dataset, feature_data, single_male)
    # plt.show()
    # ax_2 = generate_barchart(decision_maker, dataset, feature_data, single_female)
    # plt.show()

    # genereate_histogram_outcome(decision_maker, dataset, feature_data, single_male)
    # plt.show()
    generate_histogram_utility(decision_maker, dataset, feature_data, single_female)
    plt.show()


def generate_barchart(decision_maker, dataset, feature_data, feature, target_value=1, plot_title=None):
    encoded_features, target = feature_data["encoded_features"], feature_data["target"]
    dataset = dataset[dataset[feature] == target_value]

    X = dataset[encoded_features]
    y = dataset[target]
    probs = decision_maker.classifier.predict_proba(X).T[0]

    ax = plt.axis()
    plt.bar(X[y == 1]["amount"], probs[y == 1], width=2000, color="blue", label="returned loan")
    plt.bar(X[y == 2]["amount"], probs[y == 2], width=2000, color="orange", label="did not return loan")
    plt.xlabel("loan amount")
    plt.ylabel("predicted probability")
    plt.title(f"predicted probability considering loan for {feature}")
    return ax


def generate_histogram_outcome(decision_maker, dataset, feature_data, feature, target_value=1, ax=None, title=None):
    encoded_features, target = feature_data["encoded_features"], feature_data["target"]
    dataset = dataset[dataset[feature] == target_value]

    X = dataset[encoded_features]
    y = dataset[target]
    if ax:
        ax.hist(X[y == 1]["amount"], 20, label="returned loan", alpha=.5, color="blue")
        ax.hist(X[y == 2]["amount"], 20, label="did not return loan", alpha=.8, color="orange")

        ax.set_xlabel("amount of loan")
        ax.set_ylabel("amount of applicants")
        ax.legend()
        if title:
            ax.set_title(title)
    else:
        plt.hist(X[y == 1]["amount"], 20, label="returned loan", alpha=.5, color="blue")
        plt.hist(X[y == 2]["amount"], 20, label="did not return loan", alpha=.8, color="orange")
        plt.set_xlabel("amount of loan")
        plt.set_ylabel("amount of applicants")
        plt.legend()


def generate_histogram_action(decision_maker, dataset, feature_data, feature, target_value=1, ax=None, title=None, bottom_label="amount of loan"):
    encoded_features, target = feature_data["encoded_features"], feature_data["target"]
    dataset = dataset[dataset[feature] == target_value]

    X = dataset[encoded_features]

    y_hat = X.apply(decision_maker.get_best_action, axis=1)



    if ax:
        ax.hist(X[y_hat == 1]["amount"], 20, label="granted loan", alpha=.5, color="blue")
        ax.hist(X[y_hat == 0]["amount"], 20, label="refused loan", alpha=.8, color="orange")

        ax.set_xlabel(bottom_label)
        ax.legend()
        ax.set_ylabel("amount of applicants")

        if title:
            ax.set_title(title)

    else:
        plt.hist(X[y_hat == 1]["amount"], 20, label="granted loan", alpha=.5, color="blue")
        plt.hist(X[y_hat == 0]["amount"], 20, label="refused loan", alpha=.8, color="orange")

        plt.xlabel(bottom_label)
        plt.ylabel("amount of applicants")
        plt.legend()


def generate_histogram_utility(decision_maker, dataset, feature_data, feature, title=None, target_value=1, ax=None):
    encoded_features, target = feature_data["encoded_features"], feature_data["target"]
    dataset = dataset[dataset[feature] == target_value]

    X = dataset[encoded_features]
    U = X.apply(partial(decision_maker.expected_utility, action=1), axis=1)
    U = 1/(1+np.exp(-U.to_numpy()))
    y = dataset[target]


    if ax:
        ax.hist(U[y==1], 20, label="granted loan", alpha=.5, color="blue")
        ax.hist(U[y==2], 20, label="refused loan", alpha=.8, color="orange")

        ax.set_xlabel("expected_utility")
        ax.set_ylabel("amount of applicants")
        ax.legend()

        if title:
            ax.set_title(title)
    else:
        plt.hist(U[y==1], 20, label="granted loan", alpha=.5, color="blue")
        plt.hist(U[y==2], 20, label="refused loan", alpha=.8, color="orange")

        plt.xlabel("amount of loan")
        plt.ylabel("amount of applicants")
        plt.legend()


def credit_worthiness_barchart(Y1, Y0, labels, label1='returned loan',
                               label2="no returned loan", legend_anchor=0, **subplot_kwargs):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(**subplot_kwargs)
    rects1 = ax.bar(x - width/2, Y1, width, label=label1, alpha=0.8, color="blue")
    rects2 = ax.bar(x + width/2, Y0, width, label=label2, alpha=0.8, color="orange")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Loan applicants')
    ax.set_title('Distribution of returns')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc=legend_anchor)


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()


def main():
    args = parse_args()
    measure_probability_difference(args)

    # create_histogram(args)

if __name__ == "__main__":
    main()
