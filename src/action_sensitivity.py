from tqdm import tqdm

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from group4_banker import Group4Banker

from plot_config import setup, set_fig_size, set_arrowed_spines

setup()


def prep_data():

    features = ['checking account balance', 'duration', 'credit history',
                'purpose', 'amount', 'savings', 'employment', 'installment',
                'marital status', 'other debtors', 'residence time',
                'property', 'age', 'other installments', 'housing', 'credits',
                'job', 'persons', 'phone', 'foreign']

    target = 'repaid'

    df_train = pd.read_csv("../../data/credit/D_train.csv", sep=' ', names=features+[target])
    df_test = pd.read_csv("../../data/credit/D_test.csv", sep=' ', names=features+[target])

    numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'persons', 'credits']
    quantitative_features = list(filter(lambda x: x not in numerical_features, features))
    D_train = pd.get_dummies(df_train, columns=quantitative_features, drop_first=True)
    D_test = pd.get_dummies(df_test, columns=quantitative_features, drop_first=True)
    encoded_features = list(filter(lambda x: x != target, D_train.columns))

    return D_train, D_test, encoded_features, target


def oob_sampling(model, X, y, n_samples, size_sample, seed):

    np.random.seed(seed)

    Y_true = np.zeros((n_samples, size_sample))
    Y_pred = np.zeros((n_samples, size_sample))

    for i in tqdm(range(n_samples)):

        test_idx = np.random.choice(np.arange(X.shape[0]), replace=True, size=size_sample)
        
        y_test = y.iloc[test_idx]
        X_test = X.iloc[test_idx]

        Y_true[i] = y_test.values - 1
        Y_pred[i] = [int(model.get_best_action(x)) for _, x in X_test.iterrows()]

    return Y_true, Y_pred


def posterior_hypothesis(X, p):
    
    n_samples, n_trails = np.shape(X)
    
    posterior = np.zeros_like(X, dtype=float)
    for i in range(n_trails):
        
        # Parameters for Beta prior.
        alpha = 1
        beta = 1
        
        log_p, log_p_marginal = 0, 0
        for j in range(n_samples):
            
            x = X[j, i]
            p_x = (x * alpha + (1 - x) * beta) / (alpha + beta)
            
            alpha = alpha + x
            beta = beta + 1 - x
            
            # Prior for the null hypothesis being true.
            log_p = log_p + np.log(p)
            log_p_marginal = log_p_marginal + np.log(p_x)

            posterior[j, i] = np.exp(log_p - np.log(np.exp(log_p_marginal) + np.exp(log_p)))
            
    return posterior


def plot_posteriors(Y_true, Y_pred):

    delta = 0.05

    # H0:
    p_not_grant = 1 - np.mean(Y_true)
    p_not_grant_low = p_not_grant - np.sqrt(np.log(2) * delta / (2 * Y_true.shape[0]))
    p_not_grant_high = p_not_grant + np.sqrt(np.log(2) * delta / (2 * Y_true.shape[0]))

    # Posterior hypothesis of H0.
    post_grant_true = posterior_hypothesis(Y_true, p=p_not_grant)
    post_grant_pred = posterior_hypothesis(Y_pred, p=p_not_grant)
    post_grant_pred_low = posterior_hypothesis(Y_pred, p=p_not_grant_low)
    post_grant_pred_high = posterior_hypothesis(Y_pred, p=p_not_grant_high)

    fig, ax = plt.subplots(1, 1, figsize=set_fig_size(500, fraction=1))
    ax.set_title(r"Posterior probability of accepting $H_0$")
    ax.plot(np.mean(post_grant_true, axis=1), label="Test data", c="darkorange", alpha=0.7)
    ax.plot(np.mean(post_grant_pred, axis=1), label="Model predictions", alpha=0.7)
    ax.fill_between(np.arange(Y_true.shape[0]), 
                    np.mean(post_grant_pred_low, axis=1), 
                    np.mean(post_grant_pred_high, axis=1), alpha=0.2)
    ax.legend()
    ax.set_ylabel("Posterior probability")
    ax.set_xlabel("Number of bootstrap samples")
    set_arrowed_spines(fig, ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig("posterior_null_hypo.pdf")
    
    bayes_reject_grant_true = np.mean(post_grant_true < p_not_grant, axis=1)
    bayes_reject_grant_pred = np.mean(post_grant_pred < p_not_grant, axis=1)
    bayes_reject_grant_pred_low = np.mean(post_grant_pred_low < p_not_grant, axis=1)
    bayes_reject_grant_pred_high = np.mean(post_grant_pred_high < p_not_grant, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=set_fig_size(500, fraction=1))
    ax.set_title(r"Posterior probability of rejecting $H_0$")
    ax.plot(bayes_reject_grant_true, label="Test data", c="darkorange", alpha=0.7)
    ax.plot(bayes_reject_grant_pred, label="Model predictions", alpha=0.7)
    ax.fill_between(np.arange(Y_true.shape[0]), 
                    bayes_reject_grant_pred_low, 
                    bayes_reject_grant_pred_high, alpha=0.2)
    ax.legend()
    ax.set_ylabel("Posterior probability")
    ax.set_xlabel("Number of bootstrap samples")
    set_arrowed_spines(fig, ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig("reject_null_hypo.pdf")
    

def main():
    
    D_train, D_test, encoded_features, target = prep_data()

    X_train = D_train.loc[:, encoded_features] 
    y_train = D_train.loc[:, target] 

    model = Group4Banker(optimize=False, random_state=42)
    model.set_interest_rate(0.05)
    model.fit(X_train, y_train)

    X_test = D_test.loc[:, encoded_features] 
    y_test = D_test.loc[:, target] 
    
    Y_true, Y_pred = oob_sampling(model, X_test, y_test, 500, 50, 42)
    
    np.save("Y_true.npy", Y_true)
    np.save("Y_pred.npy", Y_pred)
    
    Y_true = np.load("Y_true.npy") 
    Y_pred = np.load("Y_pred.npy")

    plot_posteriors(Y_true, Y_pred)


if __name__ == "__main__":
    main()
