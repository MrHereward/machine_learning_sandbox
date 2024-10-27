from collections import defaultdict

import numpy as np
import sys

from sklearn.naive_bayes import BernoulliNB

X_train = np.array([
    [0, 1, 1],
    [0, 0, 0],
    [1, 1, 1],
    [1, 1, 0]])
Y_train = ['Y', 'N', 'Y', 'Y']
X_test = np.array([[0, 0, 0]])

def get_label_indices(labels):
    """
    :param labels: labels
    :return: dictionary, {class1: [indexes], class2: [indexes]}
    """
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices

def get_prior(label_indices):
    """
    :param label_indices:
    :return: dictionary where key is class label and value is probability a priori
    """
    prior = {label: len(indices) for label, indices in
                                    label_indices.items() }
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior

def get_likelihood(features, label_indices, smoothing=0):
    """
    :param features:
    :param label_indices:
    :param smoothing:
    :return: dictionary where key is class and value is conditional probability vector
    """
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    return likelihood

def get_posterior(X, prior, likelihood):
    """
    :param X:
    :param prior:
    :param likelihood:
    :return: dictionary, key: class label, value: a posteriori probability
    """
    posteriors = []
    for x in X:
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors

def main():
    #own implementation
    label_indices = get_label_indices(Y_train)
    print('label_indices:\n', label_indices)
    prior = get_prior(label_indices)
    print('A priori:\n', prior)
    likelihood = get_likelihood(X_train, label_indices, 1)
    print('Chance:\n', likelihood)
    posterior = get_posterior(X_test, prior, likelihood)
    print('A posteriori:\n', posterior)
    #scikit learn
    clf = BernoulliNB(alpha=1.0, fit_prior=True)
    clf.fit(X_train, Y_train)
    pred_probe = clf.predict_proba(X_test)
    print('[scikit-learn] A posteriori:\n', pred_probe)
    pred = clf.predict(X_test)
    print('[scikit-learn] Prediction:\n', pred)

if __name__ == "__main__":
    sys.exit(main())