from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from ucimlrepo import fetch_ucirepo, list_available_datasets

import numpy as np
import matplotlib.pyplot as plt

# check which datasets can be imported
# list_available_datasets()
# import dataset
heart_disease = fetch_ucirepo(id=45)
# alternatively: fetch_ucirepo(name='Heart Disease')
# access data
raw_X = heart_disease.data.features
raw_Y = heart_disease.data.targets

X = raw_X.to_numpy()
Y = raw_Y.to_numpy().flatten()
X[np.isnan(X)] = 0
Y[np.isnan(Y)] = 0
#lack of info in documentation
Y[Y > 0] = 1
k = 10
k_fold = StratifiedKFold(n_splits=k)
smoothing_factor_option = [1, 2, 3, 4, 5, 6]
fit_prior_option = [True, False]
auc_record = {}
for train_indices, test_indices in k_fold.split(X, Y):
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    for alpha in smoothing_factor_option:
        if alpha not in auc_record:
            auc_record[alpha] = {}
        for fit_prior in fit_prior_option:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(X_train, Y_train)
            prediction_prob = clf.predict_proba(X_test)
            pos_prob = prediction_prob[:, 1]
            auc = roc_auc_score(Y_test, pos_prob)
            # something is wrong
            # auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)
            auc_record[alpha][fit_prior] = auc
print('A prior smoothing Field')
for smoothing, smoothing_record in auc_record.items():
    for fit_prior, auc in smoothing_record.items():
        print(f'{smoothing}    {fit_prior!s:<6}    {auc}')

clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)
pos_prob = clf.predict_proba(X_test)[:, 1]
print('Area under line in best model:', roc_auc_score(Y_test, pos_prob))

thresholds = np.arange(0.0, 1.1, 0.05)
true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
            if y == 1:
                true_pos[i] += 1
            else:
                false_pos[i] += 1
        else:
            break
pos_test_number = (Y_test == 1).sum()
neg_test_number = (Y_test == 0).sum()
true_pos_rate = [tp / pos_test_number for tp in true_pos]
false_pos_rate = [fp / neg_test_number for fp in false_pos]
#matplotlib
plt.figure()
lw = 2
plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=lw)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Percentage of false positive cases')
plt.ylabel('Percentage of true positive cases')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()