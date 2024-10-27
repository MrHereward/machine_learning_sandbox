import numpy as np
from collections import defaultdict
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

def load_rating_data(data_path, users, movies):
    """
    :param data_path:
    :param users:
    :param movies:
    :return: ratings in array numpy [user, movie]...
    """
    data = np.zeros([users, movies], dtype=np.float32)
    movie_id_mapping = {}
    movie_rating_number = defaultdict(int)
    with open(data_path, 'r') as file:
        for line in file.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split('::')
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = int(rating)
            data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0:
                movie_rating_number[movie_id] += 1
    return data, movie_rating_number, movie_id_mapping

def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Ratings number {int(value)}: {count}')

def main():
    data_path = 'ratings.dat'
    #users = 610
    #movies = 9724
    users = 6040
    movies = 3706
    data, movie_rating_number, movie_id_mapping = load_rating_data(data_path, users, movies)
    display_distribution(data)
    movie_id_most, rating_most_number = sorted(movie_rating_number.items(), key=lambda d: d[1], reverse=True)[0]
    print(f'Movie with ID {movie_id_most} got {rating_most_number} ratings')
    X_raw = np.delete(data, movie_id_mapping[movie_id_most], axis=1)
    Y_raw = data[:, movie_id_mapping[movie_id_most]]
    X = X_raw[Y_raw > 0]
    Y = Y_raw[Y_raw > 0]
    print('Shape X:', X.shape)
    print('Shape Y:', Y.shape)
    display_distribution(Y)
    recommend = 3.0
    Y[Y < recommend] = 0
    Y[Y >= recommend] = 1
    pos_number = (Y == 1).sum()
    neg_number = (Y == 0).sum()
    print(f'Positive probes number: {pos_number}, negative: {neg_number}')
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    # print(len(Y_train), len(Y_test))
    # #scikit learn
    # clf = MultinomialNB(alpha=1.0, fit_prior=True)
    # clf.fit(X_train, Y_train)
    # prediction_prob = clf.predict_proba(X_test)
    # print(prediction_prob[0:10])
    # prediction = clf.predict(X_test)
    # print(prediction)
    # accuracy = clf.score(X_test, Y_test)
    # print(f'Accuracy of model: {accuracy*100:.1f}%')
    # print(confusion_matrix(Y_test, prediction, labels=[0, 1]))
    # print(precision_score(Y_test, prediction, pos_label=1))
    # print(recall_score(Y_test, prediction, pos_label=1))
    # print(f1_score(Y_test, prediction, pos_label=1))
    # print(classification_report(Y_test, prediction))
    # pos_prob = prediction_prob[:, 1]
    # thresholds = np.arange(0.0, 1.1, 0.05)
    # true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
    # #for matplotlib
    # for pred, y in zip(pos_prob, Y_test):
    #     for i, threshold in enumerate(thresholds):
    #         if pred >= threshold:
    #             if y == 1:
    #                 true_pos[i] += 1
    #             else:
    #                 false_pos[i] += 1
    #         else:
    #             break
    # pos_test_number = (Y_test == 1).sum()
    # neg_test_number = (Y_test == 0).sum()
    # true_pos_rate = [tp / pos_test_number for tp in true_pos]
    # false_pos_rate = [fp / neg_test_number for fp in false_pos]
    # print('Area under line:', roc_auc_score(Y_test, pos_prob))
    #cross-check
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
                #something is wrong
                #auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)
                auc_record[alpha][fit_prior] = auc
    print('A prior smoothing Field')
    for smoothing, smoothing_record in auc_record.items():
        for fit_prior, auc in smoothing_record.items():
            print(f'{smoothing}    {fit_prior!s:<6}    {auc}')
    #best from cross-check
    clf = MultinomialNB(alpha=4.0, fit_prior=True)
    clf.fit(X_train, Y_train)
    pos_prob = clf.predict_proba(X_test)[:, 1]
    print('Area under line in best model:', roc_auc_score(Y_test, pos_prob))
    #for matplotlib
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

if __name__ == "__main__":
    sys.exit(main())
