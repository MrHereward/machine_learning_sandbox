import timeit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier


def sigmoid(input):
    return 1.0 / (1 + np.exp(-input))

# z = np.linspace(-8, 8, 100)
# y = sigmoid(z)
# plt.plot(z, y)
# plt.axhline(y=0, ls='dotted', color='k')
# plt.axhline(y=0.5, ls='dotted', color='k')
# plt.axhline(y=1, ls='dotted', color='k')
# plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
# plt.xlabel('z')
# plt.ylabel('y(z)')
# plt.show()
# y_hat = np.linspace(0, 1, 1000)
# cost = -np.log(y_hat)
# plt.plot(y_hat, cost)
# plt.xlabel('Prognoza')
# plt.ylabel('Koszt')
# plt.xlim(0, 1)
# plt.ylim(0, 7)
# plt.show()
# y_hat = np.linspace(0, 1, 1000)
# cost = -np.log(1 - y_hat)
# plt.plot(y_hat, cost)
# plt.xlabel('Prognoza')
# plt.ylabel('Koszt')
# plt.xlim(0, 1)
# plt.ylim(0, 7)
# plt.show()

def compute_prediction(X, weights):
    """
    Funkcja wyliczajaca prognoze y_hat z wykorzystaniem biezacych wag
    :param X:
    :param weights:
    :return:
    """
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    return predictions

def update_weights_gd(X_train, Y_train, weights, learning_rate):
    """
    Funkcja modyfikujaca wagi w biezacym kroku
    :param X_train:
    :param Y_train:
    :param weights:
    :param learning_rate:
    :return:
    """
    predictions = compute_prediction(X_train, weights)
    weights_delta = np.dot(X_train.T, Y_train - predictions)
    m = Y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights

def update_weights_sgd(X_train, Y_train, weights, learning_rate):
    """
    Pojedyncza iteracja modyfikujaca wagi: jeden krok na bazie pojedynczej probki
    :param X_train:
    :param Y_train:
    :param weights:
    :param learning_rate:
    :return:
    """
    for X_each, Y_each in zip(X_train, Y_train):
        prediction = compute_prediction(X_each, weights)
        weights_delta = X_each.T * (Y_each - prediction)
        weights += learning_rate * weights_delta
    return weights

def compute_cost(X, Y, weights):
    """
    Funkcja wyliczajaca wagi J(w)
    :param X:
    :param Y:
    :param weights:
    :return:
    """
    predictions = compute_prediction(X, weights)
    cost = np.mean(-Y * np.log(predictions) - (1 - Y) * np.log(1 - predictions))
    return cost

def train_logistic_regression(X_train, Y_train, max_iter, learning_rate, fit_intercept=False):
    """
    Funkcja trenujaca model regresji logicznej
    :param X_train:
    :param Y_train:
    :param max_iter:
    :param learning_rate:
    :param fit_intercept: flaga z przechwyceniem w0 czy bez
    :return:
    """
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_gd(X_train, Y_train, weights, learning_rate)
        if iteration % 100 == 0:
            print(compute_cost(X_train, Y_train, weights))
    return weights

def train_logistic_regression_sgd(X_train, Y_train, max_iter, learning_rate, fit_intercept=False):
    """
    Trening modelu wykorzustujacy stochastyczny gradient prosty
    :param X_train:
    :param Y_train:
    :param max_iter:
    :param learning_rate:
    :param fit_intercept:
    :return:
    """
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_sgd(X_train, Y_train, weights, learning_rate)
        if iteration % 2 == 0:
            print(compute_cost(X_train, Y_train, weights))
    return weights

def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
    return compute_prediction(X, weights)

# X_train = np.array([[6, 7],
#                     [2, 4],
#                     [3, 6],
#                     [4, 7],
#                     [1, 6],
#                     [5, 2],
#                     [2, 0],
#                     [6, 3],
#                     [4, 1],
#                     [7, 2]])
# Y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# weights = train_logistic_regression(X_train, Y_train, max_iter=1000, learning_rate=0.1, fit_intercept=True)
# X_test = np.array([[6, 1],
#                    [1, 3],
#                    [3, 1],
#                    [4, 5]])
# predictions = predict(X_test, weights)
# print(predictions)
#
# plt.scatter(X_train[:, 0], X_train[:, 1], c=['b']*5+['k']*5, marker='o')
# colours = ['k' if prediction >= 0.5 else 'b' for prediction in predictions]
# plt.scatter(X_test[:, 0], X_test[:, 1], marker='*', c=colours)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.show()

rows = 300000
df = pd.read_csv('train.csv', nrows=rows)
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values
train_number = 10000
X_train = X[:train_number]
Y_train = Y[:train_number]
X_test = X[train_number:train_number * 2]
Y_test = Y[train_number:train_number * 2]
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)
start_time = timeit.default_timer()
#weights = train_logistic_regression(X_train_enc.toarray(), Y_train, max_iter=1000, learning_rate=0.01, fit_intercept=True)
weights = train_logistic_regression_sgd(X_train_enc.toarray(), Y_train, max_iter=10, learning_rate=0.01, fit_intercept=True)
print(f'--- {(timeit.default_timer() - start_time)}.3f s ---')
pred = predict(X_test_enc.toarray(), weights)
print(f'Liczba probek treningowych: {train_number}, pole pod krzywa ROC dla zbioru treningowego: {roc_auc_score(Y_test, pred):.3f}')
#scikit-learn
sgd_lr = SGDClassifier(loss='log_loss', penalty=None, fit_intercept=True, max_iter=10, learning_rate='constant', eta0=0.01)
sgd_lr.fit(X_train_enc.toarray(), Y_train)
pred = sgd_lr.predict_proba(X_test_enc.toarray())[:, 1]
print(f'Liczba probek treningowych: {train_number}, pole pod krzywa ROC dla zbioru treningowego: {roc_auc_score(Y_test, pred):.3f}')
#l1
sgd_lr_l1 = SGDClassifier(loss='log_loss', penalty='l1', alpha=0.0001, fit_intercept=True, max_iter=10, learning_rate='constant', eta0=0.01)
sgd_lr_l1.fit(X_train_enc.toarray(), Y_train)
coef_abs = np.abs(sgd_lr_l1.coef_)
print(np.sort(coef_abs)[0][:10])
bottom_10 = np.argsort(coef_abs)[0][:10]
feature_names = enc.get_feature_names_out()
print('10 najmniej istotnych cech:\n', feature_names[bottom_10])
print(np.sort(coef_abs)[0][-10:])
top_10 = np.argsort(coef_abs)[0][-10:]
print('10 najbardziej istotnych cech:\n', feature_names[top_10])