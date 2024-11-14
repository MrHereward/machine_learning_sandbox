import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


def add_original_feature(df, df_new):
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)

def add_avg_price(df,df_new):
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']

def add_avg_volume(df,df_new):
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']

def add_std_price(df, df_new):
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)
    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']

def add_std_volume(df, df_new):
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']

def add_return_feature(df, df_new):
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)

def generate_features(df, df_new):
    add_original_feature(df, df_new)
    add_avg_price(df, df_new)
    add_avg_volume(df, df_new)
    add_std_price(df, df_new)
    add_std_volume(df, df_new)
    add_return_feature(df, df_new)
    df_new['close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    return df_new

def compute_prediction(X, weights):
    predictions = np.dot(X, weights)
    return predictions

def update_weights_gd(X_train, Y_train, weights, learning_rate):
    predictions = compute_prediction(X_train, weights)
    weights_delta = np.dot(X_train.T, Y_train - predictions)
    m = Y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights

def update_weights_sgd(X_train, Y_train, weights, learning_rate):
    for X_each, Y_each in zip(X_train, Y_train):
        prediction = compute_prediction(X_each, weights)
        weights_delta = X_each.T * (Y_each - prediction)
        weights += learning_rate * weights_delta
    return weights

def compute_cost(X, Y, weights):
    predictions = compute_prediction(X, weights)
    cost = np.mean((predictions - Y) ** 2 / 2.0)
    return cost

def train_linear_regression(X_train, Y_train, max_iter, learning_rate, fit_intercept=False):
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_gd(X_train, Y_train, weights, learning_rate)
        if iteration % 100 == 0:
            print(compute_cost(X_train, Y_train, weights))
    return weights

def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
    return compute_prediction(X, weights)

raw_data = pd.read_csv('DJIA.csv', index_col='Date')
data = pd.DataFrame()
generate_features(raw_data, data)
print(data)
# trial
# X_train = np.array([[6], [2], [3], [4], [1], [5], [2], [6], [4], [7]])
# Y_train = np.array([5.5, 1.6, 2.2, 3.7, 0.8, 5.2, 1.5, 5.3, 4.4, 6.8])
# weights = train_linear_regression(X_train, Y_train, max_iter=100, learning_rate=0.01, fit_intercept=True)
# X_test = np.array([[1.3], [3.5], [5.2], [2.8]])
# predictions = predict(X_test, weights)
# plt.scatter(X_train[:, 0], Y_train, marker='o', c='b')
# plt.scatter(X_test[:, 0], predictions, marker='*', c='k')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
# scikit learn trial
# diabetes = datasets.load_diabetes()
# print(diabetes.data.shape)
# num_test = 30
# X_train = diabetes.data[:-num_test, :]
# Y_train = diabetes.target[:-num_test]
# weights = train_linear_regression(X_train, Y_train, max_iter=5000, learning_rate=1, fit_intercept=True)
# X_test = diabetes.data[-num_test:, :]
# Y_test = diabetes.target[-num_test:]
# predictions = predict(X_test, weights)
# print(predictions)
# print(Y_test)
# regressor = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001, learning_rate='constant', eta0=0.01, max_iter=1000)
# regressor.fit(X_train, Y_train)
# predictions = regressor.predict(X_test)
# print(predictions)
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=[X_train.shape[1]]),
#     tf.keras.layers.Dense(units=1)
# ])
# model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(1.0))
# model.fit(X_train, Y_train, epochs=100, verbose=True)
# predictions = model.predict(X_test)[:, 0]
# print(predictions)
# regression decision tree
def mse(targets):
    if targets.size == 0:
        return 0
    return np.var(targets)

def weighted_mse(groups):
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * mse(group)
    return weighted_sum

print(f'{mse(np.array([1, 2, 3])):.4f}')
print(f'{weighted_mse([np.array([1, 2, 3]), np.array([1, 2])]):.4f}')

def split_node(X, Y, index, value):
    x_index = X[:, index]
    if type(X[0, index]) in [int, float]:
        mask = x_index >= value
    else:
        mask = x_index == value
    left = [X[~mask, :], Y[~mask]]
    right = [X[mask, :], Y[mask]]
    return left, right

def get_best_split(X, Y):
    best_index, best_value, best_score, children = None, None, 1e10, None
    for index in range(len(X[0])):
        for value in np.sort(np.unique(X[:, index])):
            groups = split_node(X, Y, index, value)
            impurity = weighted_mse([groups[0][1], groups[1][1]])
            if impurity < best_score:
                best_index, best_value, best_score, children = index, value, impurity, groups
    return {'index': best_index, 'value': best_value, 'children': children}

def get_leaf(targets):
    return np.mean(targets)

def split(node, max_depth, min_size, depth):
    left, right = node['children']
    del (node['children'])
    if left[1].size == 0:
        node['right'] = get_leaf(right[1])
        return
    if right[1].size == 0:
        node['left'] = get_leaf(left[1])
        return
    if depth >= max_depth:
        node['left'], node['right'] = get_leaf(left[1]), get_leaf(right[1])
        return
    if left[1].size <= min_size:
        node['left'] = get_leaf(left[1])
    else:
        result = get_best_split(left[0], left[1])
        result_left, result_right = result['children']
        if result_left[1].size == 0:
            node['left'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['left'] = get_leaf(result_left[1])
        else:
            node['left'] = result
            split(node['left'], max_depth, min_size, depth + 1)
    if right[1].size <= min_size:
        node['right'] = get_leaf(right[1])
    else:
        result = get_best_split(right[0], right[1])
        result_left, result_right = result['children']
        if result_left[1].size == 0:
            node['right'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['right'] = get_leaf(result_left[1])
        else:
            node['right'] = result
            split(node['right'], max_depth, min_size, depth + 1)

def train_tree(X_train, Y_train, max_depth, min_size):
    root = get_best_split(X_train, Y_train)
    split(root, max_depth, min_size, 1)
    return root

CONDITION = {'numerical': {'yes': '>=', 'no': '<'},
             'categorical': {'yes': 'to', 'no': 'to nie'}}
def visualize_tree(node, depth=0):
    if isinstance(node, dict):
        if type(node['value']) in [int, float]:
            condition = CONDITION['numerical']
        else:
            condition = CONDITION['categorical']
        print('{}|- X{} {} {}'.format(depth * ' ', node['index'] + 1, condition['no'], node['value']))
        if 'left' in node:
            visualize_tree(node['left'], depth + 1)
        print('{}|- X{} {} {}'.format(depth * ' ', node['index'] + 1, condition['yes'], node['value']))
        if 'right' in node:
            visualize_tree(node['right'], depth + 1)
    else:
        print('{}[{}]'.format(depth * ' ', node))

# X_train = np.array([['blizniak', 3],
#                     ['jednorodzinny', 2],
#                     ['jednorodzinny', 3],
#                     ['blizniak', 2],
#                     ['blizniak', 4]], dtype=object)
# Y_train = np.array([600, 700, 800, 400, 700])
# tree = train_tree(X_train, Y_train, 2, 2)
# visualize_tree(tree)

# boston = datasets.fetch_california_housing()
# num_test = 10
# X_train = boston.data[:-num_test, :]
# Y_train = boston.target[:-num_test]
# X_test = boston.data[-num_test:, :]
# Y_test = boston.target[-num_test:]
# regressor = DecisionTreeRegressor(max_depth=10, min_samples_split=3)
# regressor.fit(X_train, Y_train)
# predictions = regressor.predict(X_test)
# print(predictions)
# print(Y_test)
# regressor = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=3)
# regressor.fit(X_train, Y_train)
# predictions = regressor.predict(X_test)
# print(predictions)
# SVR takes too long to train model with california_housing data
# regressor = SVR(C=0.1, epsilon=0.02, kernel='linear')
# regressor.fit(X_train, Y_train)
# predictions = regressor.predict(X_test)
# print(predictions)
# quality evaluation
# param_grid = {
#     'alpha': [1e-07, 1e-06, 1e-05],
#     'penalty': [None, 'l2'],
#     'eta0': [0.03, 0.05, 0.1],
#     'max_iter': [500, 1000]
# }
# regressor = SGDRegressor(loss='squared_error', learning_rate='constant')
# grid_search = GridSearchCV(regressor, param_grid, cv=3)
# grid_search.fit(X_train, Y_train)
# print(grid_search.best_params_)
# regressor_best = grid_search.best_estimator_
# predictions = regressor_best.predict(X_test)
# print(mean_squared_error(Y_test, predictions))
# print(mean_absolute_error(Y_test, predictions))
# print(r2_score(Y_test, predictions))
# stock price predictions
start_train = '1990-01-01'
end_train = '2022-12-31'
start_test = '2023-01-01'
end_test = '2023-12-31'
data_train = data.loc[start_train:end_train]
data_test = data.loc[start_test:end_test]
X_train = data_train.drop('close', axis=1).values
Y_train = data_train['close'].values
X_test = data_test.drop('close', axis=1).values
Y_test = data_test['close'].values
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)
param_grid = {
    'alpha': [1e-4, 3e-4, 1e-3],
    'eta0': [0.01, 0.03, 0.1]
}
lr = SGDRegressor(penalty='l2', max_iter=1000, random_state=42)
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='r2')
grid_search.fit(X_scaled_train, Y_train)
print(grid_search.best_params_)
lr_best = grid_search.best_estimator_
predictions_lr = lr_best.predict(X_scaled_test)
print(f'Mean squared error: {mean_squared_error(Y_test, predictions_lr):.3f}')
print(f'Mean absolute error: {mean_absolute_error(Y_test, predictions_lr):.3f}')
print(f'R^2: {r2_score(Y_test, predictions_lr):.3f}')
param_grid = {
    'max_depth': [30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [3, 5]
}
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
rf_best = grid_search.best_estimator_
predictions_rf = rf_best.predict(X_test)
print(f'Mean squared error: {mean_squared_error(Y_test, predictions_rf):.3f}')
print(f'Mean absolute error: {mean_absolute_error(Y_test, predictions_rf):.3f}')
print(f'R^2: {r2_score(Y_test, predictions_rf):.3f}')
param_grid = [
    {'kernel': ['linear'], 'C': [100, 300, 500], 'epsilon': [0.00003, 0.0001]},
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [10, 100, 1000], 'epsilon': [0.00003, 0.0001]}
]
svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2')
grid_search.fit(X_scaled_train, Y_train)
print(grid_search.best_params_)
svr_best = grid_search.best_estimator_
predictions_svr = svr_best.predict(X_scaled_test)
print(f'Mean squared error: {mean_squared_error(Y_test, predictions_svr):.3f}')
print(f'Mean absolute error: {mean_absolute_error(Y_test, predictions_svr):.3f}')
print(f'R^2: {r2_score(Y_test, predictions_svr):.3f}')
# figure
plt.plot(data_test.index, Y_test, c='k')
plt.plot(data_test.index, predictions_lr, c='b')
plt.plot(data_test.index, predictions_rf, c='r')
plt.plot(data_test.index, predictions_svr, c='g')
plt.xticks(range(0, 252, 10), rotation=60)
plt.xlabel('Date')
plt.ylabel('Closure price')
plt.legend(['Real values', 'Linear Regression', 'Random Forest', 'Support Vector Regression'])
plt.show()