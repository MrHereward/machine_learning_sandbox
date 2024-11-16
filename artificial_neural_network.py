import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.src.callbacks import TensorBoard, EarlyStopping
from keras.src.layers import Dense
from keras.src.optimizers import AdamW
from sklearn import datasets, preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.python import keras
from tensorflow.python.keras.optimizers import adam_v2
import pandas as pd
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.ops.math_ops import linspace
from tensorflow.python.ops.summary_ops_v2 import create_file_writer

# def sigmoid(z):
#     return 1.0 / (1 + np.exp(-z))
#
# def sigmoid_derivative(z):
#     return sigmoid(z) * (1.0 - sigmoid(z))
#
# def train(X, y, n_hidden, learning_rate, n_iter):
#     m, n_input = X.shape
#     W1 = np.random.randn(n_input, n_hidden)
#     b1 = np.zeros((1, n_hidden))
#     W2 = np.random.randn(n_hidden, 1)
#     b2 = np.zeros((1, 1))
#     for i in range(1, n_iter + 1):
#         Z2 = np.matmul(X, W1) + b1
#         A2 = sigmoid(Z2)
#         Z3 = np.matmul(A2, W2) + b2
#         A3 = Z3
#         dZ3 = A3 - y
#         dW2 = np.matmul(A2.T, dZ3)
#         db2 = np.sum(dZ3, axis=0, keepdims=True)
#         dZ2 = np.matmul(dZ3, W2.T) * sigmoid_derivative(Z2)
#         dW1 = np.matmul(X.T, dZ2)
#         db1 = np.sum(dZ2, axis=0)
#         W2 = W2 - learning_rate * dW2 / m
#         b2 = b2 - learning_rate * db2 / m
#         W1 = W1 - learning_rate * dW1 / m
#         b1 = b1 - learning_rate * db1 / m
#         if i % 100 == 0:
#             cost = np.mean((y - A3) ** 2)
#             print('Iteracja: %i, strata: %f' % (i, cost))
#     model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
#     return model
#
# def predict(x, model):
#     W1 = model['W1']
#     b1 = model['b1']
#     W2 = model['W2']
#     b2 = model['b2']
#     A2 = sigmoid(np.matmul(x, W1) + b1)
#     A3 = np.matmul(A2, W2) + b2
#     return A3
#
# boston = datasets.fetch_california_housing()
# num_test = 10
# scaler = preprocessing.StandardScaler()
# X_train = boston.data[:-num_test, :]
# X_train = scaler.fit_transform(X_train)
# y_train = boston.target[:-num_test].reshape(-1, 1)
# X_test = boston.data[-num_test:, :]
# X_test = scaler.transform(X_test)
# y_test = boston.target[-num_test:]
# n_hidden = 20
# learning_rate = 0.1
# n_iter = 2000
# model = train(X_train, y_train, n_hidden, learning_rate, n_iter)
# predictions = predict(X_test, model)
# print(predictions)
# print(y_test)
# scikit-learn
# nn_scikit = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', learning_rate_init=0.001, random_state=42, max_iter=2000)
# nn_scikit.fit(X_train, y_train)
# predictions = nn_scikit.predict(X_test)
# print(predictions)
# print(np.mean((y_test - predictions) ** 2))
# tensorflow does not work
# tf.random.set_seed(42)
# model = keras.Sequential([
#     keras.layers.Dense(units=20, activation='relu'),
#     keras.layers.Dense(units=8, activation='relu'),
#     keras.layers.Dense(units=1)
# ])
# dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# dataset = dataset.batch(32)
# model.compile(loss='mean_squared_error', optimizer=adam_v2.Adam(learning_rate=0.02))
# model.fit(dataset, epochs=300)
# predictions = model.predict(X_test)[:, 0]
# print(predictions)
# print(np.mean((y_test - predictions) ** 2))

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

raw_data = pd.read_csv('DJIA.csv', index_col='Date')
data = pd.DataFrame()
generate_features(raw_data, data)
print(data)
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

# model = Sequential([
#     Dense(units=32, activation='relu'),
#     Dense(units=1)
# ])
# model.compile(loss='mean_squared_error', optimizer=AdamW(learning_rate=0.1))
# model.fit(X_scaled_train, Y_train, epochs=100, verbose=True)
# predictions = model.predict(X_scaled_test)
# print(f'Mean squared error: {mean_squared_error(Y_test, predictions):.3f}')
# print(f'Mean absolute error: {mean_absolute_error(Y_test, predictions):.3f}')
# print(f'R^2: {r2_score(Y_test, predictions):.3f}')

HP_HIDDEN = hp.HParam('hidden_size', hp.Discrete([64, 32, 16]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([300, 1000]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.01, 0.4))
def train_test_model(hparams, logdir):
    model = Sequential([
        Dense(units=hparams[HP_HIDDEN], activation='relu'),
        Dense(units=1)
    ])
    model.compile(loss='mean_squared_error', optimizer=AdamW(learning_rate=hparams[HP_LEARNING_RATE]), metrics=['mean_squared_error'])
    model.fit(X_scaled_train, Y_train, validation_data=(X_scaled_test, Y_test), epochs=hparams[HP_EPOCHS], verbose=False, callbacks= [
        TensorBoard(logdir),
        hp.KerasCallback(logdir, hparams),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=0, mode='auto')
    ])
    _, mse = model.evaluate(X_scaled_test, Y_test)
    pred = model.predict(X_scaled_test)
    r2 = r2_score(Y_test, pred)
    return mse, r2

def run(hparams, logdir):
    with create_file_writer(logdir).as_default():
        hp.hparams_config(
            hparams = [HP_HIDDEN, HP_EPOCHS, HP_LEARNING_RATE],
            metrics=[hp.Metric('mean_square_error', display_name='mse'),
                     hp.Metric('r2', display_name='r2')]
        )
        mse, r2 = train_test_model(hparams, logdir)
        tf.summary.scalar('mean_square_error', mse, step=1)
        tf.summary.scalar('r2', r2, step=1)

# session_num = 1
# for hidden in HP_HIDDEN.domain.values:
#     for epochs in HP_EPOCHS.domain.values:
#         for learning_rate in linspace(HP_LEARNING_RATE.domain.min_value, HP_LEARNING_RATE.domain.max_value, 5):
#             hparams = {
#                 HP_HIDDEN: hidden,
#                 HP_EPOCHS: epochs,
#                 HP_LEARNING_RATE: float("%.2f"%float(learning_rate))
#             }
#             run_name = "run-%d" % session_num
#             print('--- Próba: %s' % run_name)
#             print({h.name: hparams[h] for h in hparams})
#             run(hparams, 'logs/hparam_tuning/' + run_name)
#             session_num += 1

model = Sequential([
    Dense(units=64, activation='relu'),
    Dense(units=1)
])
model.compile(loss='mean_squared_error', optimizer=AdamW(0.01))
model.fit(X_scaled_train, Y_train, epochs=300, verbose=False)
predictions = model.predict(X_scaled_test)[:, 0]

plt.plot(data_test.index, Y_test, c='k')
plt.plot(data_test.index, predictions, c='b')
plt.plot(data_test.index, predictions, c='r')
plt.plot(data_test.index, predictions, c='g')
plt.xticks(range(0, 252, 10), rotation=60)
plt.xlabel('Data')
plt.ylabel('Cena zamknięcia')
plt.legend(['Wartości rzeczywiste', 'Prognozy sieci neuronowej'])
plt.show()