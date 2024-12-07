import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import datasets
from keras import preprocessing
from keras.src import optimizers
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.src.layers import Embedding, Dense, LSTM, TimeDistributed, Bidirectional
from keras.src.models import Sequential
from keras.src.optimizers import RMSprop

# imdb
vocab_size = 500
(X_train, Y_train), (X_test, Y_test) = datasets.imdb.load_data(num_words=vocab_size)
print('Liczba próbek treningowych: ', len(Y_train))
print('Liczba pozytywnych recenzji: ', sum(Y_train))
print('Liczba próbek testowych: ', len(Y_test))
print(X_train[0])
word_index = datasets.imdb.get_word_index()
index_word = {index: word for word, index in word_index.items()}
print([index_word.get(i, ' ') for i in X_train[0]])
review_lengths = [len(x) for x in X_train]
# plt.hist(review_lengths, bins=10)
# plt.show()
maxlen = 200
X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)
print('Kształt zbioru X_train po przetworzeniu: ', X_train.shape)
print('Kształt zbioru X_test po przetworzeniu: ', X_test.shape)
tf.random.set_seed(42)
model = Sequential()
embedding_size = 32
# model.add(Embedding(vocab_size, embedding_size))
# model.add(LSTM(50))
# model.add(Dense(1, activation='sigmoid'))
# print(model.summary())
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# batch_size = 64
# n_epoch = 3
# model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epoch, validation_data=(X_test, Y_test))
# acc = model.evaluate(X_test, Y_test, verbose=0)[1]
# print('Dokładność modelu: ', acc)
model.add(Embedding(vocab_size, embedding_size))
# model.add(LSTM(50, return_sequences=True, dropout=0.2))
# model.add(LSTM(50, dropout=0.2))
model.add(Bidirectional(LSTM(50, return_sequences=True, dropout=0.2)))
model.add(Bidirectional(LSTM(50, dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
optimizer = optimizers.Adam(learning_rate=0.003)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
n_epoch = 7
batch_size = 64
model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epoch, validation_data=(X_test, Y_test))
acc = model.evaluate(X_test, Y_test, verbose=0)[1]
print('Dokładność modelu z dwiema warstwami LSTM: ', acc)
# warpeace
# def generate_text(model, gen_length, n_vocab, index_to_char):
#     index = np.random.randint(n_vocab)
#     y_char = [index_to_char[index]]
#     X = np.zeros((1, gen_length, n_vocab))
#     for i in range(gen_length):
#         X[0, i, index] = 1
#         indices = np.argmax(model.predict(X[:, max(0, i - seq_length -1):i + 1])[0], 1)
#         index = indices[-1]
#         y_char.append(index_to_char[index])
#     return ''.join(y_char)
#
# class ResultChecker(Callback):
#     @property
#     def model(self):
#         return self._model
#
#     def __init__(self, model, N, gen_length):
#         super().__init__()
#         self.model = model
#         self.N = N
#         self.gen_length = gen_length
#
#     @model.setter
#     def model(self, value):
#         self._model = value
#
#     def on_epoch_end(self, epoch, logs=None):
#         if logs is None:
#             logs = {}
#         if epoch % self.N == 0:
#             result = generate_text(self.model, self.gen_length, n_vocab, index_to_char)
#             print('\nMoja Wojna i pokój:\n' + result)
#
#
# training_file = 'drive/My Drive/warpeace_input.txt'
# raw_text = open(training_file, 'r').read()
# raw_text = raw_text.lower()
# all_words = raw_text.split()
# unique_words = list(set(all_words))
# print(f'Liczba unikatowych słów: {len(unique_words)}')
# n_chars = len(raw_text)
# print(f'Liczba znaków: {n_chars}')
# chars = sorted(set(raw_text))
# n_vocab = len(chars)
# print(f'Liczba unikatowych słów: {n_vocab}')
# print(chars)
# index_to_char = dict((i, c) for i, c in enumerate(chars))
# char_to_index = dict((c, i) for i, c in enumerate(chars))
# print(char_to_index)
# seq_length = 160
# n_seq = int(n_chars / seq_length)
# X = np.zeros((n_seq, seq_length, n_vocab))
# Y = np.zeros((n_seq, seq_length, n_vocab))
# for i in range(n_seq):
#     x_sequence = raw_text[i * seq_length : (i + 1) * seq_length]
#     x_sequence_ohe = np.zeros((seq_length, n_vocab))
#     for j in range(seq_length):
#         char = x_sequence[j]
#         index = char_to_index[char]
#         x_sequence_ohe[j][index] = 1
#     X[i] = x_sequence_ohe
#     y_sequence = raw_text[i * seq_length + 1 : (i + 1) * seq_length + 1]
#     y_sequence_ohe = np.zeros((seq_length, n_vocab))
#     for j in range(seq_length):
#         char = y_sequence[j]
#         index = char_to_index[char]
#         y_sequence_ohe[j][index] = 1
#     Y[i] = y_sequence_ohe
# print(X.shape)
# print(Y.shape)
# tf.random.set_seed(42)
# hidden_units = 700
# dropout = 0.4
# batch_size = 100
# n_epoch = 300
# model = Sequential()
# model.add(LSTM(hidden_units, input_shape=(None, n_vocab), return_sequences=True, dropout=dropout))
# model.add(LSTM(hidden_units, return_sequences=True, dropout=dropout))
# model.add(TimeDistributed(Dense(n_vocab, activation='softmax')))
# optimizer = RMSprop(learning_rate=0.001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)
# print(model.summary())
# file_path = 'drive/My Drive/weights/weights_epoch_{epoch:03d}_loss_{loss:.4f}.hdf5.keras'
# checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
# early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=50, verbose=1, mode='min')
# result_checker = ResultChecker(model, 10, 500)
# model.fit(X, Y, batch_size=batch_size, verbose=1, epochs=n_epoch, callbacks=[result_checker, checkpoint, early_stop])
