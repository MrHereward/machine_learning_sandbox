import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

def run_optimization(x, y):
    with tf.GradientTape() as g:
        logits = tf.add(tf.matmul(x, W), b)[:, 0]
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
        gradients = g.gradient(cost, [W, b])
        optimizer.apply_gradients(zip(gradients, [W, b]))

rows = 100000
df = pd.read_csv('train.csv', nrows=rows)
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values
train_number = 90000
X_train = X[:train_number]
Y_train = Y[:train_number]
X_test = X[train_number:]
Y_test = Y[train_number:]
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)
X_train_enc = X_train_enc.toarray().astype('float32')
X_test_enc = X_test_enc.toarray().astype('float32')
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')
batch_size = 1000
train_data = tf.data.Dataset.from_tensor_slices((X_train_enc, Y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
features_number = int(X_train_enc.shape[1])
W = tf.Variable(tf.zeros([features_number, 1]))
b = tf.Variable(tf.zeros([1]))
learning_rate = 0.0008
optimizer = tf._optimizers.Adam(learning_rate)
training_steps = 6000
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)
    if step % 500 == 0:
        logits = tf.add(tf.matmul(batch_x, W), b)[:, 0]
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_y, logits=logits))
        print('Liczba krokow: %i, strata: %f' % (step, loss))
logits = tf.add(tf.matmul(X_test_enc, W), b)[:, 0]
pred = tf.nn.sigmoid(logits)
auc_metric = tf._metrics.AUC()
auc_metric.update_state(Y_test, pred)
print(f'Pole pod krzywa ROC dla zbioru testowego: {auc_metric.result().numpy():.3f}')