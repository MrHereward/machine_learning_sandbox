import timeit
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

rows = 300000
df = pd.read_csv('train.csv', nrows=rows)
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values
train_number = 270000
X_train = X[:train_number]
Y_train = Y[:train_number]
X_test = X[train_number:]
Y_test = Y[train_number:]
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train)
sgd_lr_online = SGDClassifier(loss='log_loss', penalty=None, fit_intercept=True, max_iter=1, learning_rate='constant', eta0=0.01)
start_time = timeit.default_timer()
for i in range(9):
    x_train = X_train[i*30000:(i+1)*30000]
    y_train = Y_train[i*30000:(i+1)*30000]
    x_train_enc = enc.transform(x_train)
    sgd_lr_online.partial_fit(x_train_enc.toarray(), y_train, classes=[0, 1])
print(f'--- {(timeit.default_timer() - start_time)}.3f s ---')
x_test_enc = enc.transform(X_test)
pred = sgd_lr_online.predict_proba(x_test_enc.toarray())[:, 1]
print(f'Liczba probek treningowych: {train_number}, pole pod krzywa ROC dla zbioru treningowego: {roc_auc_score(Y_test, pred):.3f}')