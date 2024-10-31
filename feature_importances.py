import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

rows = 100000
df = pd.read_csv('train.csv', nrows=rows)
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values
train_number = 90000
X_train = X[:train_number]
Y_train = Y[:train_number]
X_test = X[train_number:train_number * 2]
Y_test = Y[train_number:train_number * 2]
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)
random_forest.fit(X_train_enc.toarray(), Y_train)
feature_imp = random_forest.feature_importances_
print(feature_imp)
feature_names = enc.get_feature_names_out()
print(np.sort(feature_imp)[:10])
bottom_10 = np.argsort(feature_imp)[:10]
print('10 najmniej istotnych cech:\n', feature_names[bottom_10])
print(np.sort(feature_imp)[-10:])
top_10 = np.argsort(feature_imp)[-10:]
print('10 najbardziej istotnych cech:\n', feature_names[top_10])