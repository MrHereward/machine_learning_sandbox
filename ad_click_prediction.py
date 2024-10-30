import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

rows = 300000
df = pd.read_csv('train.csv', nrows=rows)
print(df.head(5))
Y = df['click'].values
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
print(X.shape)
train_number = int(rows * 0.9)
X_train = X[:train_number]
Y_train = Y[:train_number]
X_test = X[train_number:]
Y_test = Y[train_number:]
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
print(X_train_enc.shape)
X_test_enc = enc.transform(X_test)
parameters = {'max_depth': [3, 10, None]}

#decision tree

decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=20)
grid_search = GridSearchCV(decision_tree, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)
decision_tree_best = grid_search.best_estimator_
pos_prob = decision_tree_best.predict_proba(X_test_enc)[:, 1]
print(f'Pole pod krzywa ROC dla zbioru testowego: {roc_auc_score(Y_test, pos_prob):.3f}')
pos_prob = np.zeros(len(Y_test))
click_index = np.random.choice(len(Y_test), int(len(Y_test) * 51211.0/300000), replace=False)
pos_prob[click_index] = 1
print(f'Pole pod krzywa ROC dla zbioru testowego: {roc_auc_score(Y_test, pos_prob):.3f}')

#forest classifier

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)
grid_search = GridSearchCV(random_forest, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)
random_forest_best = grid_search.best_estimator_
pos_prob = random_forest_best.predict_proba(X_test_enc)[:, 1]
print('Pole pod krzywa ROC dla zbioru testowego: {0:.3f}'.format(roc_auc_score(Y_test, pos_prob)))

#gradient boosted tree

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

le = LabelEncoder()
Y_train_enc = le.fit_transform(Y_train)
model = xgb.XGBClassifier(learning_rate=0.1, max_depth=30, n_estimators=1000)
model.fit(X_train_enc, Y_train_enc)
pos_prob = model.predict_proba(X_test_enc)[:, 1]
print(f'Pole pod krzywa ROC dla zbioru testowego: {roc_auc_score(Y_test, pos_prob):.3f}')