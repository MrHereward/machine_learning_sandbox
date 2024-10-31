from sklearn.feature_extraction import DictVectorizer
import pandas as pd

X_dict = [{'zainteresowania': 'technika', 'zawod': 'specjalista'},
          {'zainteresowania': 'moda', 'zawod': 'student'},
          {'zainteresowania': 'moda', 'zawod': 'specjalista'},
          {'zainteresowania': 'sport', 'zawod': 'student'},
          {'zainteresowania': 'technika', 'zawod': 'student'},
          {'zainteresowania': 'technika', 'zawod': 'emeryt'},
          {'zainteresowania': 'sport', 'zawod': 'specjalista'}]
dict_one_hot_encoder = DictVectorizer(sparse=False)
X_encoded = dict_one_hot_encoder.fit_transform(X_dict)
print(X_encoded)
print(dict_one_hot_encoder.vocabulary_)
new_dict = [{'zainteresowania': 'sport', 'zawod': 'emeryt'}]
new_encoded = (dict_one_hot_encoder.transform(new_dict))
print(new_encoded)
print(dict_one_hot_encoder.inverse_transform(new_encoded))
df = pd.DataFrame({'ocena': ['niska', 'wysoka', 'srednia', 'srednia', 'niska']})
print(df)
mapping = {'niska': 1, 'srednia': 2, 'wysoka': 3}
df['ocena'] = df['ocena'].replace(mapping)
print(df)