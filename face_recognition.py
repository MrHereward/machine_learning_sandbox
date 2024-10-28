from sklearn.datasets import fetch_lfw_people
#import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
#from sklearn.svm import LinearSVC

#face_data = fetch_lfw_people(min_faces_per_person=80)
face_data = fetch_lfw_people(min_faces_per_person=50)
X = face_data.data
Y = face_data.target
print("Wielkosc zbioru wejsciowego:", X.shape)
print("Wielkosc zbioru wyjsciowego:", Y.shape)
print("Etykiety:", face_data.target_names)
for i in range(5):
    print(f"Klasa {i} ilosc probek: {(Y == i).sum()}")
#fig, ax = plt.subplots(3, 4)
#for i, axi in enumerate(ax.flat):
#    axi.imshow(face_data.images[i], cmap='bone')
#    axi.set(xticks=[], yticks=[], xlabel=face_data.target_names[face_data.target[i]])
#plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
pca = PCA(n_components=100, whiten=True)
svc = SVC(class_weight='balanced', kernel='rbf')
#svc = LinearSVC(class_weight='balanced')
model = Pipeline([('pca', pca), ('svc', svc)])
parameters_pipeline = {'svc__C': [1, 3, 10],
                       'svc__gamma': [0.001, 0.005]}
#parameters_pipeline = {'svc__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
#                       'svc__dual': [True, False]}
grid_search = GridSearchCV(model, parameters_pipeline)
grid_search.fit(X_train, Y_train)
print('Najlepszy model:', grid_search.best_params_)
print('Najlepsza srednia skutecznosc:', grid_search.best_score_)
model_best = grid_search.best_estimator_
pred = model_best.predict(X_test)
print(f'Dokladnosc: {model_best.score(X_test, Y_test)*100:.1f}%')
print(classification_report(Y_test, pred, target_names=face_data.target_names))