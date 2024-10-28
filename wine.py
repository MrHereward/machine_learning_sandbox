from sklearn.datasets import load_wine
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

wine_data = load_wine()
X = wine_data.data
Y = wine_data.target
print("Wielkosc zbioru wejsciowego:", X.shape)
print("Wielkosc zbioru wyjsciowego:", Y.shape)
print("Etykiet:", wine_data.target_names)
class0 = (Y == 0).sum()
class1 = (Y == 1).sum()
class2 = (Y == 2).sum()
print(f"Liczba class0: {class0}, liczba class1: {class1}, liczba class2: {class2}")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(f"Dokladnosc: {accuracy*100:.1f}%")
pred = clf.predict(X_test)
print(classification_report(Y_test, pred))