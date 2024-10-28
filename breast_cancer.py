from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer_data = load_breast_cancer()
X = cancer_data.data
Y = cancer_data.target
print("Wielkosc zbioru wejsciowego:", X.shape)
print("Wielkosc zbioru wyjsciowego:", Y.shape)
pos_number = (Y == 1).sum()
neg_number = (Y == 0).sum()
print(f"Liczba probek pozytywnych: {pos_number}, liczba probek negatywnych: {neg_number}")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(f"Dokladnosc: {accuracy*100:.1f}%")
