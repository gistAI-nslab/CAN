from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score

from tools import load_data


def knn(X, Y):
    X_train, X_test, Y_train, Y_test = load_data()

    n = 5
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(X_train, Y_train)

    Y_ = clf.predict(X_test)
    acc = clf.score(X_test, Y_test)*100

    precision, recall, _, _ = score(Y_test, Y_, zero_division=1)

    print(f'knn {n} {acc} {precision} {recall}')

def svm(X, Y):
    X_train, X_test, Y_train, Y_test = load_data()
    
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    C = 0.01
    
    clf = svm.SVC(kernel = 'linear', C=C, max_iter=10000)
    #clf = svm.SVC(kernel = 'rbf', gamma=0.1, C=C, max_iter=10000)
    #clf = svm.SVC(kernel = 'poly', degree=5, C=C, max_iter=1000)
    clf.fit(X_train, Y_train)

    Y_ = clf.predict(X_test)
    acc = clf.score(X_test, Y_test)*100

    precision, recall, _, _ = score(Y_test, Y_, zero_division=1)

    print(f'svm poly {acc} {precision} {recall}')


if __name__ == "__main__":
    knn()
    svm()