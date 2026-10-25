from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


def classify(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    
    print("Decision Tree :")
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    y_pred = decision_tree.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum()))
    print((1 - (y_test != y_pred).sum().item() / (X_test.shape[0])))
    print("====================================================") 

    print("SVM :")
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    y_pred = svm.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum()))
    print((1 - (y_test != y_pred).sum().item() / (X_test.shape[0])))
    print("====================================================") 

    print("Gaussian Naive Bayes :")
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum()))
    print((1 - (y_test != y_pred).sum().item() / (X_test.shape[0])))
    print("====================================================") 

    print("KNN :")
    neigh = KNeighborsClassifier()
    y_pred = neigh.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum()))
    print((1 - (y_test != y_pred).sum().item() / (X_test.shape[0])))
    print("====================================================") 



def classify_cv(X,Y):

    results = {}
    print("Decision Tree :")
    param_decision_tree = {
        'random_state' : [0],
        'max_depth': [1, 2, 5, 10], 
        'criterion': ['gini', 'entropy', 'log_loss'], 
        'splitter' : ['best', 'random']
        }
    decision_tree = GridSearchCV(DecisionTreeClassifier(), param_decision_tree, cv = 4, scoring='accuracy')
    decision_tree.fit(X,Y)
    scores = cross_val_score(decision_tree.best_estimator_, X, Y, cv = 4)
    print("%0.2f mean accuracy\n %0.2f max accuracy\n %0.2f min accuracy\n %0.2f standard deviation" % 
        (scores.mean(), scores.max(), scores.min(), scores.std()))
    print(decision_tree.best_params_)
    results['Decision tree'] = round(scores.mean() * 100, 2)
    print("====================================================")
    
    print("SVM :")
    param_svm = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    #svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svm = GridSearchCV(SVC(), param_svm, cv = 4, scoring='accuracy')
    svm.fit(X,Y)
    scores = cross_val_score(svm.best_estimator_, X, Y, cv = 4)
    print("%0.2f mean accuracy\n %0.2f max accuracy\n %0.2f min accuracy\n %0.2f standard deviation" % 
        (scores.mean(), scores.max(), scores.min(), scores.std()))
    print(svm.best_params_)
    results['SVM'] = round(scores.mean() * 100, 2)
    print("====================================================")

    print("Gaussian Naive Bayes :")
    gnb = GaussianNB()
    gnb.fit(X,Y)
    scores = cross_val_score(gnb, X, Y, cv = 4)
    print("%0.2f mean accuracy\n %0.2f max accuracy\n %0.2f min accuracy\n %0.2f standard deviation" % 
        (scores.mean(), scores.max(), scores.min(), scores.std()))
    results['GNB'] = round(scores.mean() * 100, 2)
    print("====================================================")

    print("KNN :")
    param_knn = {
        'n_neighbors': [5, 7, 10, 20], 
        'weights': ['uniform', 'distance']
    }
    knn = GridSearchCV(KNeighborsClassifier(), param_knn, cv = 4, scoring='accuracy')
    knn.fit(X,Y)
    scores = cross_val_score(knn.best_estimator_, X, Y, cv = 4)
    print("%0.2f mean accuracy\n %0.2f max accuracy\n %0.2f min accuracy\n %0.2f standard deviation" % 
        (scores.mean(), scores.max(), scores.min(), scores.std()))
    print(knn.best_params_)
    results['KNN'] = round(scores.mean() * 100, 2)
    print("====================================================")
    return results