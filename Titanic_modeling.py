from sklearn import neighbors, linear_model, svm, tree, ensemble
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from random import randint

def clf_results(clf,X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=randint(0,1000));
    clf.fit(X_train,y_train);
    y_train_predicted = clf.predict(X_train);
    y_test_predicted = clf.predict(X_test);
    score_train = accuracy_score(y_train,y_train_predicted);
    score_test = accuracy_score(y_test,y_test_predicted);
    print("Score for train: "+str(score_train));
    print("Score for test: "+str(score_test));

# KNN
def knn_results(X,y):
    print("KNN");
    knn = neighbors.KNeighborsClassifier();
    clf_results(knn,X,y);

# Logistic Regression
def logistic_regression_results(X,y):
    print("Logistic Regression");
    lr = linear_model.LogisticRegression();    
    clf_results(lr,X,y);
    
# SVM
def svm_results(X,y):
    print("SVM");
    svc = svm.SVC();
    clf_results(svc,X,y);
    
# Decision Tree
def decission_tree_results(X,y):
    print("Decision Tree");
    dt = tree.DecisionTreeClassifier();
    clf_results(dt,X,y);
    
# Random Forest
def random_forest_results(X,y):
    print("Random Forest");    
    rf = ensemble.RandomForestClassifier();
    clf_results(rf,X,y);
    