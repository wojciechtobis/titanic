from sklearn import neighbors, linear_model, svm, tree, ensemble
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def train_clf(clf,X,y,n):
    # init tables
    score_train = [];
    score_test = [];
    y_train_table = []; 
    y_test_table = [];
    y_train_predicted_table = [];
    y_test_predicted_table = [];
    
    # loop for training model many times
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randint(0,1000));
        
        y_train_table.append(y_train);
        y_test_table.append(y_test);
        
        clf.fit(X_train,y_train);
        y_train_predicted_table.append(clf.predict(X_train));
        y_test_predicted_table.append(clf.predict(X_test));
        score_train.append(accuracy_score(y_train,y_train_predicted_table[i]));
        score_test.append(accuracy_score(y_test,y_test_predicted_table[i]));
        
    result = {
                'score_train': score_train,
                'score_test': score_test,
                'y_train_table': y_train_table,
                'y_test_table': y_test_table,
                'y_train_predicted_table': y_train_predicted_table,
                'y_test_predicted_table': y_test_predicted_table,
                'clf': clf
            }
    
    return result;

def print_roc(y_train_true, y_train_score, y_test_true, y_test_score):
    fpr_train, tpr_train, treshold_train = roc_curve(y_train_true, y_train_score);
    fpr_test, tpr_test, treshold_test = roc_curve(y_test_true, y_test_score);
    plt.plot(fpr_train, tpr_train, color='red', label='Train (AUC: %.2f)'% auc(fpr_train, tpr_train));
    plt.plot(fpr_test, tpr_test, color='green', label='Test (AUC: %.2f)'% auc(fpr_test, tpr_test));
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--');
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.01]);
    plt.title('ROC Curve');
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.legend(loc="lower right");
    plt.show();    

def clf_results(clf,X,y):
    result = train_clf(clf,X,y,51);
    
    score_train = result['score_train'];
    score_test = result['score_test'];
   
    # training results
    data = {
            'mean': [np.mean(score_train), np.mean(score_test)], 
            'std': [np.std(score_train), np.std(score_test)],
            'median': [np.median(score_train), np.median(score_test)]
        };
    results = pd.DataFrame(data=data, index=['train', 'test']);
    return results;
    
def clf_roc(clf,X,y):
    result = train_clf(clf,X,y,51);
    
    score_test = result['score_test'];
    y_train_table = result['y_train_table'];
    y_test_table = result['y_test_table'];
    y_train_predicted_table = result['y_train_predicted_table'];
    y_test_predicted_table = result['y_test_predicted_table'];
    
    index = np.argsort(score_test)[len(score_test)//2];   
    print_roc(
            y_train_table[index], y_train_predicted_table[index],
            y_test_table[index], y_test_predicted_table[index]);

def cfl_param_optimisation(clf,X,y,params):
    paramNames = list(params.keys());
    
    param0 = params[paramNames[0]];
    param1 = params[paramNames[1]];
    param2 = params[paramNames[2]];
    param3 = params[paramNames[3]];
    
    results = [];
    
    for p0 in param0:
        setattr(clf,paramNames[0],p0);
        for p1 in param1:
            setattr(clf,paramNames[1],p1);
            for p2 in param2:
                setattr(clf,paramNames[2],p2);
                for p3 in param3:
                    setattr(clf,paramNames[3],p3);
                    value = { 
                            paramNames[0]: p0, 
                            paramNames[1]: p1,
                            paramNames[2]: p2,
                            paramNames[3]: p3,
                            'results': clf_results(clf,X,y) 
                            }
                    results.append(value);  
                    
    sumup = pd.DataFrame();
    sumup['results_test'] = [x['results']['mean'][1] for x in results];
    sumup['results_train'] = [x['results']['mean'][0] for x in results];
    sumup['difference'] = [x['results']['mean'][0]-x['results']['mean'][1] for x in results];
    for pn in paramNames:
        sumup[pn] = [x[pn] for x in results];

    return sumup;

def show_results(clf,X,y):
    print(clf_results(clf,X,y));
    clf_roc(clf,X,y);

# KNN
def knn_show_results(X,y):
    print("KNN");
    knn = neighbors.KNeighborsClassifier();
    show_results(knn,X,y);
    
def knn_params_analysis(X,y):
    print("KNN analysis");
    
    params = {
            'n_neighbors': range(3,13),
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
            'p': [2]
            }
    
    clf = neighbors.KNeighborsClassifier();
    
    return cfl_param_optimisation(clf,X,y,params);

# Logistic Regression
def logistic_regression_show_results(X,y):
    print("Logistic Regression");
    lr = linear_model.LogisticRegression(penalty='l1', tol=0.1, C=10, max_iter=200);    
    show_results(lr,X,y);
    
def logistic_regression_params_analysis(X,y):
    print("Logistic Regression analysis");
    
    params = {
            'penalty': ['l1', 'l2'],
            'dual': [False],
            'tol': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            'C_vals': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            }
    
    clf = linear_model.LogisticRegression();
    
    return cfl_param_optimisation(clf,X,y,params);
    
# SVM
def svm_show_results(X,y):
    print("SVM");
    svc = svm.SVC(kernel='rbf', gamma=0.01, C=1000);
    show_results(svc,X,y);
    
def svm_params_analysis(X,y):
    print("SVM analysis");
    
    params = {
            'C_vals': [10, 100, 1000, 10000],
            'kernel': ['rbf'],
            'degree': [1],
            'gamma': [1e-3,1e-2,1e-1]
            }
    
    clf = svm.SVC();
    
    return cfl_param_optimisation(clf,X,y,params);
    
# Decision Tree
def decission_tree_show_results(X,y):
    print("Decision Tree");
    dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4);
    show_results(dt,X,y);
    
def decission_tree_params_analysis(X,y):
    print("Decision Tree analysis");
    
    params = {
            'criterion': ['gini','entropy'],
            'splitter': ['best','random'],
            'max_depth': range(1,10),
            'min_samples_split': range(2,5)
            }
    
    clf = tree.DecisionTreeClassifier();
    
    return cfl_param_optimisation(clf,X,y,params);
    
# Random Forest
def random_forest_show_results(X,y):
    print("Random Forest");    
    rf = ensemble.RandomForestClassifier(criterion='entropy', max_features='log2', max_depth=4, min_samples_split=3);
    show_results(rf,X,y);

def random_forest_params_analysis(X,y):
    print("Random Forest analysis"); 
    
    params = {
            'criterion': ['gini','entropy'],
            'max_features': ['sqrt','log2'],
            'max_depth': range(2,7),
            'min_samples_split': range(2,5)
            }
    
    clf = ensemble.RandomForestClassifier();
    
    return cfl_param_optimisation(clf,X,y,params);