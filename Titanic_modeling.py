from sklearn import neighbors, linear_model, svm, tree, ensemble
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def train_clf(clf,X,y):
    # init tables
    score_train = [];
    score_test = [];
    y_train_table = []; 
    y_test_table = [];
    y_train_predicted_table = [];
    y_test_predicted_table = [];
    
    # loop for training model many times
    for i in range(51):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randint(0,1000));
        
        y_train_table.append(y_train);
        y_test_table.append(y_test);
        
        clf.fit(X_train,y_train);
        y_train_predicted_table.append(clf.predict(X_train));
        y_test_predicted_table.append(clf.predict(X_test));
        score_train.append(accuracy_score(y_train,y_train_predicted_table[i]));
        score_test.append(accuracy_score(y_test,y_test_predicted_table[i]));
    
    return score_train,score_test,y_train_table,y_test_table,y_train_predicted_table,y_test_predicted_table;

def print_roc(y_train_true, y_train_score, y_test_true, y_test_score):
    fpr_train, tpr_train, treshold_train = roc_curve(y_train_true, y_train_score);
    fpr_test, tpr_test, treshold_test = roc_curve(y_test_true, y_test_score);
    plt.plot(fpr_train, tpr_train, color='red', label='Logistic regression for train (AUC: %.2f)'% auc(fpr_train, tpr_train));
    plt.plot(fpr_test, tpr_test, color='green', label='Logistic regression for train (AUC: %.2f)'% auc(fpr_test, tpr_test));
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--');
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.01]);
    plt.title('ROC Curve');
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.legend(loc="lower right");
    plt.show();    

def clf_results(clf,X,y):
    score_train,score_test,y_train_table,y_test_table,y_train_predicted_table,y_test_predicted_table = train_clf(clf,X,y);
   
    # training results
    data = {
            'mean': [np.mean(score_train), np.mean(score_test)], 
            'std': [np.std(score_train), np.std(score_test)],
            'median': [np.median(score_train), np.median(score_test)]
        };
    results = pd.DataFrame(data=data, index=['train', 'test']);
    print(results);
    
    
    index = np.argsort(score_test)[len(score_test)//2];   
    print_roc(
            y_train_table[index], y_train_predicted_table[index],
            y_test_table[index], y_test_predicted_table[index]);
            
def clf_results_values(clf,X,y):
    
    score_train,score_test,_,_,_,_ = train_clf(clf,X,y);
    
    # training results
    data = { 
            'mean': [np.mean(score_train), np.mean(score_test)], 
            'std': [np.std(score_train), np.std(score_test)],
            'median': [np.median(score_train), np.median(score_test)]
        };
    results = pd.DataFrame(data=data, index=['train', 'test']);
    return results;

# KNN
def knn_results(X,y):
    print("KNN");
    knn = neighbors.KNeighborsClassifier();
    clf_results(knn,X,y);

# Logistic Regression
def logistic_regression_results(X,y):
    print("Logistic Regression");
    lr = linear_model.LogisticRegression(penalty='l1', tol=0.1, C=10, max_iter=200);    
    clf_results(lr,X,y);
    
# SVM
def svm_results(X,y):
    print("SVM");
    svc = svm.SVC(kernel='rbf', gamma=0.01, C=1000);
    clf_results(svc,X,y);
    
# Decision Tree
def decission_tree_results(X,y):
    print("Decision Tree");
    dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4);
    clf_results(dt,X,y);
    
# Random Forest
def random_forest_results(X,y):
    print("Random Forest");    
    rf = ensemble.RandomForestClassifier(criterion='entropy', max_features='log2', max_depth=4, min_samples_split=3);
    clf_results(rf,X,y);

def knn_analysis(X,y):
    print("KNN analysis");
    
    n_neighbors = range(3,13);
    weights = ['uniform', 'distance'];
    algorithm = ['ball_tree', 'kd_tree', 'brute', 'auto'];
    
    results = [];
    
    for n in n_neighbors:
        for w in weights:
            for a in algorithm: 
                knn = neighbors.KNeighborsClassifier(n_neighbors=n, weights=w, algorithm=a);            
                value = { 'neighbors': n, 'weights': w, 'algorithm': a, 'results': clf_results_values(knn,X,y) }
                results.append(value);

    sumup = pd.DataFrame();
    sumup['results_test'] = [x['results']['mean'][1] for x in results];
    sumup['results_train'] = [x['results']['mean'][0] for x in results];
    sumup['difference'] = [x['results']['mean'][0]-x['results']['mean'][1] for x in results];
    sumup['algorithm'] = [x['algorithm'] for x in results];
    sumup['weights'] = [x['weights'] for x in results];
    sumup['neighbors'] = [x['neighbors'] for x in results];
    return sumup;

def logistic_regression_analysis(X,y):
    print("Logistic Regression analysis");
    
    penalty = ['l1', 'l2'];
    dual = [False];
    tol = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1];
    C_vals = [10, 100, 1000];
    
    results =[];
    
    for p in penalty:
        for d in dual:
            for t in tol:
                for c in C_vals:
                    lr = linear_model.LogisticRegression(penalty=p, dual=d, tol=t, C=c, max_iter=200);
                    value = { 'penalty': p, 'dual': d, 'tol': t, 'C': c, 'results': clf_results_values(lr,X,y) }
                    results.append(value);                
                    
    sumup = pd.DataFrame();
    sumup['results_test'] = [x['results']['mean'][1] for x in results];
    sumup['results_train'] = [x['results']['mean'][0] for x in results];
    sumup['difference'] = [x['results']['mean'][0]-x['results']['mean'][1] for x in results];
    sumup['penalty'] = [x['penalty'] for x in results];
    sumup['dual'] = [x['dual'] for x in results];
    sumup['tol'] = [x['tol'] for x in results];
    sumup['C'] = [x['C'] for x in results];
    return sumup;

def svm_analysis(X,y):
    print("SVM analysis");
    
#    C_vals = [1e-2, 1e-1, 1, 10, 100];
#    kernel = ['linear', 'poly', 'rbf', 'sigmoid'];
#    degree = range(5);
#    gamma = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10];
    
    C_vals = [10, 100, 1000, 10000];
    kernel = ['rbf'];
    degree = [1];
    gamma = [1e-3,1e-2,1e-1];
    
    results =[];
    
    for k in kernel:
        for d in degree:
            for g in gamma:
                for c in C_vals:
                    svc = svm.SVC(kernel=k, degree=d+1, gamma=g, C=c);
                    value = { 'kernel': k, 'degree': d, 'gamma': g, 'C': c, 'results': clf_results_values(svc,X,y) }
                    results.append(value);                
                    
    sumup = pd.DataFrame();
    sumup['results_test'] = [x['results']['mean'][1] for x in results];
    sumup['results_train'] = [x['results']['mean'][0] for x in results];
    sumup['difference'] = [x['results']['mean'][0]-x['results']['mean'][1] for x in results];
    sumup['kernel'] = [x['kernel'] for x in results];
    sumup['degree'] = [x['degree'] for x in results];
    sumup['gamma'] = [x['gamma'] for x in results];
    sumup['C'] = [x['C'] for x in results];
    return sumup;
    
def decission_tree_analysis(X,y):
    print("Decision Tree analysis");
    
    criterion = ['gini','entropy'];
    splitter = ['best','random'];
    max_depth = range(1,10);
    min_samples_split = range(2,5);
    
    results =[];
    
    for c in criterion:
        for s in splitter:
            for md in max_depth:
                for mss in min_samples_split:
                    dt = tree.DecisionTreeClassifier(criterion=c, splitter=s, max_depth=md, min_samples_split=mss);
                    value = { 'criterion': c, 'splitter': s, 'max_depth': md, 'min_samples_split': mss, 'results': clf_results_values(dt,X,y) }
                    results.append(value);                
                    
    sumup = pd.DataFrame();
    sumup['results_test'] = [x['results']['mean'][1] for x in results];
    sumup['results_train'] = [x['results']['mean'][0] for x in results];
    sumup['difference'] = [x['results']['mean'][0]-x['results']['mean'][1] for x in results];
    sumup['criterion'] = [x['criterion'] for x in results];
    sumup['splitter'] = [x['splitter'] for x in results];
    sumup['max_depth'] = [x['max_depth'] for x in results];
    sumup['min_samples_split'] = [x['min_samples_split'] for x in results];
    return sumup;

def random_forest_analysis(X,y):
    print("Random Forest analysis"); 
    
    criterion = ['gini','entropy'];
    max_features = ['sqrt','log2'];
    max_depth = range(2,7);
    min_samples_split = range(2,5);
    
    results =[];
    
    for c in criterion:
        for mf in max_features:
            for md in max_depth:
                for mss in min_samples_split:
                    rf = ensemble.RandomForestClassifier(criterion=c, max_features=mf, max_depth=md, min_samples_split=mss);
                    value = { 'criterion': c, 'max_features': mf, 'max_depth': md, 'min_samples_split': mss, 'results': clf_results_values(rf,X,y) }
                    results.append(value);                
                    
    sumup = pd.DataFrame();
    sumup['results_test'] = [x['results']['mean'][1] for x in results];
    sumup['results_train'] = [x['results']['mean'][0] for x in results];
    sumup['difference'] = [x['results']['mean'][0]-x['results']['mean'][1] for x in results];
    sumup['criterion'] = [x['criterion'] for x in results];
    sumup['max_features'] = [x['max_features'] for x in results];
    sumup['max_depth'] = [x['max_depth'] for x in results];
    sumup['min_samples_split'] = [x['min_samples_split'] for x in results];
    return sumup;
    
    