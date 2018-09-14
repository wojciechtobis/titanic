import Titanic_data_analysis as tda
import Titanic_feature_engineering as tfe
import Titanic_modeling as tm
import Titanic_results as tr
import numpy as np
from sklearn.metrics import accuracy_score

##### Data analysis #####
#tda.complex_analysis();

##### Feature engineering #####
features, results = tfe.get_features();

##### Feature selection #####

##### Modeling #####
tm.knn_show_results(features,results);
tm.logistic_regression_show_results(features,results);
tm.svm_show_results(features,results);
tm.decission_tree_show_results(features,results);
tm.random_forest_show_results(features,results);


# Models' parameters analysis
knnResults = tm.knn_params_analysis(features,results);
logisticRegressionResults = tm.logistic_regression_params_analysis(features,results);
svmResults = tm.svm_params_analysis(features,results);
decisionTreeResults = tm.decission_tree_params_analysis(features,results);
randomForestResults = tm.random_forest_params_analysis(features,results);


##### Results #####
test_features = tr.calculate_features();
test_results = tr.calculate_results(test_features);
train_results = tr.calculate_results(features);

#train_results["Survived"] =  results

knn_score = accuracy_score(results,train_results["KNN"]);
lr_score = accuracy_score(results,train_results["Logistic Regression"]);
svc_score = accuracy_score(results,train_results["SVM"]);
dt_score = accuracy_score(results,train_results["Logistic Regression"]);
rf_score = accuracy_score(results,train_results["Random Forest"]);
ens_score = accuracy_score(results,train_results["Median"]);