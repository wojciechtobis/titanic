import Titanic_data_analysis as tda
import Titanic_feature_engineering as tfe
import Titanic_modeling as tm

##### Data analysis #####
#tda.complex_analysis();

##### Feature engineering #####
features, results = tfe.get_features();

##### Feature selection #####

##### Modeling #####
tm.knn_results(features,results);
tm.logistic_regression_results(features,results);
tm.svm_results(features,results);
tm.decission_tree_results(features,results);
tm.random_forest_results(features,results);