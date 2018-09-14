import pandas as pd
from sklearn.externals import joblib
import sklearn.preprocessing as preproc
import numpy as np

test_data = pd.read_csv('Data/test.csv');
features = pd.DataFrame();
picklesDir = 'Pickles';

def get_standarized_column(column):
    name = picklesDir + '/' + column + '_StandardScaler.pkl';
    scaler = joblib.load(name);
    standarizedList = scaler.transform(test_data[[column]]);
    flatStandarizedList = [x for y in standarizedList for x in y];
    return pd.Series(flatStandarizedList);

def get_minmax_scaled_column(column): 
    name = picklesDir + '/' + column + '_MinMaxScaler.pkl';
    scaler = joblib.load(name);
    scaledList = scaler.transform(test_data[[column]]);
    flatScaledList = [x for y in scaledList for x in y];
    return pd.Series(flatScaledList);

def get_labeled_column(column):
    name = picklesDir + '/' + column + '_LabelEncoder.pkl';
    encoder = joblib.load(name);
    labeledList = encoder.transform(test_data[column]);
    return pd.Series(labeledList);   

def get_one_hot_encoded_column(column):
    name = picklesDir + '/' + column + '_OneHotEncoder.pkl';
    encoder = joblib.load(name);
    oneHotEncodedList = encoder.transform(test_data[column].values.reshape(-1, 1)).toarray();
    return oneHotEncodedList;

def calculate_features():
    
    global test_data;
    global features;
    global picklesDir;
    
    # 'Age' column analysis
    # missing values can be filled by mean (median or most_frequent value) or skipped
    name = picklesDir + '/Age_Imputer.pkl';
    imputer = joblib.load(name);
    filledAgeList = imputer.transform(test_data[["Age"]]);
    flatFilledAgeList = [x for y in filledAgeList for x in y];
    test_data["FilledAge"] = pd.Series(flatFilledAgeList);
    features["Age"] = get_standarized_column("FilledAge");
    test_data = test_data.drop(columns=["FilledAge"], axis=1);
    
    # quantized 'Age'
#    test_data["QuantizedAge"] = pd.qcut(test_data["Age"],10,labels=False).fillna(10);
#    features["Age"] = get_minmax_scaled_column("QuantizedAge");
#    test_data = test_data.drop(columns=["QuantizedAge"], axis=1);
    
    
    # 'Sex' column analysis
    test_data["LabeledSex"] = get_labeled_column("Sex");
    features["Sex"] = get_minmax_scaled_column("LabeledSex");
    test_data = test_data.drop(columns=["LabeledSex"], axis=1);
    
    # 'Pclass' column analysis    
    features["Pclass"] = get_minmax_scaled_column("Pclass");
    
    # 'SibSp' column analysis
    features["SibSp"] = get_minmax_scaled_column("SibSp");
    
    # 'Parch' column analysis    
    features["Parch"] = get_minmax_scaled_column("Parch");
    
    # 'Fare' column analysis    
    imputer = preproc.Imputer();
    filledFareList = imputer.fit_transform(test_data[["Fare"]]);
    flatFilledFareList = [x for y in filledFareList for x in y];
    test_data["Fare"] = pd.Series(flatFilledFareList);
    features["Fare"] = get_standarized_column("Fare");
    
    # quantized 'Fare'
#    test_data["QuantizedFare"] = pd.qcut(test_data["Fare"],10,labels=False).fillna(10);
#    features["Age"] = get_minmax_scaled_column("QuantizedFare");
#    test_data = test_data.drop(columns=["QuantizedFare"], axis=1);
    
    # 'Embarked' column analysis
    # fill missing values with 'S' - the most frequent value 
    test_data["FilledEmbarked"] = test_data["Embarked"].fillna('S');
    test_data["LabeledEmbarked"] = get_labeled_column("FilledEmbarked");
    
    # labeledEmbarked can be scaled
    features["Embarked"] = get_minmax_scaled_column("LabeledEmbarked");
    features["Embarked"] = test_data["LabeledEmbarked"];
    
    # one hot
#    oneHotEncodedList = get_one_hot_encoded_column("LabeledEmbarked");
#    features["Embarked_C"] = pd.Series(oneHotEncodedList[:,0]);
#    features["Embarked_Q"] = pd.Series(oneHotEncodedList[:,1]);
#    features["Embarked_S"] = pd.Series(oneHotEncodedList[:,2]);
    
    test_data = test_data.drop(columns=["FilledEmbarked","LabeledEmbarked"], axis=1);
    
#    # 'Name' column analysis
#    name = test_data["Name"];
#    title = [[y for y in x.split(' ') if '.' in y][0] for x in name];
#    test_data["Title"] = pd.Series(data=title);
#    test_data["LabeledTitle"] = get_labeled_column("Title");
#    features["Title"] = get_minmax_scaled_column("LabeledTitle");
#    test_data = test_data.drop(columns=["Title","LabeledTitle"], axis=1);
    
    # 'Ticket' column analysis
        
    # 'Cabin' column analysis
    cabin = test_data["Cabin"];
    isCabin = [0 if x!=x else 1 for x in cabin];
    features["IsCabin"] = pd.Series(data=isCabin);
    
    
    return features;

def get_clf(name):
    pickleName = picklesDir + '/' + name + '_clf.pkl';
    clf = joblib.load(pickleName);
    return clf;

def calculate_results(features):
    global picklesDir;
    
    knn = get_clf("KNN");
    lr = get_clf("Logistic Regression");
    svc = get_clf("SVM");
    dt = get_clf("Decision Tree");
    rf = get_clf("Random Forest");
    
    knn_results = knn.predict(features);
    lr_results = lr.predict(features);
    svc_results = svc.predict(features);
    dt_results = dt.predict(features);
    rf_results = rf.predict(features);
    
    features["KNN"] = pd.Series(knn_results);
    features["Logistic Regression"] = pd.Series(lr_results);
    features["SVM"] = pd.Series(svc_results);
    features["Decision Tree"] = pd.Series(dt_results);
    features["Random Forest"] = pd.Series(rf_results);
    
    results = pd.DataFrame(data = [knn_results,lr_results,svc_results,dt_results,rf_results]);
    
    features["Mean"] = np.mean(results);
    features["Median"] = np.median(results,axis=0);
    
    return features;
    