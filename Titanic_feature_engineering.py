import pandas as pd
import sklearn.preprocessing as preproc
from sklearn.externals import joblib

##### Collecting data #####

# import data
train_data = pd.read_csv('Data/train.csv');
features = pd.DataFrame();
picklesDir = 'Pickles';

def get_standarized_column(column):
    scaler = preproc.StandardScaler();
    standarizedList = scaler.fit_transform(train_data[[column]]);
    flatStandarizedList = [x for y in standarizedList for x in y];
    name = picklesDir + '/' + column + '_StandardScaler.pkl';
    joblib.dump(scaler,name);
    return pd.Series(flatStandarizedList);

def get_minmax_scaled_column(column): 
    scaler = preproc.MinMaxScaler();
    scaledList = scaler.fit_transform(train_data[[column]]);
    flatScaledList = [x for y in scaledList for x in y];
    name = picklesDir + '/' + column + '_MinMaxScaler.pkl';
    joblib.dump(scaler,name);
    return pd.Series(flatScaledList);

def get_labeled_column(column):
    encoder = preproc.LabelEncoder();
    labeledList = encoder.fit_transform(train_data[column]);
    name = picklesDir + '/' + column + '_LabelEncoder.pkl';
    joblib.dump(encoder,name);
    return pd.Series(labeledList);   

def get_one_hot_encoded_column(column):
    encoder = preproc.OneHotEncoder();
    oneHotEncodedList = encoder.fit_transform(train_data[column].values.reshape(-1, 1)).toarray();
    name = picklesDir + '/' + column + '_OneHotEncoder.pkl';
    joblib.dump(encoder,name);
    return oneHotEncodedList;

def get_features():
    global train_data;
    global features;
    global picklesDir;
        
    # 'Age' column analysis
    # missing values can be filled by mean (median or most_frequent value) or skipped
    imputer = preproc.Imputer();
    filledAgeList = imputer.fit_transform(train_data[["Age"]]);
    flatFilledAgeList = [x for y in filledAgeList for x in y];
    train_data["FilledAge"] = pd.Series(flatFilledAgeList);
    features["Age"] = get_standarized_column("FilledAge");
    train_data = train_data.drop(columns=["FilledAge"], axis=1);
    name = picklesDir + '/Age_Imputer.pkl';
    joblib.dump(imputer,name);
    
    # quantized 'Age'
#    train_data["QuantizedAge"] = pd.qcut(train_data["Age"],10,labels=False).fillna(10);
#    features["Age"] = get_minmax_scaled_column("QuantizedAge");
#    train_data = train_data.drop(columns=["QuantizedAge"], axis=1);
    
    
    # 'Sex' column analysis
    train_data["LabeledSex"] = get_labeled_column("Sex");
    features["Sex"] = get_minmax_scaled_column("LabeledSex");
    train_data = train_data.drop(columns=["LabeledSex"], axis=1);
    
    # 'Pclass' column analysis    
    features["Pclass"] = get_minmax_scaled_column("Pclass");
    
    # 'SibSp' column analysis
    features["SibSp"] = get_minmax_scaled_column("SibSp");
    
    # 'Parch' column analysis    
    features["Parch"] = get_minmax_scaled_column("Parch");
    
    # 'Fare' column analysis    
    features["Fare"] = get_standarized_column("Fare");
    
    # quantized 'Fare'
#    train_data["QuantizedFare"] = pd.qcut(train_data["Fare"],10,labels=False).fillna(10);
#    features["Age"] = get_minmax_scaled_column("QuantizedFare");
#    train_data = train_data.drop(columns=["QuantizedFare"], axis=1);
    
    # 'Embarked' column analysis
    # fill missing values with 'S' - the most frequent value 
    train_data["FilledEmbarked"] = train_data["Embarked"].fillna('S');
    train_data["LabeledEmbarked"] = get_labeled_column("FilledEmbarked");
    
    # labeledEmbarked can be scaled
    features["Embarked"] = get_minmax_scaled_column("LabeledEmbarked");
    features["Embarked"] = train_data["LabeledEmbarked"];
    
    # one hot
#    oneHotEncodedList = get_one_hot_encoded_column("LabeledEmbarked");
#    features["Embarked_C"] = pd.Series(oneHotEncodedList[:,0]);
#    features["Embarked_Q"] = pd.Series(oneHotEncodedList[:,1]);
#    features["Embarked_S"] = pd.Series(oneHotEncodedList[:,2]);
    
    train_data = train_data.drop(columns=["FilledEmbarked","LabeledEmbarked"], axis=1);
    
#    # 'Name' column analysis
#    name = train_data["Name"];
#    title = [[y for y in x.split(' ') if '.' in y][0] for x in name];
#    train_data["Title"] = pd.Series(data=title);
#    train_data["LabeledTitle"] = get_labeled_column("Title");
#    features["Title"] = get_minmax_scaled_column("LabeledTitle");
#    train_data = train_data.drop(columns=["Title","LabeledTitle"], axis=1);
    
    # 'Ticket' column analysis
        
    # 'Cabin' column analysis
    cabin = train_data["Cabin"];
    isCabin = [0 if x!=x else 1 for x in cabin];
    features["IsCabin"] = pd.Series(data=isCabin);

    return features, train_data["Survived"];
    