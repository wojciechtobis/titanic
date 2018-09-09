import pandas as pd
import sklearn.preprocessing as preproc

##### Collecting data #####

# import data
train_data = pd.read_csv('Data/train.csv');
features =pd.DataFrame();

def get_standarized_column(column):
    standarizedList = preproc.StandardScaler().fit_transform(train_data[[column]]);
    flatStandarizedList = [x for y in standarizedList for x in y];
    return pd.Series(flatStandarizedList);

def get_minmax_scaled_column(column):    
    scaledList = preproc.MinMaxScaler().fit_transform(train_data[[column]]);
    flatScaledList = [x for y in scaledList for x in y];
    return pd.Series(flatScaledList);

def get_labeled_column(column):
    labeledList = preproc.LabelEncoder().fit_transform(train_data[column]);
    return pd.Series(labeledList);   

def get_features():
    global train_data;
    global features;
        
    # 'Age' column analysis
    # missing values can be filled by mean (median or most_frequent value) or skipped
    filledAgeList = preproc.Imputer().fit_transform(train_data[["Age"]]);
    flatFilledAgeList = [x for y in filledAgeList for x in y];
    train_data["FilledAge"] = pd.Series(flatFilledAgeList);
    features["Age"] = get_standarized_column("FilledAge");
    train_data = train_data.drop(columns=["FilledAge"], axis=1);
    
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
    
    # 'Embarked' column analysis
    # fill missing values with 'S' - the most frequent value 
    train_data["FilledEmbarked"] = train_data["Embarked"].fillna('S');
    train_data["LabeledEmbarked"] = get_labeled_column("FilledEmbarked");
    features["Embarked"] = get_minmax_scaled_column("LabeledEmbarked");
    train_data = train_data.drop(columns=["FilledEmbarked","LabeledEmbarked"], axis=1);
    
    # 'Name' column analysis
    name = train_data["Name"];
    title = [[y for y in x.split(' ') if '.' in y][0] for x in name];
    train_data["Title"] = pd.Series(data=title);
    train_data["LabeledTitle"] = get_labeled_column("Title");
    features["Title"] = get_minmax_scaled_column("LabeledTitle");
    train_data = train_data.drop(columns=["Title","LabeledTitle"], axis=1);
    
    # 'Ticket' column analysis
        
    # 'Cabin' column analysis
    cabin = train_data["Cabin"];
    isCabin = [0 if x!=x else 1 for x in cabin];
    features["IsCabin"] = pd.Series(data=isCabin);

    return features, train_data["Survived"];
    