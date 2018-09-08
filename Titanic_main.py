import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##### Collecting data #####

# import data
train_data = pd.read_csv('Data/train.csv');
test_data = pd.read_csv('Data/test.csv');
test_results = pd.read_csv('Data/gender_submission.csv');

# column names
columns = list(train_data);


##### Data analysis #####

## missing values analysis
#missingvalues = train_data.isna().sum();
#missingvalues = missingvalues.sort_values(ascending=False);
#print("missing values count in each column")
#print(missingvalues);

def column_analysis(column):
    survived = train_data.loc[train_data["Survived"]==1][column].dropna();
    dead = train_data.loc[train_data["Survived"]==0][column].dropna();
    plt.hist(survived,bins=25,alpha=0.5,label=column+' for survived');
    plt.hist(dead,bins=25,alpha=0.5,label=column+' for dead');
    plt.legend(loc='upper right');
    plt.title("Histograms for '"+column+"'");
    plt.show();

def stacked_bar_columns(column):
    # unique values
    uniqueValues = train_data[column].dropna().unique();
    uniqueValues.sort();
    
    # empty series for unique values
    baseSeries = pd.Series(data=np.zeros(len(uniqueValues)), index=uniqueValues);
    
    # column separated for survived and dead 
    survived = train_data.loc[train_data["Survived"]==1][column].dropna();
    dead = train_data.loc[train_data["Survived"]==0][column].dropna();
    
    # column value counts
    survivedCounts = pd.concat([baseSeries,survived.value_counts()], axis=1)[column].fillna(0);
    deadCounts = pd.concat([baseSeries,dead.value_counts()], axis=1)[column].fillna(0);
    
    # stacked bar chart 
    survivedCounts.plot.bar(color="#006D2C",label="Survived");
    deadCounts.plot.bar(bottom=survivedCounts,color="#31A354",stacked=True,label="Dead");
    plt.legend(loc='upper right');
    plt.title("Stacked classes for '"+column+"'");
    plt.show();

def stacked_bar_class(column):
    # unique values
    uniqueValues = train_data[column].dropna().unique();
    uniqueValues.sort();
    
    #initial marginBottom
    marginBottom = pd.Series(data=[0,0])
    
    #chart colors
    colors = ["#006D2C","#31A354","#74C476","#556D2C","#55A354","#55C476","#DD6D2C","#DDA354","#DDC476"];

    # stacked bar chart    
    for num, value in enumerate(uniqueValues):
        col = train_data.loc[train_data[column]==value]["Survived"].dropna();
        colValues = col.value_counts().reindex([0,1]);
        colValues.plot.bar(bottom=marginBottom,color=colors[num],stacked=True,label=value);
        marginBottom += colValues;
    
    plt.legend(loc='upper right');
    plt.title("Stacked '"+column+"' values for classes");
    plt.show();

# 'Age' column analysis
column_analysis("Age");

# 'Sex' column analysis
stacked_bar_columns("Sex");
stacked_bar_class("Sex");

# 'Pclass' column analysis
stacked_bar_columns("Pclass");
stacked_bar_class("Pclass");

# 'SibSp' column analysis
stacked_bar_columns("SibSp");
stacked_bar_class("SibSp");

# 'Parch' column analysis
stacked_bar_columns("Parch");
stacked_bar_class("Parch");

# 'Fare' column analysis
column_analysis("Fare");

# 'Embarked' column analysis
stacked_bar_columns("Embarked");
stacked_bar_class("Embarked");
