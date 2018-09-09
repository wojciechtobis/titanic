import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

##### Collecting data #####

# import data
train_data = pd.read_csv('Data/train.csv');

##### Data analysis #####

def continuous_column_analysis(column):
    survived = train_data.loc[train_data["Survived"]==1][column].dropna();
    dead = train_data.loc[train_data["Survived"]==0][column].dropna();
    plt.hist(survived,bins=25,alpha=0.5,label=column+' for survived');
    plt.hist(dead,bins=25,alpha=0.5,label=column+' for dead');
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05));
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
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05));
    plt.title("Stacked classes for '"+column+"'");
    plt.show();

def stacked_bar_class(column):
    # unique values
    uniqueValues = train_data[column].dropna().unique();
    uniqueValues.sort();
    
    #initial marginBottom
    marginBottom = pd.Series(data=[0,0])
    
    #chart colors
    colors = ["#006D2C","#31A354","#74C476","#556D2C","#55A354","#55C476","#DD6D2C","#DDA354","#DDC476",
              "#006D00","#31A300","#74C400","#556D55","#55A355","#55C455","#DD6DDD","#DDA3DD","#DDC4DD"];

    # stacked bar chart    
    for num, value in enumerate(uniqueValues):
        col = train_data.loc[train_data[column]==value]["Survived"].dropna();
        colValues = col.value_counts().reindex([0,1]).fillna(0);
        colValues.plot.bar(bottom=marginBottom,color=colors[num],stacked=True,label=value);
        marginBottom += colValues;
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05));
    plt.title("Stacked '"+column+"' values for classes");
    plt.show();

def values_percentage(column):
    values = train_data[column].dropna();
    valuesCount = len(values);
    
    # unique values
    uniqueValues = values.unique();
    uniqueValues.sort();
    
    # empty series for unique values
    baseSeries = pd.Series(data=np.zeros(len(uniqueValues)), index=uniqueValues);
    
    survived = train_data.loc[train_data["Survived"]==1][column].dropna();
    dead = train_data.loc[train_data["Survived"]==0][column].dropna();
    
    survivedCounts = pd.concat([baseSeries,survived.value_counts()], axis=1)[column].fillna(0);
    deadCounts = pd.concat([baseSeries,dead.value_counts()], axis=1)[column].fillna(0);
    
    table = pd.concat([
                survivedCounts.rename('Survived'),
                deadCounts.rename('Dead')
            ], axis=1) / valuesCount;
    
    sns.heatmap(table, annot=True, cmap="Greens");
    plt.show();

def quantized_column_analysis(column):
    stacked_bar_columns(column);
    stacked_bar_class(column);
    values_percentage(column);    

def complex_analysis():
    
    global train_data
    
    # missing values analysis
    missingvalues = train_data.isna().sum();
    missingvalues = missingvalues.sort_values(ascending=False);
    print("missing values count in each column")
    print(missingvalues);
    
    # 'Age' column analysis
    continuous_column_analysis("Age");
    
    # 'Sex' column analysis
    quantized_column_analysis("Sex");
    
    # 'Pclass' column analysis
    quantized_column_analysis("Pclass");
    
    # 'SibSp' column analysis
    quantized_column_analysis("SibSp");
    
    # 'Parch' column analysis
    quantized_column_analysis("Parch");
    
    # 'Fare' column analysis
    continuous_column_analysis("Fare");
    
    # 'Embarked' column analysis
    quantized_column_analysis("Embarked");
    
    # 'Name' column analysis
    name = train_data["Name"];
    
    # extract title from name
    title = [[y for y in x.split(' ') if '.' in y][0] for x in name];
    
    # titles analysis
    train_data["Title"] = pd.Series(data=title);
    quantized_column_analysis("Title");    
    train_data = train_data.drop(columns=["Title"], axis=1);
    
    # 'Ticket' column analysis
        
    # 'Cabin' column analysis
    cabin = train_data["Cabin"];
    
    # check if cabin exists
    isCabin = [0 if x!=x else 1 for x in cabin];
    
    # isCabin analysis
    train_data["IsCabin"] = pd.Series(data=isCabin)
    quantized_column_analysis("IsCabin");    
    train_data = train_data.drop(columns=["IsCabin"], axis=1);