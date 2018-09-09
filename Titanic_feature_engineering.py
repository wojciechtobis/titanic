import pandas as pd

##### Collecting data #####

# import data
train_data = pd.read_csv('Data/train.csv');
features =pd.DataFrame();

def getFeatures():
    global train_data;
    global features;
    
    print("Features");