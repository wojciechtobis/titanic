import pandas as pd

##### Collecting data #####

# import data
train_data = pd.read_csv('Data/train.csv');
test_data = pd.read_csv('Data/test.csv');
test_results = pd.read_csv('Data/gender_submission.csv');

# column names
columns = list(train_data);




