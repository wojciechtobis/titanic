# Titanic
Kaggle project for beginners. Binary classification problem. As a result it would be possible to predict if passenger survived disaster or not.

# Required steps

## Collecting data

Files with train and test data are stored in 'Data' directory.

## Data analysis

Selected columns, e.g. 'Age', 'Sex' etc should be analysed. 

## Feature engineering

Depending on results from data analysis, selected columns shuld be modified. Expected modification:

### Dealing with missing values

Rows with missing values should be skipped or these fields should be filled (e.g. mean interpolation)

### Tokenization

Required for non-numeric values

### Quantization/Binning

Probably it would be helpful for 'Age' or 'Fare'

### Power Transforms

Log or Box-Cox transformation for distribution 'imporvement'

### Scaling/Normalisation

Depending on data analysis and columns distribution, data should be scaled or normalised

## Feature selection

Select just a few the most important features.

## Modeling 

Models used for classification:
* KNN
* Logistic Regression
* SVM
* Decision Tree
* Random Forest

## Charting

This problem-solving results would be accuracy and f-score for testing data. These measures would be calculated after every step of feature engineering, feature selection and model tuning to show improvement.