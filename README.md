# Pregnancy-Report-Strati-fication

## Welcome to the Repository!

### Installation

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included. 

### Exploratory Data Analysis

Exploratory data analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task.

**Data**

The pregnancy report dataset consists of 6262 data points, with each datapoint having 135 features.

**Observations**

-Dataset have features with following datatypes(No.of columns) : datetime64[ns](12), float64(62), int64(16), object(45)

-There are 32 empty features which we drop from the dataset for further analysis.After dropping them,we get a dataset with 6262 data points with each having 103 features.

-Plotted Heatmap of missing values,box plot and histogram plots of all the features to analyze them. 
These plots can be referred from [EDA_plots](./EDA_Plots)

### Data Preprocessing

- Loading the Dataset: Load Dataset in the from of a excel file.
  
- Imputing the Dataset : Imputed missing values in numerical features and categorical features seperately using required imputer in the dataset.
  
- Handling Date columns : There are some features having datatype datetime64[ns] present in the dataset 
  which needed to be converted into categorical features for further training.
  
- Encoding Categorical Columns : Categorical columns must be encoded into numerical ones for the 
  training . Therefore multiple differnt encoders including One Hot Encoder, Gap Encoder,Hash 
  Encoder and some others are used.
  
- Scaling the Dataset : To ensure that no single feature dominates the distance calculations in an algorithm, and to improve the performance of the algorithm, dataset is scaled before splitting the dataset.

-Splitting the dataset : Dataset is splitted into 3 sets- training set, validation set, and test set int the ratio(%) 80 : 10: :10.

### Training the datset

Training the dataset includes Fitting data.

 Various algorithms are used to train the model to find best accuracy model :

 - Support Vector Machine
 - DecisionTreeClassifier
 - GaussianNB
 - CatBoostClassifier
 - RandomForestClassifier
 - XGBClassifier
 - KNeighborsClassifier

### Evaluation

Evaluating trained against for a test set and got sensitivity of 0.70 and specificity of 0.80 for xgb classifier with catboost encoding
Also plotted all the important curves like Precison Curve and ROC Curve for all the classifiers and stored the results including accuracy ,sensitivity,specificity,prevalence,Classification Report,Confusion Matrix for all the classifiers and different encoding techniques.
To check out results and plots:[Click Here]{./Results}


 




