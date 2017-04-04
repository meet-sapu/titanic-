# titanic-
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

To start with this a kaggle question (https://www.kaggle.com/c/titanic) . We start by importing the dataset then by reading it and pre - processing it.
We then use KNN to impute missing values in the data .
Then we create our base model with decision tree . We also apply XGboost and SVM to this model .
For feature engineering we apply PCA to the data and to this dimensionally reduced data , we apply SVM and feed the test data this model and we check the accuracy and quality of this model by confusion matrix and ROC plot .

R
As usual, we will first download our datasets locally, and then we will load them into data frames in both, R . Source of dataset : https://archive.ics.uci.edu/ml/datasets/Housing In R, we use read.csv to read CSV files into data.frame variables. Although the R function read.csv can work with URLs, https is a problem for R in many cases, so you need to use a package like RCurl to get around it. Libraries used : 1)library(readxl) #to read .xlsv file . 2)library(caTools) #for sample.split . 3)library(rpart) #for prediction() , performance() . 4)library(rpart.plot) #for plotting ROC curve . 5)library(xgboost) #for applying XGboost . 6)library(DMwR) #for applying SMOTE . 7)library(factoextra) #for PCA . 8)ibrary(e1071)#for SVM .


