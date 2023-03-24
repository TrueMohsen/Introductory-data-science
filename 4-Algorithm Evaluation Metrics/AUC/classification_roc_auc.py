# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:57:41 2023

@author: mohsenshojaeiyeganeh@gmail.com
"""

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#To read data_set
file_name = "../../Data_Set/pima-indians-diabetes.csv"
column_names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data_set = read_csv(file_name, names=column_names)

#Converting Data frame into an array
data_in_array = data_set.values

#Separating data_array into input and output arrays
input_array = data_in_array[:,0:8]
output_array = data_in_array[:,8]

number_of_splits = 10
randommness = 7

kfold = KFold(n_splits = number_of_splits, random_state = randommness)
model = LogisticRegression(solver = "liblinear")
scoring = 'roc_auc'

results = cross_val_score(model, input_array,output_array,cv = kfold, scoring = scoring)
print("AUC: %.3f (%.3f)" %(results.mean(),results.std())) 
