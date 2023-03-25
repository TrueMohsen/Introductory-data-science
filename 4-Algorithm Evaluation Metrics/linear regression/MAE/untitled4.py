# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:34:40 2023

@author: mohsen
"""

from pandas import read_csv 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

#To read data_set
file_name = "../../../Data_Set/BostonHousing_withoutcomments.csv"
column_names = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat','medv']
data_set = read_csv(file_name, names=column_names)

print(data_set.shape)

#Converting Data frame into an array
data_in_array = data_set.values

#Separating data_array into input and output arrays
input_array = data_in_array[:,0:13]
output_array = data_in_array[:,13]


kfold = KFold(n_splits = 10, random_state = 7)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, input_array,output_array , cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)"%(results.mean(),results.std()))