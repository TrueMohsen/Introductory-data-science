# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:57:41 2023

@author: mohsenshojaeiyeganeh@gmail.com
"""

from pandas import read_csv
import numpy
from sklearn.ensemble import ExtraTreesClassifier 

#To read data_set
file_name = "../../Data_Set/pima-indians-diabetes.csv"
column_names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data_set = read_csv(file_name, names=column_names)

#Converting Data frame into an array
data_in_array = data_set.values

#Separating data_array into input and output arrays
input_array = data_in_array[:,0:8]
output_array = data_in_array[:,8]

model = ExtraTreesClassifier(n_estimators=100)
model.fit(input_array,output_array)
print(model.feature_importances_)