# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:57:41 2023

@author: mohsenshojaeiyeganeh@gmail.com
"""

from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
import numpy

#To read data_set
file_name = "../../Data_Set/pima-indians-diabetes.csv"
column_names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data_set = read_csv(file_name, names=column_names)

#Converting Data frame into an array
data_in_array = data_set.values

#Separating data_array into input and output arrays
input_array = data_in_array[:,0:8]
output_array = data_in_array[:,8]

scaler = MinMaxScaler(feature_range=(0, 1))

rescaled_input_array = scaler.fit_transform(input_array)


set_printoptions(precision=3)
print(input_array)
print(rescaled_input_array)

