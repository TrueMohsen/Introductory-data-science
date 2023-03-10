# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:57:14 2023

@author: mohsen
"""

from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix
import numpy


# 1.preg = number of times of pregnancy
# 2.plas = plasma glucose concentration 
# 3.pres = diastolic blood pressure (mm hg)
# 4.skin = triceps skin fold thickness(mm)
# 5.test = 2-hour serum insulin (mu U/ml)
# 6.mass = body mass index (weight in kg /(height in m)^2)
# 7.pedi = diabetes pedigree function
# 8.age = age(years)
# 9.class = 0-> No diabetes 1->diabetes

#To read data_set
file_name = "../../Data_Set/pima-indians-diabetes.csv"
column_names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data_set = read_csv(file_name, names=column_names)


#To draw correlation plot
scatter_matrix(data_set)
pyplot.show()