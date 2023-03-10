# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:28:16 2023

@author: mohsenshojaeiyeganeh@gmail.com
"""


from matplotlib import pyplot
from pandas import read_csv
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
column_names = ['pregnancy','plasma','pres','skin','test','mass','pedi','age','class']
data_set = read_csv(file_name, names=column_names)

#To calculate correlation between data
data_correlation = data_set.corr()

#To draw correlation plot
figure = pyplot.figure("Correlation_matrix_plot")
pane = figure.add_subplot(111)
outer = pane.matshow(data_correlation, vmin=-1, vmax=1)
figure.colorbar(outer)
ticks = numpy.arange(0,9,1)
pane.set_xticks(ticks)
pane.set_yticks(ticks)
pane.set_xticklabels(column_names)
pane.set_yticklabels(column_names)
pyplot.show()



