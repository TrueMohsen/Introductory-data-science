# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:57:41 2023

@author: mohsenshojaeiyeganeh@gmail.com
"""

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
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

randommness = 7
test_size = 0.33

X_train,X_test,Y_train,Y_test = train_test_split(input_array,output_array,test_size = test_size , random_state = randommness )
model = LogisticRegression(solver = "liblinear")

model.fit(X_train,Y_train)
predictions = model.predict(X_test)

matrix = confusion_matrix(Y_test,predictions)
print(matrix) 
