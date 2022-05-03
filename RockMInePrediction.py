# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:57:53 2022

@author: DELL
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


sonar_dataset = pd.read_csv('Copy of sonar data.csv', header = None)

x = sonar_dataset.drop(columns = 60, axis = 1)
y = sonar_dataset[60]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, stratify = y, random_state = 1)

model = LogisticRegression()
model.fit(x_train, y_train)

x_train_predicted = model.predict(x_train)
x_train_accuracy = accuracy_score(x_train_predicted, y_train)


x_test_predicted = model.predict(x_test)
x_test_accuracy  = accuracy_score(x_test_predicted, y_test)

input_data = (0.0115,0.0150,0.0136,0.0076,0.0211,0.1058,0.1023,0.0440,0.0931,0.0734,0.0740,0.0622,0.1055,0.1183,0.1721,0.2584,0.3232,0.3817,0.4243,0.4217,0.4449,0.4075,0.3306,0.4012,0.4466,0.5218,0.7552,0.9503,1.0000,0.9084,0.8283,0.7571,0.7262,0.6152,0.5680,0.5757,0.5324,0.3672,0.1669,0.0866,0.0646,0.1891,0.2683,0.2887,0.2341,0.1668,0.1015,0.1195,0.0704,0.0167,0.0107,0.0091,0.0016,0.0084,0.0064,0.0026,0.0029,0.0037,0.0070,0.0041)
input_data_as_array = np.asarray(input_data)
input_data_reshaped = input_data_as_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)



