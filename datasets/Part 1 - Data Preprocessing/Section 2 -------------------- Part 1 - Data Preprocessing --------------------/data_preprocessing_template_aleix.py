# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:16:53 2020

@author: Sancho
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importar data set
dataset = pd.read_csv('Data.csv')

# Variable x hace referencia a las variables independientes
X = dataset.iloc[:, :-1].values

# Variable y hace referencia a las variables dependientes (variables a predecir)
y = dataset.iloc[:, -1].values

# Dividir el dataset en entrenamiento y testing (80/20) 20,25,30 de testing como mucho
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


'''
# Escalado de variables
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''