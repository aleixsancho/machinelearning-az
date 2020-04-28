# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:26:23 2020

@author: Sancho
"""

# Regresión lineal simple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Importar data set
dataset = pd.read_csv('Salary_Data.csv')

# Variable x hace referencia a las variables independientes
X = dataset.iloc[:, :-1].values

# Variable y hace referencia a las variables dependientes (variables a predecir)
y = dataset.iloc[:, -1].values

# Dividir el dataset en entrenamiento y testing (80/20) 20,25,30 de testing como mucho
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

'''
# Escalado de variables
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

# Crear modelo Regresión linal simple con el train
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el test
y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)')
plt.xlabel('Años de Experiencia')
plt.ylabel('Sueldo (en $)')
plt.show()

# Visualizar los resultados de test
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)')
plt.xlabel('Años de Experiencia')
plt.ylabel('Sueldo (en $)')
plt.show()
