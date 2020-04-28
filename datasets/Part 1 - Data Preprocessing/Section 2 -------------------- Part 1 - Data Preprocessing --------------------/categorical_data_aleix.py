# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:38:31 2020

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

# Codificar datos categóricos con label encoder
from sklearn import preprocessing

labelencoder_x = preprocessing.LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:,0])

y = labelencoder_x.fit_transform(y)

# Codificar datos categóricos con onehotencoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories = 'auto'), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)

