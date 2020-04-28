# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:36:42 2020

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

# Tratar nan
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean", verbose=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])