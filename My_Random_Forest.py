# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:29:29 2024

@author: azarf
"""
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(n_estimators=50,max_depth=4, random_state = 100)