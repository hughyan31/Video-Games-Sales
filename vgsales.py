# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 18:49:10 2023

@author: hughy
"""
import numpy as np
import pandas as pd
import random
np.random.seed(24)

# Load the data
data = pd.read_csv("vgsales.csv")

# EDA
print(data.head())
# Check for missing data
print(data.isnull().sum())
# Check for unique value
print(data.nunique())