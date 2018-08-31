# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:36:43 2018

@author: s1883483
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.neighbors import LSHForest
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.neighbors import KNeighborsClassifier

#%% load data, preprocessing, normalization

stock_price = pd.read_pickle(r"C:\Users\s1883483\Desktop\Advanced analytics projects\coding cafe\prices.pkl")
stock_price.drop('Currency', axis=1, inplace=True)
stock_price.head(10)

# forward fill nan first, then back fill nan to keep the trend

def normalize_stock(stock_df):
    
    stock_df.fillna(method='ffill', axis=1, inplace=True)
    stock_df.fillna(method='bfill', axis=1, inplace=True)
    stock_df = stock_df.iloc[:, 1:].div(stock_df.iloc[:, 0], axis=0)
    return stock_df

stock_price_trend = normalize_stock(stock_df)

#%%