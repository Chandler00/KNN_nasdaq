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
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import LSHForest
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.neighbors import KNeighborsClassifier

#%% load data, preprocessing, normalization

stock_price = pd.read_pickle(r"C:\Users\s1883483\Desktop\Advanced analytics projects\coding cafe\prices.pkl")
stock_price.drop('Currency', axis=1, inplace=True)
stock_price.head(10)

#%% preprocessing functions
# filter data frame based on date in the "yyyy-mm-dd" format

def date_range(date_start, date_end,stock_df):
    
    return stock_df.loc[:, pd.to_datetime(date_start):pd.to_datetime(date_end)]

# forward fill nan first, then back fill nan to keep the trend.
    
def normalize_stock(stock_df):
    
    stock_df.fillna(method='ffill', axis=1, inplace=True)
    stock_df.fillna(method='bfill', axis=1, inplace=True)
    stock_df = stock_df.iloc[:, 1:].div(stock_df.iloc[:, 0], axis=0)
    return stock_df

#%% analysis functions

class stock_analysis:
    
    def __init__(self, stock_df, date_start, date_end, tar_equity, top_k):
        self.stock_df = stock_df
        self.date_start = date_start
        self.date_end = date_end
        self.tar_equity = tar_equity
        self.top_k = top_k
        
    def plot_corr(self):
        stock_df = normalize_stock(self.stock_df)
        stock_df = date_range(self.date_start, self.date_end, stock_df)
        corr = stock_df.T.corr()
        corr_top_index = corr[self.tar_equity].sort_values(ascending=False)[:self.top_k].index.tolist()
        corr_top = stock_df.T[corr_top_index ].corr()
        plt.figure(figsize=(16, 16))
        corr_map = sns.heatmap(corr_top, xticklabels=corr_top.columns, yticklabels=corr_top.columns, annot=True)
        figure = corr_map.get_figure()    
        figure.savefig(r'C:\Users\s1883483\Desktop\Advanced analytics projects\coding cafe\KNN output\stock_corr.png', dpi=800)
        


# define KNN algorithms for use

class KNN_algorithms:
    
    def __init__(self, KNN_alg):
        self.KNN_alg = KNN_alg
        
        
        
#%%
# using APPL.US stock as sample to check correlation heat map
test = stock_analysis(normalize_stock(stock_price), "2017-01-01", "2018-01-01", "AAPL.US", 20)
test.plot_corr()

#%%