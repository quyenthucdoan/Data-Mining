import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
import streamlit as st



def ReadNDrop(data):
    data = data.drop(['Unnamed: 0','id'],axis=1)
    data['Arrival Delay in Minutes'] = data['Arrival Delay in Minutes'].fillna(int(data['Arrival Delay in Minutes'].median()))
    return data
def Balance(data):
    Range = int(len(data[data['satisfaction']=='neutral or dissatisfied'])) - int(len(data[data['satisfaction']=='satisfied']))
    if Range >0:
        index = data[data['satisfaction'] == 'neutral or dissatisfied'].index.values
        index = index.tolist()
        array = random.sample(index, Range)
        data = data.drop(labels=array, axis=0)
    else:
            index = data[data['satisfaction'] == 'satisfied'].index.values
            index = index.tolist()
            array = random.sample(index, -Range)
            data = data.drop(labels=array, axis=0)
    return data
def OutlierRaw(data):
    Q3D = np.quantile(data['Departure Delay in Minutes'], 0.75)
    Q1D = np.quantile(data['Departure Delay in Minutes'], 0.25)
    IQRD = Q3D - Q1D
    stepD = IQRD * 3
    maxmD = Q3D + stepD
    data = data[data['Departure Delay in Minutes'] < maxmD]
    Q3A = np.quantile(data['Departure Delay in Minutes'], 0.75)
    Q1A = np.quantile(data['Departure Delay in Minutes'], 0.25)
    IQRA = Q3A - Q1A
    stepA = IQRA * 3
    maxmA = Q3A + stepA
    data = data[data['Departure Delay in Minutes'] < maxmA]
    return data,maxmA,maxmD
def OutlierImport(data,maxmA,maxmD):
    data = data[data['Arrival Delay in Minutes'] < maxmA]
    data = data[data['Departure Delay in Minutes'] < maxmD]
    return data
def Tranform(data):
    mm = MinMaxScaler()
    data = pd.get_dummies(data,columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'])
    X = data.drop('satisfaction',axis=1)
    Y = data['satisfaction']
    X[X.columns] = mm.fit_transform(X[X.columns])
    return X,Y,data
def TranformImport(data):
    mm = MinMaxScaler()
    data = pd.get_dummies(data,columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'])
    data[data.columns] = mm.fit_transform(data[data.columns])
    return data
def PCAImport(data):
    pca = PCA(n_components=23)
    data = pca.fit_transform(data)
    return data

def PreprocessCluster():
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")
    data = pd.concat([train,test])
    data = data.dropna()
    data.drop(['Unnamed: 0','id'], axis=1, inplace=True)
    data.drop(['Inflight wifi service','Ease of Online booking', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1, inplace=True)
    data.select_dtypes(exclude=['int64', 'float64']).columns
    data_onehot = pd.get_dummies(data, columns=data.select_dtypes(exclude=['int64', 'float64']).columns)
    data.drop(['Gate location','Departure/Arrival time convenient', 'Gender'], axis=1, inplace=True)
    return data;


