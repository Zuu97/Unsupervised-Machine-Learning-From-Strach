from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def get_data():
    df = pd.read_csv('iris.csv')
    data = df.dropna(axis = 0, how ='any')
    data = shuffle(data)
    X , Y = data.iloc[:,:-1].to_numpy(), data.iloc[:,-1].to_numpy()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.2, random_state=42)
    return Xtrain, Xtest, Ytrain, Ytest

def dim_reduction(Xtrain):
    pca = PCA(n_components=2)
    pca.fit(Xtrain)
    return pca
