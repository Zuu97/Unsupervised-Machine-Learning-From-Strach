from variables import *
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

def read_csv_files(csv_file):
        data = pd.read_csv(csv_file)
        data = data.dropna(axis = 0, how ='any')
        Y = data.iloc[:,0]
        X = data.iloc[:,1:] / 255.0
        X,Y = shuffle(X,Y)
        return X, Y

def get_data():
    Xtrain, Ytrain = read_csv_files(train_path)
    Xtest , Ytest  = read_csv_files(test_path)
    Xtrain = Xtrain[:1000]
    Ytrain = Ytrain[:1000]
    Xtest  = Xtest[:300]
    Ytest  = Ytest[:300]
    print("Data is ready")
    return Xtrain, Ytrain, Xtest , Ytest

def dim_reduction(Xtrain):
    pca = PCA(n_components=2)
    pca.fit(Xtrain)
    return pca
