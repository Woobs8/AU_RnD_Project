import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from itertools import chain

from functools import reduce

def ScaleData(Data):
    Data = Data-np.min(Data,axis=0)
    return Data / np.max(Data)

def CenterData(Data):
    Data = Data-np.mean(Data,axis=0)
    return Data

def ScaleFeatures(Data):
    mmscaler = MinMaxScaler()
    mmscaler.fit(Data)
    return mmscaler.transform(Data)

def PcaRotation(Data):
    pca = PCA(n_components=3, whiten=False,)
    pca.fit(Data)
    return pca.transform(Data)

