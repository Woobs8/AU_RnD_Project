# -*- coding: utf-8 -*-

import numpy as np
from Classifiers.Classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier

class KNearestNeighbors(Classifier):
    def __init__(self, k=1, weights='uniform', n_jobs=1,metric='minkowski',name='KNN'):
        super().__init__(name)
        self.k = k
        self.weights = weights
        self.n_jobs = n_jobs
        self.metric = metric
        self.clf = KNeighborsClassifier(k, weights=weights, n_jobs=n_jobs, metric=metric)

    def fit(self, data, labels):
        self.clf.fit(data, labels)

    def predict(self, data):
        return self.clf.predict(data)


