# -*- coding: utf-8 -*-
import numpy as np
from Classifiers.Classifier import Classifier
from sklearn.naive_bayes import  GaussianNB as NB

class NaiveBayes(Classifier):
    """k Nearest Neighbors classifier.
    Each class is represented by a set of samples that it contains. New samples is classified to the class 
    that the majority of the k nearest samples belongs to.

    Parameters
    ----------
    batch_size : int,
        is the ...
    distance_metric : string
        is the ...

    Attributes
    ----------
    centroids : array-like, shape = [n_classes, n_subclasses, n_features]
    Centroid of each subclass
    """

    def __init__(self,name='NB'):
        super().__init__(name)
        self.nb = NB()


    def fit(self, data, labels):
        """
        Store the input samples as they are the parameters for classifying with the kNN
        ----------
        X_vals : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
            Note that centroid shrinking cannot be used with sparse matrices.
        y_vals : array, shape = [n_samples]
            Target values (integers)
        """
        self.nb.fit(data, labels.ravel())

        return self


    def predict(self, data):
        """Perform classification on an array of test vectors X.
        The predicted class C for each sample in X is returned.
        Parameters
        ----------
        X_vals_test : array-like, shape = [n_samples, n_features]
        y_vals_test : array, shape = [n_samples]
            Target values (integers)

        Returns
        -------
        C : array, shape = [n_samples]
        """        
        return self.nb.predict(data)

