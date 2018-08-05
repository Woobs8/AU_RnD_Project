import numpy as np
from sklearn.svm import SVC, LinearSVC
from Classifiers.Classifier import Classifier


class SupportVectorMachine(Classifier):
    """
    Inspired by: https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines
    """
    def __init__(self, kernel='linear', gamma='auto', max_iter=100, coef0=0, tol=1e-3, degree=3,C=1,name='SVM'):
        """
        Linear SVM:
        alpha is called the "soft-margin" term and is a regularization parameter, which can be increased 
        to allow for more erroneous classification and prevent overfitting. Setting alpha to zero implies hard-margin.

        Non-Linear SVM (kernel method):
        Not all classification problems are linearly separable in the original "raw" feature space. A solution to this
        problem is transforming the "raw" data points to a higher-dimensional feature space where the data points may be 
        linearly separable. However, this can be a computationally expensive operation.

        Kernel method utilize similarity functions called "kernels", which measures similarity in data points. In this sense, 
        kernel methods can be considered "instance-based learners", which mean they do not explicitly try to generalize, 
        but instead compare new problem instances with problems encountered in training.
        A kernel method therefore does not learn weights for each of the features in the inputs, but instead "remembers"
        each (xi, yi) training pair and learns their contribution with a corresponding weight.

        Applying kernels as described above, allows a kernel method classifier to operate in a high-dimensional
        implicit feature space without performing the actual computationally expensive mapping. This is called the
        "kernel trick". A function K(x,x') in the input space X may be expressed as an inner product in another space V.
        The inner product is a measure of similarity, and this can therefore be utilized to perform the similarity measure on 
        data implicitly mapped to another feature space using only a kernel function. 
        
        It is therefore given that for a feature map phi, which maps X -> V, there is a kernel function, which satisfies:
            K(x,x')=〈phi(x), phi(x')〉
        Phi does not need to be explicitly represented, so long as V is an inner product space (meaning the inner product 
        operation is defined for that space).

        Constructing additional features may not be necessary when applying a kernel method such as a SVM, and kernel methods
        do not scale well to high-dimensional feature space. 
        """
        super().__init__(name)

        # Hyperparameters
        self.kernel = kernel
        self.max_iter = max_iter
        self.gamma=gamma
        self.coef0 = coef0
        self.tol = tol
        self.degree = degree
        self.dual = True
        self.C = C

        # Data parameters
        self.num_features = None
        self.num_samples = None

        # Classifier
        if kernel=='linear':
            self.clf=LinearSVC(max_iter=self.max_iter,tol=self.tol)
        elif kernel=='rbf' or kernel=='poly' or kernel=='sigmoid':
            self.clf=SVC(kernel=self.kernel,degree=self.degree,gamma=self.gamma,max_iter=self.max_iter, coef0=self.coef0, tol=self.tol,C=self.C)
        else:
            print(str(self.__class__)+" Error: unknown kernel '"+kernel+"'")
            exit(1)
        

    def fit(self, data, labels):
        self.num_samples, self.num_features = data.shape

        if self.kernel=='linear' and (self.num_samples > self.num_features):
            self.dual = False
            self.clf.set_params(dual=self.dual)
    
        self.clf.fit(data,labels)


    def predict(self, data):
        return self.clf.predict(data)
