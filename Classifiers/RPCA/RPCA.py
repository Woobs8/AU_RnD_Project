# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
import numpy as np
from Classifiers.Classifier import Classifier
#from r_pca import R_pca
from Tools.Printing import create_pbar
from Tools.Datagrids import datagrid

try:
    # Python 2: 'xrange' is the iterative version
    range = xrange
except NameError:
    # Python 3: 'range' is iterative - no need for 'xrange'
    pass


class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.norm_p(self.D, 2))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(self.D), 2)

        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(np.abs(self.D - Lk - Sk), 2)
            iter += 1
            if iter_print != None:
                if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                    print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

class RobustPCAGrid(Classifier):
    def __init__(self, grid=[100,100,0], max_iter=1000, window_type='rectangle',overlap=0, confidence=2,predict_method='full',name='RPCA'):
        super().__init__(name)
        self.grid = grid
        self.max_iter = max_iter
        self.overlap = overlap
        self.predict_method = predict_method
        self.confidence=confidence
        
        window_types = ['rectangle','ellipse','cuboid','ellipsoid']
        if not any(n in window_type.lower() for n in window_types):
            print("ERROR: Only allowed window types are: {0}".format(', '.join(map(str, window_types))) )
            exit(1)
        self.window_type = window_type.lower()
        if len(self.grid)==2:
            self.L_list = [None] * (self.grid[0]*self.grid[1])
            self.S_list = [None] * (self.grid[0]*self.grid[1])
            self.sample_list = [None] * (self.grid[0]*self.grid[1])
        else:
            self.L_list = [None] * (self.grid[0]*self.grid[1]*self.grid[2])
            self.S_list = [None] * (self.grid[0]*self.grid[1]*self.grid[2])
            self.sample_list = [None] * (self.grid[0]*self.grid[1]*self.grid[2])


    def fit(self, data, lbls=None):
        # data dimensionality
        self.n, self.d = data.shape

        # data must be 3-dimensional
        if self.d > 3:
            print("Error: input data is a {0}-dimensional array - expected 3-dimensional array".format(self.d))
            exit(1)

        # prepare grid
        subset_idcs = datagrid(data,self.grid,self.window_type,self.overlap)
        perms = len(subset_idcs)
        # Create a progress bar
        pbar = create_pbar(perms,"Calculating RPCA")

        # iterate all squares in the grid
        none_elements = [] 
        count_samples= 0

        for i in range(perms):
            self.sample_list[i] = subset_idcs[i][:].reshape((len(subset_idcs[i]),1))
            # apply robust pca to determine low-rank and sparse matrix
            if subset_idcs[i].shape[0] > 0:
                subset_data = data[subset_idcs[i]]
                rpca = R_pca(subset_data)
                L, S = rpca.fit(max_iter=self.max_iter, iter_print=None)
                self.S_list[i] = S
                self.L_list[i] = L
            else:
                none_elements.append(i)
            pbar.update(i)
        pbar.finish()
        for index in sorted(none_elements, reverse=True):
            del self.S_list[index]
            del self.L_list[index] 
            del self.sample_list[index]
        
        return self


    def predict(self, Data):
        # prepare arrays used for traversing the point cloud              
        labels = np.full((self.n,1), 0 ,dtype=int)     # labels for all points
        if  self.predict_method == 'full':           
            # Iteratively estimate a zero-mean Gaussian distribution of the z-value of inlier samples in the sparse representation
            z_mean = 0
            S_z = self.get_full_S()[:,2]
            inliers = S_z
            outlier_idcs = np.arange(self.n)
            while outlier_idcs.size > 0 and inliers.size > 0:   # iteratively find and remove outliers (<5% confidence) until no outliers are present in the estimation
                z_std = np.std(inliers)
                outlier_idcs = np.argwhere( (inliers < (z_mean-self.confidence*z_std)) | (inliers > (z_mean+self.confidence*z_std)) )
                inliers = np.delete(inliers, outlier_idcs[:,0], axis=0)

            # find outliers in the subset data by filtering the samples based on the z-value in the sparse domain
            outlier_idcs = np.argwhere( (S_z < (z_mean-self.confidence*z_std)) | (S_z > (z_mean+self.confidence*z_std)) )
            labels[outlier_idcs] = 1

        elif  self.predict_method == 'voting':
            pbar = create_pbar(len(self.S_list),"Filtering outliers")
            for idx in range(len(self.S_list)):
                # Iteratively estimate a zero-mean Gaussian distribution of the z-value of inlier samples in the sparse representation
                inliers = self.S_list[idx][:,2]
                outlier_idcs = np.arange(self.n)
                while outlier_idcs.size > 0 and inliers.size > 0:   # iteratively find and remove outliers (<5% confidence) until no outliers are present in the estimation
                    z_mean = np.mean(inliers)
                    z_std = np.std(inliers)
                    outlier_idcs = np.argwhere( (inliers < (z_mean-self.confidence*z_std)) | (inliers > (z_mean+self.confidence*z_std)) )
                    inliers = np.delete(inliers, outlier_idcs[:,0], axis=0)

                # find outliers in the subset data by filtering the samples based on the z-value in the sparse domain
                outlier_idcs = self.sample_list[idx][np.argwhere( (self.S_list[idx][:,2] < (z_mean-self.confidence*z_std)) | (self.S_list[idx][:,2] > (z_mean+self.confidence*z_std)) )]
                inlier_idcs = np.setdiff1d(self.sample_list[idx],outlier_idcs)
                labels[outlier_idcs] += 1
                labels[inlier_idcs] -= 1
                pbar.update(idx)
            pbar.finish()
            labels = np.where(labels >0 ,1,0)
        else:
            print("ERROR: Method does not exist, choose between 'full' or 'voting' ")
            exit(1)
        return labels

    def get_full_S(self):
        S = np.zeros((self.n,self.d))
        for i in range(len(self.sample_list)):
            S[self.sample_list[i].ravel(),:] += self.S_list[i]

        # Stack the created sample list
        sample_list_stack = np.vstack(self.sample_list)
        
        # Initialize S and L Matrix
        _, sample_counts = np.unique(sample_list_stack,return_counts=True)

        # Mean of added samples
        return S / sample_counts[:,None]
    
    def get_full_L():
        L = np.zeros((self.n,self.d))
        for i in range(len(self.sample_list)):
            L[self.sample_list[i],:] += self.L_list[i]
        # Stack the created sample list
        sample_list_stack = np.vstack(self.sample_list)
        
        # Initialize S and L Matrix
        _, sample_counts = np.unique(sample_list_stack,return_counts=True)

        # Mean of added samples
        return L / sample_counts[:,None]
        
            
