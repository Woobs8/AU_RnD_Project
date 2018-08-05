# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import progressbar as ProgressBar
import sys

# Mean distance to K nearest neighbors
def knn_mean_dist(raw_data, k):
    print("Creating Features based on knn mean distance")    
    # Shape of input data
    num_samples, num_features = raw_data.shape
    samples_pct = np.floor(num_samples/100)
    
    # Initialize feature output matrix
    if isinstance(k,list):
        features = np.zeros((num_samples,len(k)))
        max_k = max(k)
    else:
        features = np.zeros((num_samples,1))
        max_k = k
    
    # Make KDTree for nearest neighbor queries
    tree = KDTree(raw_data, leaf_size=2)             

    # Query of the k_max closest points from the KDTree
    dist, ind = tree.query(raw_data, k=max_k+1,sort_results=True)

    # Create a progress bar
    pbar = create_pbar(num_samples)

    # Iterate all points
    for idx, point in enumerate(raw_data):
        # Find distance of kth nearest neighbor
        if isinstance(k,list):
            for i in range(len(k)):
                features[idx,i] = np.mean(dist[idx][1:k[i]])
        else:
            features[idx] = np.mean(dist[idx][1:])
        # Update the progressbar. Though only in hole percentages to save time
        if idx % samples_pct == 0:
            pbar.update(idx) 
    pbar.finish()
    return features


# Max distance to K nearest neighbors
def knn_max_dist(raw_data, k):
    print("Creating Features based on knn max distance")
    # Shape of input data
    num_samples, num_features = raw_data.shape
    samples_pct = np.floor(num_samples/100)
    
    # Initialize feature output matrix
    if isinstance(k,list):
        features = np.zeros((num_samples,len(k)))
        max_k = max(k)
    else:
        features = np.zeros((num_samples,1))
        max_k = k
    
    # Make KDTree for nearest neighbor queries
    tree = KDTree(raw_data, leaf_size=2)             

    # Query of the k_max closest points from the KDTree
    dist, ind = tree.query(raw_data, k=max_k+1,sort_results=True)

    # Create a progress bar
    pbar = create_pbar(num_samples)

    # Iterate all points
    for idx, point in enumerate(raw_data):
        # Find distance of kth nearest neighbor
        if isinstance(k,list):
            for i in range(len(k)):
                features[idx,i] = dist[idx][k[i]]
        else:
            features[idx] = dist[idx][k]
        # Update the progressbar. Though only in hole percentages to save time
        if idx % samples_pct == 0:
            pbar.update(idx) 
    pbar.finish()
    return features


# Mean Z-distance to K nearest neighbors
def knn_mean_z_dist(raw_data, k):
    print("Creating Features based on knn mean z distance")
        # Shape of input data
    num_samples, num_features = raw_data.shape
    samples_pct = np.floor(num_samples/100)
    
    # Initialize feature output matrix
    if isinstance(k,list):
        features = np.zeros((num_samples,len(k)))
        max_k = max(k)
    else:
        features = np.zeros((num_samples,1))
        max_k = k
    
    # Make KDTree for nearest neighbor queries
    tree = KDTree(raw_data, leaf_size=2)             

    # Query of the k_max closest points from the KDTree
    dist, ind = tree.query(raw_data, k=max_k+1,sort_results=True) 
    
    # Create a progress bar
    pbar = create_pbar(num_samples)
    
    # Iterate all points
    for idx, point in enumerate(raw_data):
        # Find distance of kth nearest neighbor
        if isinstance(k,list):
            for i in range(len(k)):
                # Find corresponding neighbors
                neighbors = raw_data[ind[idx][1:k[i]],2]
                # Find mean Z-distance to nearest neighbors
                features[idx,i] = np.mean(abs(neighbors-point[2]))
        else:
            # Find corresponding neighbors
            neighbors = raw_data[ind[idx][1:k+1],2]
            # Find mean Z-distance to nearest neighbors
            features[idx] = np.mean(abs(neighbors-point[2]))
        # Update the progressbar. Though only in hole percentages to save time
        if idx % samples_pct == 0:
            pbar.update(idx) 
    pbar.finish()

    return features


def dist_to_plane(raw_data, radius):
    print("Creating Features based on distance to plane")
    num_samples, num_features = raw_data.shape
    feature = np.zeros((num_samples,1))

    # Iterate all points
    for idx, point in enumerate(raw_data):
        # Calculate distance to all other points in 2D
        dist = np.sqrt(((point[:2] - (raw_data[:,:2])[:,np.newaxis,:]) ** 2).sum(axis=2))

        # Filter data points by distance (filter in a radius)
        indices = np.where(dist<=radius)
        data_circle = raw_data[indices[0],:3]

        # Estimate plane using eigenanalysis (PCA)
        pca = PCA(n_components=2)
        pca.fit(data_circle)
        V = pca.components_

        # Normal (orthogonal) vector to the plane is the third pca component
        norm = V[:,2]

        # Dist = |norm * point| / |norm|
        feature[idx] = np.abs(np.dot(norm,V[:,0])) / np.linalg.norm(norm)
    return feature

def centered_z_squared(raw_data):
    print("Creating Features based on centered z squared")
    # Center Data
    mean = np.mean(raw_data,axis=0)
    raw_data[:,0] -= mean[0] 
    raw_data[:,1] -= mean[1] 
    raw_data[:,2] -= mean[2] 
    # Squre z values
    feature = np.power(raw_data[:,2],2)
    return feature

def samples_within_sphere(raw_data, radius):
    print("Creating Features based on samples within sphere")
    num_samples, num_features = raw_data.shape
    samples_pct = np.floor(num_samples/100)
    
    # Initialize feature output matrix
    if isinstance(radius,list):
        features = np.zeros((num_samples,len(radius)))
        max_r = max(radius)
    else:
        features = np.zeros((num_samples,1))
        max_r = radius
    
    # Make KDTree for nearest neighbor queries
    tree = KDTree(raw_data)         

    # Query of the number of samples with in sphere from the KDTree
    count = tree.query_radius(raw_data, r=radius, count_only=True)

    # Create a progress bar
    pbar = create_pbar(num_samples)

    # Iterate all points
    for idx, point in enumerate(raw_data):
        if isinstance(radius,list):
            #ind, dist = tree.query_radius([point], r=max_r, return_distance = True)
            for i,r in enumerate(radius):
                # Query of the number of samples with in sphere from the KDTree
                count = tree.query_radius([point], r=r, count_only=True)
                features[idx,i] = count[0]
        else:
            features[idx] = count[idx]
       
        # Update the progressbar. Though only in hole percentages to save time
        if idx % samples_pct == 0:
            pbar.update(idx) 
    pbar.finish() 

    return features

def centered_z_summation_within_sphere(raw_data, radius):
    print("Creating Features based on centered z summation within sphere")
    num_samples, num_features = raw_data.shape
    samples_pct = np.floor(num_samples/100)
    
    # Initialize feature output matrix
    if isinstance(radius,list):
        features = np.zeros((num_samples,len(radius)))
        max_r = max(radius)
    else:
        features = np.zeros((num_samples,1))
        max_r = radius

    # Center Data
    mean = np.mean(raw_data,axis=0)
    raw_data[:,0] -= mean[0] 
    raw_data[:,1] -= mean[1] 
    raw_data[:,2] -= mean[2] 
    
    # Make KDTree for nearest neighbor queries
    tree = KDTree(raw_data, leaf_size=2)         

    # Query of the k_max closest points from the KDTree
    ind, dist = tree.query_radius(raw_data[:], r=max_r, return_distance = isinstance(radius,list))

    # Create a progress bar
    pbar = create_pbar(num_samples)

    # Iterate all points
    for idx, point in enumerate(raw_data):
        if isinstance(radius,list):
            #ind, dist = tree.query_radius([point], r=max_r, return_distance = True)
            for i,r in enumerate(radius):
                # Filter data points by distance (filter in a radius)
                indices = np.where(dist[idx]<=r)[0]
                data_sphere = raw_data[ind[idx][indices],:3]
                # Sum z values in sphere
                features[idx,i] = np.sum(data_sphere[:,2])
        else:
            data_sphere = raw_data[ind[idx],:3]
            # Sum z values in sphere
            features[idx] = np.sum(data_sphere[:,2])
       
        # Update the progressbar. Though only in hole percentages to save time
        if idx % samples_pct == 0:
            pbar.update(idx) 
    pbar.finish() 

    return features

def centered_z_mean_within_sphere(raw_data, radius):
    print("Creating Features based on centered z mean within sphere")
    num_samples, num_features = raw_data.shape
    samples_pct = np.floor(num_samples/100)
    
    # Initialize feature output matrix
    if isinstance(radius,list):
        features = np.zeros((num_samples,len(radius)))
        max_r = max(radius)
    else:
        features = np.zeros((num_samples,1))
        max_r = radius

    # Center Data
    mean = np.mean(raw_data,axis=0)
    raw_data[:,0] -= mean[0] 
    raw_data[:,1] -= mean[1] 
    raw_data[:,2] -= mean[2] 
    
    # Make KDTree for nearest neighbor queries
    tree = KDTree(raw_data, leaf_size=2)         

    # Query of the k_max closest points from the KDTree
    ind, dist = tree.query_radius(raw_data[:], r=max_r, return_distance = isinstance(radius,list))

    # Create a progress bar
    pbar = create_pbar(num_samples)

    # Iterate all points
    for idx, point in enumerate(raw_data):
        if isinstance(radius,list):
            #ind, dist = tree.query_radius([point], r=max_r, return_distance = True)
            for i,r in enumerate(radius):
                # Filter data points by distance (filter in a radius)
                indices = np.where(dist[idx]<=r)[0]
                data_sphere = raw_data[ind[idx][indices],:3]
                # Mean of z values in sphere
                features[idx,i] = np.mean(data_sphere[:,2])
        else:
            data_sphere = raw_data[ind[idx],:3]
            # Mean of z values in sphere
            features[idx] = np.mean(data_sphere[:,2])
    
    return features


def create_pbar(max_val):
    widgets = ['Progress: ', ProgressBar.Percentage(), ' ', ProgressBar.Bar(marker='#',left='[',right=']'),
        ' ', ProgressBar.ETA()] #see docs for other options

    pbar = ProgressBar.ProgressBar(widgets=widgets, maxval=max_val)
    pbar.start()
    return pbar