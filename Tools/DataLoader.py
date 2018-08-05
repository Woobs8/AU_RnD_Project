#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_raw_data(file_name, path):
    path_full, file_extension = os.path.splitext(os.path.join(path,file_name))
    path_full_numpy = path_full+'.npy'
    # Load data from file 
    if os.path.exists(path_full_numpy):
        # If Numpy file exists the numpy data is read directly
        data = np.load(path_full_numpy)
    else:
        # Files have different format and feature length
        # Try to load file with 9 features and string label
        try:
            # Open .txt file and store in "numpy" readable format
            data = np.loadtxt(path_full+file_extension,delimiter=' ', 
                                                        dtype={'names': ('x', 'y', 'z', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5','feature6','label'),
                                                            'formats': (np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, 'U5')})
            # make sure all labels are binary
            data_len = len(data)
            parsed_data = np.zeros((data_len,10))
            
            for i, sample in enumerate(data):
                if sample[-1] == 'True':
                    sample[-1] = 1
                elif sample[-1] == 'False':
                    sample[-1] = 0
                parsed_data[i,:] = list(sample)
            data = parsed_data
        # otherwise load file with 8 features and binary label
        except IndexError:
            data = data = np.loadtxt(path_full+file_extension,delimiter=' ')
        
        np.save(path_full_numpy,data)
    return data


def load_and_split_data(file_name, path, test_size=0.25, size=-1):
    data = load_raw_data(file_name, path)
    return split_data(data,test_size,size)

def split_data(data, test_size=0.25, size=-1):    
    # Splice data into labels and feature vectors
    bin_label = data[:,-1]
    data = data[:,:-1]

    # Create smaller data set from original dataset (for testing purposes)
    # TODO: currently includes all noise samples in the subset no matter the size
    if(size != -1):
        data_noise = np.squeeze(data[ np.where(bin_label == 1 ),:],axis=0)
        data_not_noise = np.squeeze(data[ np.where(bin_label == 0 ),:],axis=0)
        noise_samples, noise_features = data_noise.shape
        bin_label = np.concatenate((np.ones((noise_samples,)), np.zeros((size-noise_samples,))),axis=0)
        data = np.concatenate((data_noise, data_not_noise[0:size-noise_samples,:]), axis=0)

    # Create train/test split (stratify ensures samples with same labels are equally spread between sets
    training_data, test_data, training_labels, test_labels = train_test_split(data, bin_label, test_size=0.25,stratify=bin_label)
    return training_data, np.expand_dims(training_labels,1), test_data, np.expand_dims(test_labels,1)
