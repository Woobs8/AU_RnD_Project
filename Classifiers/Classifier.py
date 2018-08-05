from sklearn.metrics import accuracy_score as sk_accuracy
import numpy as np
import sys

class Classifier(object):
    def __init__(self, name):
        self.name = name

    def accuracy(self, pred_labels, true_labels):
        # The error count
        if len(pred_labels.shape) == 1:
            pred_labels = np.expand_dims(pred_labels,1)
        
        if len(true_labels.shape) == 1:
            true_labels = np.expand_dims(true_labels,1)
            
        err_count = np.count_nonzero(pred_labels-true_labels)

        # Overall accuracy
        scores = sk_accuracy(true_labels, pred_labels, normalize=True, sample_weight=None)
        
        # Initialize mapping of labels for confusion matrix
        unique_lbls  = np.unique(true_labels)
        num_unique_lbls = len(unique_lbls)
        lbls_map = dict(enumerate(unique_lbls))
        lbls_reverse_map = dict(map(reversed, lbls_map.items()))      
        
        # Confusion Mastrix
        conf_matrix = np.zeros((num_unique_lbls,num_unique_lbls))
        for i, lbl in lbls_map.items():
            # Find the indexes in the true label array for current label
            positives_idx = np.where(true_labels==lbl)[0]
            # Find the unique values in the predicted labels array for the above indexes (also the occurance count of each value)
            u, counts = np.unique(pred_labels[positives_idx],return_counts=True)
            # Make sure that all unique true labels are represented
            insert_idxs = [lbls_reverse_map[u[i]] for i in range(len(u))]
            # Insert the percentage wise counts in the confusion matrix
            conf_matrix[i,insert_idxs] = counts/np.sum(counts)

        return scores, err_count, conf_matrix
