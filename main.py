import argparse
import vispy.scene
from Tools.DataLoader import load_raw_data, split_data
from Classifiers.kNN import KNearestNeighbors
from Classifiers.Bayes import NaiveBayes
from Classifiers.SVM import SupportVectorMachine
from Classifiers.RPCA.RPCA import RobustPCAGrid
from Classifiers.Classify import classify
from Tools.Features import dist_to_plane, knn_mean_dist, knn_mean_z_dist, knn_max_dist, centered_z_summation_within_sphere, samples_within_sphere
from Tools.Scatter import ScatterPlot3D
from Tools.Preprocess import ScaleData, PcaRotation, ScaleFeatures
from Tools.Printing import matrix_string


import numpy as np
import os
import sys

# Global Variables
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/Data'

def main(data_path_head,data_path_tail,args):   
      ### ********** LOAD DATA ********** ###
      print("-- Loading Data  --\n")
      data = load_raw_data(data_path_tail, data_path_head)
      print("Noise Ratio in data %f"%(np.count_nonzero(data[-1])/data.shape[0]))
      
      ### ********** PREPROCESS DATA ********** ###
      print("\n-- Preprocessing Data  --\n")
      # preprocess true coordinates
      data[:,:3] = PcaRotation(data[:,:3])
      data[:,:3] = ScaleData(data[:,:3])
      # preproess features
      data[:,3:-2] = ScaleFeatures(data[:,3:-2])

      ### ********** CREATE FEATURES ********** ###
      # dist = 1
      # dtp_feature = dist_to_plane(data[:,:3],dist)
      
      # k=[3,6,9,12,15]
      # kmean_dist_feature = knn_mean_dist(data[:,:3],[3,6,9,12,15])
      # kmean_z_dist_feature = knn_mean_z_dist(data[:,:3],[3,6,9,12,15])
      # kmaxd_feature= knn_max_dist(data[:,:3],k)

      # radius = [0.1 0.2 0.4 0.8 1]
      # #n_in_sphere_feature = samples_within_sphere(data[:,:3],radius)
      # #cent_z_sum_sphere_feature = centered_z_summation_within_sphere(data[:,0:3],radius)

      # # Write code to incorporate features in the dataset
      # data = data ### !!!


      ### ********** TRAIN TEST SPLIT ********** ###
      chosen_features = np.array([1,2,3])
      features_incl_lbls = np.append(chosen_features,-1)
      train_data, train_lbls, test_data, test_lbls = split_data(data[:,features_incl_lbls],test_size=0.25)

      ### ********** Nearest Neighbors Classifier ********** ###
      if args.knn is True:
            print("-- Performing kNN  --")
            k_neighbors = 1
            knn = KNearestNeighbors(k=k_neighbors)
            knn, knn_prediction, knn_score, knn_err, knn_confusion_matrix = classify(knn, train_data, train_lbls, test_data, test_lbls, stats=args.stats)
            conf_mat = matrix_string(knn_confusion_matrix)
            print("\tkNN k=%d Score: %.6f \n %s" %(k_neighbors,knn_score,conf_mat))
      
      ### ********** Naive Bayes Classifier********** ###
      if args.nb is True:
            print("-- Performing Naive Bayes --")
            nb = NaiveBayes()
            nb, nb_prediction, nb_score, nb_err, nb_confusion_matrix = classify(nb, train_data, train_lbls, test_data, test_lbls, save=True, stats=args.stats)
            conf_mat = matrix_string(nb_confusion_matrix)
            print("\tnb Score: %.6f \n %s" %(nb_score,conf_mat))

      ### ********** Linear/Non-linear SVM Classifier********** ###
      if args.svm is True:
            print("-- Performing Linear SVM --")
            lin_svm = SupportVectorMachine(max_iter=200)
            lin_svm, lin_svm_pred, lin_svm_score, lin_svm_err, lin_confusion_matrix = classify(lin_svm, train_data, train_lbls, test_data, test_lbls, stats=args.stats)
            conf_mat = matrix_string(lin_confusion_matrix)
            print("\tLinear SVM Score: %.6f \n %s" %(lin_svm_score,conf_mat))

            print("-- Performing Kernel SVM --")
            kern_svm = SupportVectorMachine(kernel='rbf', max_iter=1000,gamma='auto',C=0.1,tol=1e-3)
            kern_svm, kern_svm_pred, kern_svm_score, kern_svm_err, kern_confusion_matrix = classify(kern_svm, train_data, train_lbls, test_data, test_lbls, stats=args.stats)
            conf_mat = matrix_string(kern_confusion_matrix)
            print("\tKernel SVM Score: %.6f \n %s" %(kern_svm_score,conf_mat))
                 
      ### ********** Robust PCA ********** ###
      if args.rpca is True:
            print("-- Performing Robust PCA --")
            rpca = RobustPCAGrid([101,101], max_iter=2000, overlap=0, window_type='rectangle',predict_method='voting')
            rpca, rpca_prediction, rpca_score, rpca_err, rpca_confusion_matrix = classify(rpca, train_data=data[:,:3], train_lbls=data[:,-1],test_data=data[:,:3], test_lbls=data[:,-1], stats=args.stats, save=False)
            conf_mat = '\t['+']\n\t['.join('\t'.join('%0.3f' %x for x in y) for y in rpca_confusion_matrix) + ']'
            print("\trpca Score: %.6f \n %s" %(rpca_score,conf_mat))
            if args.figsshow:          
                  ScatterPlot3D(data, labels=rpca_prediction, x_feat=0, y_feat=1, z_feat=2, title="RPCA: Predictions")
                  filt_data = np.delete(data,np.argwhere(rpca_prediction == 1)[:,0],axis=0)
                  ScatterPlot3D(filt_data,labels=np.zeros(len(filt_data)), title="RPCA: Noise Filtered")

                  #reconstr = surface_reconstruction(filt_data, resolution=[500,500])
                  #ScatterPlot3D(reconstr,labels=np.zeros(len(reconstr)), title="RPCA: Surface Estimation")

      ### ********** PLOTTING DATA ********** ####
      """ScatterPlot3D(data, labels=training_labels, x_feat=0, y_feat=1, z_feat=2, label_feat=-1, title="Scatterplot")"""


if __name__ == "__main__":

      print("\n")
      parser = argparse.ArgumentParser(prog='EIVA R&D Project',
                                          description='''This script provides the ability to train machine 
                                          learning algorithms to classify noisy samples in a sonar 3D point cloud data set provided by EIVA''')

      parser.add_argument('path', 
            help='Input path and name of file containing the data to be plotted')

      # Commands for Machine Learning Algorithms
      parser.add_argument('-knn',
                              help='Use k Nearest Neightbors Classifier',
                              action="store_true")
      
      parser.add_argument('-nb',
                              help='Use Naive Bayes Classifier',
                              action="store_true")

      parser.add_argument('-svm',
                              help='Use SVM Classifier',
                              action="store_true")
      parser.add_argument('-rpca',
                              help='Use SVM Classifier',
                              action="store_true")
      # Command for showing figures
      parser.add_argument('-figsshow',
                              help='Plot Figures',
                              action="store_true")

      # Commands for classification analysis
      parser.add_argument('-stats',
                        help='Measure the training error and the timing of the training and classification',
                        action="store_true")

      #  Parse Arguments
      args = parser.parse_args()


      # Path is a data file
      if os.path.exists(args.path):
            # Get file name and path from argument
            head, tail = os.path.split(args.path)

      else:
            print("Error: file '" + args.path + "' not found")
            exit(1)

      # If no algorithm is specified run all
      if (args.knn or args.nb or args.svm or args.rpca) is False:
            args.knn = args.nb = args.svm = args.rpca =  True

      print("Using the following arguments:"
            "\n", args, "\n")
      
      main(data_path_head=head,data_path_tail=tail,args=args)

      if args.figsshow:
            if sys.flags.interactive != 1:
                  vispy.app.run()
      exit()