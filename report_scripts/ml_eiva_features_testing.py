import sys
import os
sys.path.append(os.getcwd())

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
import dill


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
      
      ### ********** PREPROCESS DATA ********** ###
      print("\n-- Preprocessing Data  --\n")
      
      # preprocess true coordinates
      data[:,:3] = PcaRotation(data[:,:3])
      data[:,:3] = ScaleData(data[:,:3])
      
      # preproess features
      data[:,3:-1] = ScaleFeatures(data[:,3:-1])


      ### ********** TRAIN TEST SPLIT ********** ###
      eiva_feature_names = ["x",
                            "y",
                            "z", 
                            "Dist_to_neighbour",
                            "Dist_to_avg_surf_r80cm" , 
                            "Neighbours_in_sphere_r80cm" , 
                            "Z_sum_in_circ_r80cm" , 
                            "kNN_mean_Z_dist_n8" , 
                            "kNN_mean_dist_k8"]

      for feature,feature_name in enumerate(eiva_feature_names[3:]):
            feature=feature+3 # offset dont count x,y,z 
            feature_incl_labels = np.append(feature,-1)
            
            train_data, train_lbls, test_data, test_lbls = split_data(data[:,feature_incl_labels],test_size=0.25)
            
            ### ********** Nearest Neighbors Classifier ********** ###
            if args.knn is True:
                  print("-- Performing kNN  --")
                  for k_neighbors in 2**np.arange(9):
                        save_path = args.save_path + "/kNN"
                        classifier_name = "kNN_k{}_".format(k_neighbors)+"Feature{}={}".format(feature,feature_name)
                        knn = KNearestNeighbors(k=k_neighbors,name=classifier_name)
                        knn, knn_prediction, knn_score, knn_err, knn_confusion_matrix = classify(knn, train_data, train_lbls, test_data, test_lbls, stats=args.stats, save=True, save_dir=save_path)
                        conf_mat = matrix_string(knn_confusion_matrix)
                        print("\tkNN k=%d Score: %.6f \n %s" %(k_neighbors,knn_score,conf_mat))
            
            ### ********** Naive Bayes Classifier********** ###
            if args.nb is True:
                  print("-- Performing Naive Bayes --")
                  save_path = args.save_path + "/NB"
                  classifier_name ="NB_"+"Feature{}={}_".format(feature,feature_name)
                  nb = NaiveBayes(name=classifier_name)
                  nb, nb_prediction, nb_score, nb_err, nb_confusion_matrix = classify(nb, train_data, train_lbls, test_data, test_lbls, stats=args.stats, save=True, save_dir=save_path)
                  conf_mat = matrix_string(nb_confusion_matrix)
                  print("\tnb Score: %.6f \n %s" %(nb_score,conf_mat))

            ### ********** Linear/Non-linear SVM Classifier********** ###
            if args.svm is True:
                  for C in (2.**np.arange(-5,16,10)).tolist():
                        print(C)
                        for tol in (10.**np.arange( -5, 0, 2 )).tolist():
                              print("-- Performing Linear SVM --")
                              save_path = args.save_path + "/linSVM"
                              classifier_name = "LinSVM_C{}_tol{}_".format(C,tol)+"Feature{}={}_".format(feature,feature_name)
                              lin_svm = SupportVectorMachine(max_iter=200,C=C,tol=tol,name=classifier_name)
                              lin_svm, lin_svm_pred, lin_svm_score, lin_svm_err, lin_confusion_matrix = classify(lin_svm, train_data, train_lbls, test_data, test_lbls, stats=args.stats, save=True, save_dir=save_path)
                              conf_mat = matrix_string(lin_confusion_matrix)
                              print("\tLinear SVM Score: %.6f \n %s" %(lin_svm_score,conf_mat))
                              for gamma in (2.**np.arange(-10,0,3)).tolist():
                                    print("-- Performing Kernel SVM --")
                                    save_path = args.save_path + "/RbfSVm"
                                    classifier_name = "RbfSVM_C{}_tol{}_gamme{}".format(C,tol,gamma)+"Feature{}={}_".format(feature,feature_name)
                                    kern_svm = SupportVectorMachine(kernel='rbf', max_iter=200,gamma=gamma, C=C, tol=tol,name=classifier_name)
                                    kern_svm, kern_svm_pred, kern_svm_score, kern_svm_err, kern_confusion_matrix = classify(kern_svm, train_data, train_lbls, test_data, test_lbls, stats=args.stats, save=True, save_dir=save_path)
                                    conf_mat = matrix_string(kern_confusion_matrix)
                                    print("\tKernel SVM Score: %.6f \n %s" %(kern_svm_score,conf_mat))
                        


if __name__ == "__main__":

      print("\n")
      parser = argparse.ArgumentParser(prog='EIVA R&D Project',
                                          description='''This script provides the ability to train machine 
                                          learning algorithms to classify noisy samples based on a single feature''')

      parser.add_argument(    'path', 
                              help='Input path and name of file containing the data to be plotted')
      
      parser.add_argument('-save_path', 
                              help='Results save path',
                              type=str,
                              default="Results")

      # Commands for Machine Learning Algorithms

      parser.add_argument('-knn',
                                    help='Use SVM Classifier',
                                    action="store_true")
      parser.add_argument('-nb',
                                    help='Use SVM Classifier',
                                    action="store_true")
      parser.add_argument('-svm',
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
      if (args.knn or args.nb or args.svm) is False:
            args.knn = args.nb = args.svm =  True

      print("Using the following arguments:"
            "\n", args, "\n")
      
      main(data_path_head=head,data_path_tail=tail,args=args)

      if args.figsshow:
            if sys.flags.interactive != 1:
                  vispy.app.run()
      exit()