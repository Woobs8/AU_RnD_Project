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
      data = load_raw_data(data_path_tail, data_path_head)
      
      ### ********** PREPROCESS DATA ********** ###
      data[:,:3] = PcaRotation(data[:,:3])
      data[:,:3] = ScaleData(data[:,:3])
      
      ### ********** THREE BEST CONFIGURATIONS FROM REPORT "GRID SEARCH" ******** ###
      rpca_testing_windows = [[225,225],[225,225],[125,125]]
      rpca_testing_windows_types = ['rectangle','ellipse','rectangle']
      rpca_testing_overlaps = [0.5, 0.8, 0.75]
      rpca_testing_confidence = [1.5,2,2.5]

      ### ********** PERFROM RPCA ON NEW DATASET WITH DEFINED CONFIGURATIONS ********* ###
      for window,window_type,overlap,confidence in zip(rpca_testing_windows,rpca_testing_windows_types,rpca_testing_overlaps,rpca_testing_confidence):
            # Define Naming Strings
            save_path = args.save_path + "/RPCA_generalization_voting"
            string_overlap = "{:.2f}".format(overlap).replace(".","")
            classifier_name = "RPCA_{}_{}x{}_ol{}".format(window_type,window[0],window[1],string_overlap)
            
            # Perfrom rpca and save it
            rpca = RobustPCAGrid(window, max_iter=2000, overlap=overlap, window_type=window_type,predict_method='voting',confidence=confidence,name=classifier_name)
            rpca, rpca_prediction, rpca_score, rpca_err, rpca_confusion_matrix = classify(rpca, train_data=data[:,:3], train_lbls=data[:,-1],test_data=data[:,:3], test_lbls=data[:,-1], stats=args.stats, save=True, save_dir=save_path)
            
            # Create results matrix and print to stdout
            conf_mat = matrix_string(rpca_confusion_matrix)
            print("*** {} ***".format(classifier_name))
            print("\trpca Score: %.6f \n %s" %(rpca_score,conf_mat))


if __name__ == "__main__":

      print("\n")
      parser = argparse.ArgumentParser(prog='EIVA R&D Project',
                                          description='''This script provides the ability to perform rpca on a new data set with defined best parameters from the project report results''')

      parser.add_argument(    'path', 
                              help='Input path and name of file containing the data')
      
      parser.add_argument('-save_path', 
                              help='Results save path',
                              type=str,
                              default="Results")

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

      print("Using the following arguments:"
            "\n", args, "\n")
      
      main(data_path_head=head,data_path_tail=tail,args=args)

      if args.figsshow:
            if sys.flags.interactive != 1:
                  vispy.app.run()
      exit()