import sys
import os
sys.path.append(os.getcwd())

import argparse
import vispy.scene
from Tools.DataLoader import load_raw_data, split_data
from Classifiers.RPCA.RPCA import RobustPCAGrid
from Classifiers.Classify import classify
from Tools.Scatter import ScatterPlot3D
from Tools.Preprocess import ScaleData, PcaRotation
from Tools.Printing import matrix_string

def main(data_path_head,data_path_tail,args):   
      ### ********** LOAD DATA ********** ###
      print("-- Loading Data  --\n")
      data = load_raw_data(data_path_tail, data_path_head)
      
      ### ********** PREPROCESS DATA ********** ###
      print("\n-- Preprocessing Data  --\n")
      data[:,:3] = PcaRotation(data[:,:3])
      data[:,:3] = ScaleData(data[:,:3])
      
      rpca_testing_windows = [[25,25],[75,75],[125,125],[175,175],[225,225]]
      rpca_testing_rect_overlaps = [0, 0.25, 0.5, 0.75]
      rpca_testing_ellip_overlaps = [0.5, 0.6, 0.7, 0.8]
      rpca_testing_sigmas = [1.5,2,2.5]

      ### ********** Perfrom Grid search of defined parameters with Robust PCA ********** ###
      for sigma in rpca_testing_sigmas:
            save_path = args.save_path + "_std{}".format(sigma).replace(".","")
            for window in rpca_testing_windows:
                  for overlap in rpca_testing_rect_overlaps:
                        save_path_rectangle = save_path + "/RPCA_Rect_voting"
                        string_overlap = "{:.2f}".format(overlap).replace(".","")
                        classifier_name = "RPCA_Rect_{}x{}_ol{}".format(window[0],window[1],string_overlap)
                        rpca = RobustPCAGrid(window, max_iter=2000, overlap=overlap, window_type='rectangle',predict_method='voting',name=classifier_name)
                        rpca, rpca_prediction, rpca_score, rpca_err, rpca_confusion_matrix = classify(rpca, train_data=data[:,:3], train_lbls=data[:,-1],test_data=data[:,:3], test_lbls=data[:,-1], stats=args.stats, save=True, save_dir=save_path_rectangle)
                        conf_mat = matrix_string(rpca_confusion_matrix)
                        print("*** {} ***".format(classifier_name))
                        print("\trpca Score: %.6f \n %s" %(rpca_score,conf_mat))
                  
                  for overlap in rpca_testing_ellip_overlaps:
                        save_path_ellipse = save_path + "/RPCA_Ellipse_voting"
                        string_overlap = "{:.2f}".format(overlap).replace(".","")
                        classifier_name = "RPCA_Ellipse_{}x{}_ol{}".format(window[0],window[1],string_overlap)
                        rpca = RobustPCAGrid(window, max_iter=2000, overlap=overlap, window_type='ellipse',predict_method='voting',name=classifier_name)
                        rpca, rpca_prediction, rpca_score, rpca_err, rpca_confusion_matrix = classify(rpca, train_data=data[:,:3], train_lbls=data[:,-1],test_data=data[:,:3], test_lbls=data[:,-1], stats=args.stats, save=True, save_dir=save_path_ellipse)
                        conf_mat = matrix_string(rpca_confusion_matrix)
                        print("*** {} ***".format(classifier_name))
                        print("\trpca Score: %.6f \n %s" %(rpca_score,conf_mat))



if __name__ == "__main__":

      print("\n")
      parser = argparse.ArgumentParser(prog='EIVA R&D Project',
                                          description='''This script provides the ability to train machine 
                                          learning algorithms to classify noisy samples in a sonar 3D point cloud data set provided by EIVA''')

      parser.add_argument(    'path', 
                              help='Input path and name of file containing the data')
      
      parser.add_argument('-save_path', 
                              help='Results save path',
                              type=str,
                              default="Results")
                 
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

      print("Using the following arguments:"
            "\n", args, "\n")
      
      main(data_path_head=head,data_path_tail=tail,args=args)

      if args.figsshow:
            if sys.flags.interactive != 1:
                  vispy.app.run()
      exit()