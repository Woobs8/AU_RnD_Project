import sys
import os
sys.path.append(os.getcwd())

import argparse
import vispy.scene
from Tools.DataLoader import load_raw_data, split_data
from Classifiers.RPCA.RPCA import RobustPCAGrid
from Classifiers.Classify import classify
from Tools.Scatter import ScatterPlot3D
import numpy as np

def main(head,tail,classification_path):
      ### ********** LOAD DATA ********** ###
      data = load_raw_data(tail, head)

      # Open specified file
      test_prediction= np.loadtxt(classification_path)
      
      ScatterPlot3D(data[:,:3],labels=test_prediction, title="Test Prediction")
      filt_data = np.delete(data,np.argwhere(test_prediction == 1)[:,0],axis=0)
      ScatterPlot3D(filt_data[:,:3],labels=filt_data[:,-1], title="Filtered Dataset With true labels")
      
      if sys.flags.interactive != 1:
                  vispy.app.run()



if __name__ == "__main__":

      print("\n")
      parser = argparse.ArgumentParser(prog='EIVA R&D Project',
                                          description='''This script provides the ability to plot the test predictions and the filtered data st with true labels''')

      parser.add_argument(    'path', 
                              help='Input path and name of file containing the data to be plotted')

      parser.add_argument('classification_path', 
                          help='RPCA object path path and name of file containing the data to be plotted')
      
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
      
      main(head=head,tail=tail,classification_path=args.classification_path)
            
      exit()