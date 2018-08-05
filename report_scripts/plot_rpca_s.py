import sys
import os
sys.path.append(os.getcwd())

import argparse
import vispy.scene
from Tools.DataLoader import load_raw_data, split_data
from Classifiers.RPCA.RPCA import RobustPCAGrid
from Classifiers.Classify import classify
from Tools.Scatter import ScatterPlot3D
import dill


import numpy as np
import os
import sys

# Global Variables
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/Data'

def main(head,tail,rpca_path,s_idx,classification_path=None):
      ### ********** LOAD DATA ********** ###
      data = load_raw_data(tail, head)
      if classification_path:
            # Open specified file
            test_prediction= np.loadtxt(classification_path)

      # Open specified file
      with open(rpca_path, 'rb') as f:
            rpca = dill.load(f)
      if s_idx<=0:
            ScatterPlot3D(rpca.get_full_S(),labels=data[:,-1], title="Full S of {} With True labels".format(rpca.name) )
            if classification_path:
                  ScatterPlot3D(rpca.get_full_S(),labels=test_prediction, title="Full S of {} With Classified labels".format(rpca.name) )
      else:
            ScatterPlot3D(rpca.S_list[s_idx],labels=data[rpca.sample_list[s_idx],-1], title="Subset S with index {} of {} With True labels".format(s_idx,rpca.name) )
            if classification_path:
                  ScatterPlot3D(rpca.S_list[s_idx],labels=test_prediction[s_idx], title="Subset S with index {} of {} With Classified labels".format(s_idx,rpca.name) )
      if sys.flags.interactive != 1:
                  vispy.app.run()



if __name__ == "__main__":

      print("\n")
      parser = argparse.ArgumentParser(prog='EIVA R&D Project',
                                          description='''This script provides the ability to plot the full or partial S matrixes of a saved RPCA object''')

      parser.add_argument(    'path', 
                              help='Input path and name of file containing the data to be plotted')

      parser.add_argument('rpca_path', 
                          help='RPCA object path path and name of file containing the data to be plotted')
      
      parser.add_argument('-classifications', 
                          help='Classifications path and name of file containing the data to be plotted',
                          default=False)

      parser.add_argument('-s_idx', 
                          help='Specify index for S to plot from S_list in RPCA object, otherwise full S is printed',
                          type=int,
                          default=-1)
      
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
      
      main(head=head,tail=tail,rpca_path=args.rpca_path,s_idx=args.s_idx,classification_path=args.classifications)
            
      exit()