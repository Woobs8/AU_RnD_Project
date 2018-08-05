import sys
import os
sys.path.append(os.getcwd())
import argparse
import vispy.scene
from Tools.DataLoader import load_raw_data, split_data
from Tools.Scatter import ScatterPlot3D
import numpy as np

def main(head,tail):
      ### ********** LOAD DATA ********** ###
      data = load_raw_data(tail, head)

      ### ********** FILTER DATA ********** ###
      filt_data = np.delete(data[:,:3],np.argwhere(data[:,-1] == 1)[:,0],axis=0)
      zeros = np.zeros(len(filt_data))
      zeros[-1] = 1
      
      ### ********** PLOTTING *********** ###
      ScatterPlot3D(filt_data,labels=zeros, title="True Samples Only" )
      ScatterPlot3D(data,labels=data[:,-1], title="All Data" )
      
      if sys.flags.interactive != 1:
                  vispy.app.run()



if __name__ == "__main__":

      print("\n")
      parser = argparse.ArgumentParser(prog='EIVA R&D Project',
                                          description='''This script provides the ability to plot the true data. That is the full dataset with labels and the fulldata with all noise filtered''')

      parser.add_argument(    'path', 
                              help='Input path and name of file containing the data to be plotted')
      
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
      
      main(head=head,tail=tail)
            
      exit()