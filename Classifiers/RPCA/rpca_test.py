import numpy as np
from RPCA import RobustPCAGrid
import vispy.scene
from vispy.scene import visuals
from vispy.color import Colormap
import argparse
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
from Tools.Scatter import ScatterPlot3D
from Tools.DataLoader import load_and_split_data, load_raw_data
from Tools.Preprocess import ScaleData, PcaRotation, ScaleFeatures

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Robust PCA',
                                    description='''Test of Robust PCA''')
    parser.add_argument('path', 
                        help='Input path and name of file containing the data to be plotted')

    #  Parse Arguments
    args = parser.parse_args()
    
    # Path is a data file
    if os.path.exists(args.path):
        print("Test")
        # Get file name and path from argument
        head, tail = os.path.split(args.path)

        # Read data from file
        data = load_raw_data(tail, head)[:,:]
        ### ********** PREPROCESS DATA ********** ###
        print("\n-- Preprocessing Data  --\n")
        # preprocess true coordinates
        data[:,:3] = PcaRotation(data[:,:3])
        data[:,:3] = ScaleData(data[:,:3])

        # Grid RPCA
        rpca = RobustPCAGrid([101,101], max_iter=1000, overlap=0.5, window_type='rectangle',predict_method='voting') #50,50
        rpca.fit(data[:,:3])   
        labels = rpca.predict(None)
        ScatterPlot3D(data[:,:3],labels=labels, title="Predicted labels")
        ScatterPlot3D(rpca.S,labels=data[:,-1], title="S With True labels")
        
        print(len(rpca.S_list))
        #ScatterPlot3D(rpca.S_list[20],labels=data[rpca.sample_list[20],-1], title="S1 With True labels")
        #ScatterPlot3D(rpca.S_list[1900],labels=data[rpca.sample_list[1900],-1], title="S2 With True labels")
        #vispy.app.run()
        #ScatterPlot3D(rpca.S_list[2700],labels=data[rpca.sample_list[2700],-1], title="S3 With True labels")
        #vispy.app.run()
        #ScatterPlot3D(rpca.S_list[3700],labels=data[rpca.sample_list[3700],-1], title="S4 With True labels")

    else:
        print("Error: file '" + args.path + "' not found")
        exit(1)
    
    if sys.flags.interactive != 1:
        vispy.app.run()
    