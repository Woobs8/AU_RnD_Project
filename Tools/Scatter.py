# -*- coding: utf-8 -*-

import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy.color import Colormap
import os
import argparse

def ScatterPlot3D(data, labels=None, x_feat=0, y_feat=1, z_feat=2, label_feat=-1, title="Scatterplot"):
    """"
    Creates a 3D scatterplot of data with unique colors for each label
    :param
        @data: matrix of row vectors containing data feature and optionally labels
        @labels: optional vector of labels for the samples in @data
        @x_feat: which feature in @data to use for the x-axis
        @y_feat: which feature in @data to use for the y-axis
        @z_feat: which feature in @data to use for the z-axis
        @label_feat: which column in @data to use as labels (ignored if @labels is provided)
        @title: plot window title
    :returns
        None
    """
    # Extract data and labels if no label vector is supplied
    if labels is None:
        labels = data[:,label_feat]
        data = np.delete(data,label_feat,axis=1)

    # Create a canvas and add view
    canvas = vispy.scene.SceneCanvas(title=title,keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    # Separate data into point clouds based on label
    unique_labels = np.unique(labels)
    cm = Colormap(['r', 'g', 'b'])
    for y in unique_labels:
        cloud = data[np.where(labels.ravel()==y),:]
        cloud = np.squeeze(cloud,axis=0)

        # Define Colors
        y = y/len(unique_labels)
        color = cm[y]

        # Create scatter object and fill in the data
        scatter = visuals.Markers()
        scatter.set_data(cloud[:,[x_feat,y_feat,z_feat]], edge_color=None, face_color=color, size=5)
        view.add(scatter)

    # Define camara rotation
    view.camera = 'turntable'  # or try 'arcball'

    # Add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    from DataLoader import load_and_split_data, load_raw_data
    import sys

    parser = argparse.ArgumentParser(prog='3D Scatterplot',
                                    description='''Creates a 3D scatter of plot of data in specified file''')
    parser.add_argument('path', 
                        help='Input path and name of file containing the data to be plotted')
    parser.add_argument('-x', 
                        help='Specify which feature to utilize in the X-dimension', 
                        metavar='x feature',
                        required=False, 
                        default=0,
                        type=int, 
                        choices=[0,1,2,3,4,5,6,7])
    parser.add_argument('-y', 
                        help='Specify which feature to utilize in the Y-dimension', 
                        metavar='y feature',
                        required=False, 
                        default=1,
                        type=int, 
                        choices=[0,1,2,3,4,5,6,7])
    parser.add_argument('-z', 
                        help='Specify which feature to utilize in the Z-dimension', 
                        metavar='z feature',
                        required=False, 
                        default=2,
                        type=int, 
                        choices=[0,1,2,3,4,5,6,7])
    parser.add_argument('-lbl', 
                        help='Specify which column to utilize as the label', 
                        metavar='label',
                        required=False, 
                        default=-1,     # end
                        type=int, 
                        choices=[0,1,2,3,4,5,6,7,-1])

    #  Parse Arguments
    args = parser.parse_args()

    # Path is a directory (containing training and test data from prior classification is the assumption)
    if os.path.isdir(args.path):
        if not os.path.exists(args.path+'training_data.txt') or not os.path.exists(args.path+'test_data.txt') or not os.path.exists(args.path+'test_classification.txt'):
            print('Error: directory does not contain the expected files')
            exit(1)

        # Read training and test data from files
        if os.path.exists(args.path+'training_data.txt'):
            train_data = load_raw_data('training_data.txt', args.path)
        if os.path.exists(args.path+'test_data.txt'):
            test_data = load_raw_data('test_data.txt', args.path)
        if os.path.exists(args.path+'test_classification.txt'):
            test_classification = load_raw_data('test_classification.txt',args.path)

        # Plot Training data
        ScatterPlot3D(train_data,x_feat=args.x,y_feat=args.y,z_feat=args.z,label_feat=-1, title='Labeled Training Data')
        if os.path.exists(args.path+'training_classification.txt'):
            train_classification = load_raw_data('training_classification.txt',args.path)
            ScatterPlot3D(train_data, labels=train_classification, x_feat=args.x,y_feat=args.y,z_feat=args.z, title='Classified Training Data')

        # Plot Test data
        ScatterPlot3D(train_data,x_feat=args.x,y_feat=args.y,z_feat=args.z,label_feat=-1, title='Labeled Test Data')
        ScatterPlot3D(train_data, labels=test_classification, x_feat=args.x,y_feat=args.y,z_feat=args.z, title='Classified Test Data')
    
    # Path is a data file
    elif os.path.exists(args.path):
        # Get file name and path from argument
        head, tail = os.path.split(args.path)

        # Read data from file
        raw_data = load_raw_data(tail, head)

        # Plot data
        ScatterPlot3D(raw_data,x_feat=args.x,y_feat=args.y,z_feat=args.z,label_feat=args.lbl, title=tail)
    else:
        print("Error: file '" + args.path + "' not found")
        exit(1)
    
    if sys.flags.interactive != 1:
        vispy.app.run()
    