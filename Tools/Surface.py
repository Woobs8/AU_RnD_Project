from scipy.interpolate import griddata
import numpy as np
from sklearn.neighbors import KDTree

def surface_reconstruction(data, resolution=[100,100], interpolation='linear'):
    # create mesh grid
    max_vals = np.max(data[:,0:3], axis=0)
    min_vals = np.min(data[:,0:3], axis=0)
    val_range = max_vals - min_vals   
    x = np.linspace(min_vals[0], max_vals[0], resolution[0])
    y = np.linspace(min_vals[1], max_vals[1], resolution[1])
    grid_x, grid_y = np.meshgrid(x, y)

    # estimate surface z-values by interpolating from known points
    grid_z = griddata(data[:,:2], data[:,2], (grid_x,grid_y), method=interpolation)
    
    # construct grid 
    grid = np.column_stack([grid_x.ravel(),grid_y.ravel(), grid_z.ravel()])

    # return only strictly numeric rows (no NaN elements)
    return grid[~np.isnan(grid).any(axis=1)]