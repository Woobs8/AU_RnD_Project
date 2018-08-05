import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy_indexed as npi
from sklearn.neighbors import KDTree


def datagrid(Data, grid=[10,10,0], window_type='rectangle', overlap=0):
    if window_type == 'rectangle':
        if len(grid) == 2:
            grid = grid + [0]
        data_idx_list = cuboid_grid__(Data,grid,overlap)
    elif window_type == 'cuboid':
        data_idx_list = cuboid_grid__(Data,grid,overlap)
    elif window_type == 'ellipse':
        if len(grid) == 2:
            grid = grid + [0]
        data_idx_list = ellipsoid_grid__(Data,grid,overlap)
    elif window_type == 'ellipsoid':
        data_idx_list = ellipsoid_grid__(Data,grid,overlap)
    else:
        print("ERROR: Window type not recognized")
        exit(1)
    return data_idx_list

def cuboid_grid__(Data,grid_bounds,overlap):
    if len(grid_bounds) != 3:
        print("ERROR: Parameter grid_bounds must be an array of 3. A grid bound for each direction x, y and z. If grid z=0 a rectangular grid is made")
        exit(1)
    if not (overlap >=0 and overlap < 1):
        print("ERROR: Overlap not possible, the value must be between {0 < overlap < 1}; If the value is 0 no overlap is used.")
        exit(1)
    if  overlap > 0 and (grid_bounds[0] % 2 == 0 or grid_bounds[1] % 2 == 0 or ( grid_bounds[2] % 2 == 0 and grid_bounds[2] > 0 ) ):
        print("ERROR: When overlap is used an odd number of grids in x, y and z direction must be used.")
        exit(1)

    # Constants for readability
    x = 0
    y = 1
    z = 2
    use_z = grid_bounds[z] > 0

    # Number of grids
    if use_z:
        num_grids = grid_bounds[x]*grid_bounds[y]*grid_bounds[z]
    else:
        num_grids = grid_bounds[x]*grid_bounds[y]
    
    # List to contain grid
    grid_idx_list = [None] * num_grids

    # Scale Data into grid numpers for each axis
    if use_z:
        datamin = np.min(Data[:,:3],axis=0)
        datamax = np.max(Data[:,:3],axis=0)   
    else:
        datamin = np.min(Data[:,:2],axis=0)
        datamax = np.max(Data[:,:2],axis=0)

    if overlap == 0: # Efficient and fast implementation when not using overlap
        # Create grid index array to match a point to a grid
        if use_z:
            Data_scaled = np.floor(np.nextafter(np.array(grid_bounds),0)[0:3]*(Data[:,:3] - datamin)/(datamax-datamin))
            data_grid_idxs = Data_scaled[:,x]+grid_bounds[x]*Data_scaled[:,y]+grid_bounds[x]*grid_bounds[y]*Data_scaled[:,z]
        else:
            Data_scaled = np.floor(np.nextafter(np.array(grid_bounds),0)[0:2]*(Data[:,:2] - datamin)/(datamax-datamin))
            data_grid_idxs = Data_scaled[:,x]+grid_bounds[x]*Data_scaled[:,y]
        
        # Split data indexes into correct data grids
        data_grid_idxs_arg_sort = np.argsort(data_grid_idxs)
        u, x_split_at = np.unique(data_grid_idxs[data_grid_idxs_arg_sort],return_index=True)
        grid_idx_list = np.split(data_grid_idxs_arg_sort,x_split_at)
        grid_idx_list = grid_idx_list+[np.array([])]*(num_grids-len(grid_idx_list)) # Fill in empty numpy arrays for the grids that had no samples in it
    else: # inefficient implementation by easy: we use kdtree en each dimension
        # Scale variables each axis so that the rectangles will become regular squares.
        grid_bounds_ratio = (np.array(grid_bounds)-1) / np.max(np.array(grid_bounds))
        
        # KDTree of the data
        # The Data given to KDtree is firstly scaled so that each axis goes from 0->1. Afterwards multiplied with grid_bound ratio so that elipsoids becomes spheres 
        if use_z:
            tree_x = KDTree((Data[:,[x]]-datamin[x])/datamax[x]*grid_bounds_ratio[x],leaf_size=1000)
            tree_y = KDTree((Data[:,[y]]-datamin[y])/datamax[y]*grid_bounds_ratio[y],leaf_size=1000)
            tree_z = KDTree((Data[:,[z]]-datamin[z])/datamax[z]*grid_bounds_ratio[z],leaf_size=1000)
        else:
            tree_x = KDTree((Data[:,[x]]-datamin[x])/datamax[x]*grid_bounds_ratio[x],leaf_size=1000)
            tree_y = KDTree((Data[:,[y]]-datamin[y])/datamax[y]*grid_bounds_ratio[y],leaf_size=1000)
        
        # Create rectangle/cuboid centers to use based on the specified number of cuboids in each direction
        x_centers = np.arange(0,1+1/grid_bounds[x],1/(grid_bounds[x]-1))*grid_bounds_ratio[x]
        y_centers = np.arange(0,1+1/grid_bounds[y],1/(grid_bounds[y]-1))*grid_bounds_ratio[y]
        
        if use_z:
            z_centers = np.arange(0,1+1/grid_bounds[z],1/(grid_bounds[z]-1))*grid_bounds_ratio[z]
            cuboid_centers = np.array(np.meshgrid(x_centers,y_centers,z_centers)).T.reshape(-1,3)
        else:
            cuboid_centers = np.array(np.meshgrid(x_centers,y_centers)).T.reshape(-1,2)

        # Query of all cuboids 
        if use_z:
            grid_idx_list_x = tree_x.query_radius(cuboid_centers[:,[x]].tolist(),x_centers[1]*(1+2*overlap)) # x_centers[1]=y_centers[1]=z_centers[1] will contain the 1/2 the length of the cube
            grid_idx_list_y = tree_y.query_radius(cuboid_centers[:,[y]].tolist(),x_centers[1]*(1+2*overlap)) # x_centers[1]=y_centers[1]=z_centers[1] will contain the 1/2 the length of the cube
            grid_idx_list_z = tree_z.query_radius(cuboid_centers[:,[z]].tolist(),x_centers[1]*(1+2*overlap)) # x_centers[1]=y_centers[1]=z_centers[1] will contain the 1/2 the length of the cube
            grid_idx_list = [npi.intersection(grid_idx_list_x[i],grid_idx_list_y[i],grid_idx_list_z[i]) for i in range(len(grid_idx_list)) ]
        else:
            grid_idx_list_x = tree_x.query_radius(cuboid_centers[:,[x]],x_centers[1]*(0.5+overlap)) # x_centers[1]=y_centers[1]=z_centers[1] will contain the 1/2 the length of the cube
            grid_idx_list_y = tree_y.query_radius(cuboid_centers[:,[y]],x_centers[1]*(0.5+overlap)) # x_centers[1]=y_centers[1]=z_centers[1] will contain the 1/2 the length of the cube
            grid_idx_list = [np.intersect1d(grid_idx_list_x[i],grid_idx_list_y[i]) for i in range(len(grid_idx_list)) ]
   
    return grid_idx_list

def ellipsoid_grid__(Data,grid_bounds,overlap):
    if len(grid_bounds) != 3:
        print("ERROR: Parameter grid_bounds must be an array of 3. A grid bound for each direction x, y and z. If grid z=0 a rectangular grid is made")
        exit(1)
    if not (overlap >=0.5 and overlap < 1):
        print("ERROR: Overlap not possible, the value must be between {0.5 =< overlap < 1}; This is to ensure all points are covered with ellipses.")
        exit(1)
    if  grid_bounds[0] % 2 == 0 or grid_bounds[1] % 2 == 0 or ( grid_bounds[2] % 2 == 0 and grid_bounds[2] > 0 ):
        print("WARNING: An odd number of grids in x, y and z direction is adviced as it gives best numerical results.")

    # Constants for readability
    x = 0
    y = 1
    z = 2
    use_z = grid_bounds[z] > 0

    # Scale variables each axis so that the elipsoids will become regular spheres. This will allow for using KDtree for creating the grid
    grid_bounds_ratio = (np.array(grid_bounds)-1) / np.max(np.array(grid_bounds))
    if use_z:
        datamin = np.min(Data[:,:3],axis=0)
        datamax = np.max(Data[:,:3],axis=0)
    else:
        datamin = np.min(Data[:,:2],axis=0)
        datamax = np.max(Data[:,:2],axis=0)

    # KDTree of the data
    # The Data given to KDtree is firstly scaled so that each axis goes from 0->1. Afterwards multiplied with grid_bound ratio so that elipsoids becomes spheres 
    if use_z:
        tree = KDTree((Data[:,:3]-datamin)/(datamax-datamin)*grid_bounds_ratio,leaf_size=1000)
    else:
        tree = KDTree((Data[:,:2]-datamin)/(datamax-datamin)*grid_bounds_ratio[x:y],leaf_size=1000)

    # Create sphere/circle centers to use based on the specified number of spheres in each direction
    x_centers = np.arange(0,1+1/grid_bounds[x],1/(grid_bounds[x]-1))*grid_bounds_ratio[x]
    y_centers = np.arange(0,1+1/grid_bounds[y],1/(grid_bounds[y]-1))*grid_bounds_ratio[y]

    if use_z:
        z_centers = np.arange(0,1+1/grid_bounds[z],1/(grid_bounds[z]-1))*grid_bounds_ratio[z]
        sphere_centers = np.array(np.meshgrid(x_centers,y_centers,z_centers)).T.reshape(-1,3)
    else:
        sphere_centers = np.array(np.meshgrid(x_centers,y_centers)).T.reshape(-1,2)
    # Query of all spheres 
    if use_z:
        grid_idx_list = tree.query_radius(sphere_centers.tolist(),x_centers[1]*(0.5+overlap)) # x_centers[1]=y_centers[1]=z_centers[1] will contain the radius of spheres with an overlap of 50%
    else:
        grid_idx_list = tree.query_radius(sphere_centers.tolist(),x_centers[1]*(0.5+overlap)) # x_centers[1]=y_centers[1] will contain the radius of spheres with an overlap of 50%
    return grid_idx_list