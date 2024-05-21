import numpy as np

def create_centered_mesh_grid(h,k,shape):
    """Create two meshes whose center (0) is specific be h,k
    
    Args:
        h: x cordinate of center to use
        k: y cordinate of center to use
        shape: shape of matrix
        
    Returns:
        xv, yv - xv is a mesh centered around h, yv is a mesh centered around k
    """
    (rows,cols) = shape
    x=np.arange(0,cols)-h 
    y=np.arange(0,rows)-k
    
    xv, yv = np.meshgrid(x, y)
    return xv, yv

def create_angle_matrix(xv,yv,theta):
    """creates an array where each pixel if the angle from theta
    
    Args:
        xv: x centered_mesh 
        yv: y centered_mesh 
        
    Returns:
        angle matrix
    """
    return np.arctan2(yv,xv)-theta

def create_dist_matrix(xv,yv):
    """creates an array where each pixel is the distance from the pixel xv,yv
    
    Args:
        xv: x centered_mesh 
        yv: y centered_mesh 
        
    Returns:
        distance matrix
    """
    return np.sqrt(xv**2 + yv**2)

def create_disk_angle_matrix(h,k,theta,shape):
    """creates disk angle matrix - angle from positive major axis
    Important: return values in range [0,2*pi]
    
    Args:
        h: x cordinate of center
        k: y cordinate of center
        theta: angle in radians of semi-major axis from positive x-axis of Cartesian graph
        shape: shape of disk angle matrix
        
    Returns:
        disk angle matrix
    """
    #Step 1) Create a mesh and angle matrix:
    xv, yv = create_centered_mesh_grid(h,k,shape)
    angle_matrix = create_angle_matrix(xv,yv,theta)
    
    #Step 2) Create a matrix with angles [0,2*pi] from the POS semi-major axis
    return np.mod(angle_matrix,2*np.pi)

def create_major_axis_angle_matrix(h,k,theta,shape):
    """"creates major axis angle matrix - angle from major axis
    Important: return values in range [-pi/2,pi/2]
    
    Args:
        h: x cordinate of center
        k: y cordinate of center
        theta: angle in radians of semi-major axis from positive x-axis of Cartesian graph
        shape: shape of disk angle matrix
        
    Returns:
        disk major axis angle matrix
    """
    #Step 1) Create a mesh and angle matrix (with additional offset of +pi/2):
    xv, yv = create_centered_mesh_grid(h,k,shape)
    angle_matrix = create_angle_matrix(xv,yv,theta)+np.pi/2
    
    #Step 2) Create a matrix with angles [-pi/2,pi/2] from the major axis:
    return np.abs(np.mod(angle_matrix,np.pi))-np.pi/2

def create_minor_axis_angle_matrix(h,k,theta,shape):
    """"creates major axis angle matrix - angle from minor axis
    Important: return values in range [-pi/2,pi/2]
    
    Args:
        h: x cordinate of center
        k: y cordinate of center
        theta: angle in radians of semi-major axis from positive x-axis of Cartesian graph
        shape: shape of disk angle matrix
        
    Returns:
        disk minor axis angle matrix
    """
    #Step 1) Create a mesh and angle matrix:
    xv, yv = create_centered_mesh_grid(h,k,shape)
    angle_matrix = create_angle_matrix(xv,yv,theta)
    
    #Step 2) Create a matrix with angles [-pi/2,pi/2] from the minor axis:
    return np.abs(np.mod(angle_matrix,np.pi))-np.pi/2

def normalize_matrix(data,to_diff_mask):
    """"Normalize all values in data masked by to_diff_mask so that values are in range [0,1] if in to_diff_mask, 0 else
    Important: return values in range [-pi/2,pi/2]
    
    Args:
        data - data to mask to normalize on
        to_diff_mask:  a single boolean mask indicating where to create diff (1's = included, 0's = ignored)
        
    Returns:
        normalized data
    """
    normalized = np.zeros(data.shape)
    the_min = np.min(data[to_diff_mask]); the_max = np.max(data[to_diff_mask])
    normalized[to_diff_mask] = (data[to_diff_mask]- the_min)/(the_max-the_min)
    return normalized
