import numpy as np

def create_centered_mesh_grid(h: float, k: float, shape: tuple[int,int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Create two meshes whose center (0) is specific.

    Args:
        h (float): x coordinate of center.
        k (float): y coordinate of center.
        shape (tuple[int, int]): shape of the matrix.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: xv is a mesh centered around h,
            yv is a mesh centered around k.
    """
    (rows,cols) = shape
    x=np.arange(0,cols)-h 
    y=np.arange(0,rows)-k
    
    xv, yv = np.meshgrid(x, y)
    return xv, yv

def create_angle_matrix(xv: np.ndarray, yv: np.ndarray, theta: float) -> np.ndarray:
    """
    Creates an array where each pixel is the angle from the given angle theta.

    Args:
        xv (np.ndarray): x centered mesh.
        yv (np.ndarray): y centered mesh.
        theta (float): Angle of line from positive x-axis measure counter clockwise.

    Returns:
        np.ndarray: Angle matrix where each pixel is the angle from line specified by theta.
    """
    return np.arctan2(yv,xv)-theta

def create_dist_matrix(xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    """
    Create an array where each pixel is the distance from the pixel (xv, yv).

    Args:
        xv (np.ndarray): x centered mesh.
        yv (np.ndarray): y centered mesh.

    Returns:
        np.ndarray: Distance matrix from (x, y) which is specified by center of centered meshes (xv, yv)
    """
    return np.sqrt(xv**2 + yv**2)

def create_disk_angle_matrix(h: float, k: float, theta: float, shape: tuple[int,int]) -> np.ndarray:
    """
    Creates a disk angle matrix - angle from the positive major axis.

    Important:
        Return values are in the range [0, 2*pi].

    Args:
        h (float): x coordinate of center.
        k (float): y coordinate of center.
        theta (float): Angle in radians of the semi-major axis from the positive
            x-axis of the Cartesian graph.
        shape (tuple[int, int]): Shape of the disk angle matrix.

    Returns:
        numpy.ndarray: Disk angle matrix with values in range [0, 2*pi].
    """
    #Step 1) Create a mesh and angle matrix:
    xv, yv = create_centered_mesh_grid(h,k,shape)
    angle_matrix = create_angle_matrix(xv,yv,theta)
    
    #Step 2) Create a matrix with angles [0,2*pi] from the POS semi-major axis
    return np.mod(angle_matrix,2*np.pi)

def create_major_axis_angle_matrix(h: float, k: float, theta: float, shape: tuple[int,int]) -> np.ndarray:
    """
    Creates a major axis angle matrix - angle from the major axis.

    Args:
        h (float): x coordinate of center.
        k (float): y coordinate of center.
        theta (float): Angle in radians of semi-major axis from positive x-axis
            of Cartesian graph.
        shape (tuple[int, int]): Shape of the major axis angle matrix.

    Returns:
        numpy.ndarray: Major axis angle matrix with values in range [-pi/2, pi/2].
    """
    #Step 1) Create a mesh and angle matrix (with additional offset of +pi/2):
    xv, yv = create_centered_mesh_grid(h,k,shape)
    angle_matrix = create_angle_matrix(xv,yv,theta)+np.pi/2
    
    #Step 2) Create a matrix with angles [-pi/2,pi/2] from the major axis:
    return np.abs(np.mod(angle_matrix,np.pi))-np.pi/2

def create_minor_axis_angle_matrix(h: float, k: float, theta: float, shape: tuple[int,int]) -> np.ndarray:
    """
    Creates a minor axis angle matrix - angle from minor axis.

    Args:
        h (float): x coordinate of center.
        k (float): y coordinate of center.
        theta (float): angle in radians of semi-major axis from positive x-axis
            of Cartesian graph.
        shape (tuple[int, int]): shape of disk angle matrix.

    Returns:
        numpy.ndarray: Disk minor axis angle matrix with values in range [-pi/2, pi/2].
    """
    #Step 1) Create a mesh and angle matrix:
    xv, yv = create_centered_mesh_grid(h,k,shape)
    angle_matrix = create_angle_matrix(xv,yv,theta)
    
    #Step 2) Create a matrix with angles [-pi/2,pi/2] from the minor axis:
    return np.abs(np.mod(angle_matrix,np.pi))-np.pi/2

def normalize_matrix(data: np.ndarray, to_diff_mask: np.ndarray) -> np.ndarray:
    """
    Normalize all values in `data` masked by `to_diff_mask` so that values are in range [0, 1] if in `to_diff_mask`,
    0 otherwise.

    Args:
        data: Data to mask to normalize on.
        to_diff_mask: A single boolean mask indicating where to create diff (1's = included, 0's = ignored).

    Returns:
        Normalized data.
    """
    normalized = np.zeros(data.shape)
    the_min = np.min(data[to_diff_mask]); the_max = np.max(data[to_diff_mask])
    normalized[to_diff_mask] = (data[to_diff_mask]- the_min)/(the_max-the_min)
    return normalized
