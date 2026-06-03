import numpy as np

def create_distance_array(cx: float, cy: float ,shape: tuple) -> np.ndarray:
    """creates an array where each element is euclidian distance from point (h,k)
    
    Args:
        cx: x-axis value of point to calulcate distance from
        cy: y-axis value of point to calulcate distance from
        shape: the shape of the array (assumes 2D array)
        
    Returns:
        distance matrix
    """
    
    # Create coordinate arrays
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx, yy = np.meshgrid(x, y)
    
    return np.sqrt((xx - cx)**2 + (yy - cy)**2)