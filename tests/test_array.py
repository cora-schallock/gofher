import pytest
import numpy as np

from gofher.array import create_distance_array, create_angle_array, create_major_axis_angle_array, create_minor_axis_angle_array, normalize_array

@pytest.mark.parametrize(
    "cx, cy, shape, expected_exception",
    [
        (10, "a", (100,100), ValueError),    #Case 1: cx not floatable 
        ([], -10, (100,100), ValueError),   #Case 2: cy not floatable 
        (50, 10, [], ValueError),    #Case 3: shape not tuple
        (10.0, 15.0, (50,50,50), ValueError),   #Case 4: shape not 2D
    ]
)
def test_create_distance_array_exceptions(cx, cy, shape, expected_exception):
    with pytest.raises(expected_exception):
        create_distance_array(cx, cy, shape)

def test_create_distance_array():
    """Create a distance array where each element is the distance
    from the point (cx,cy)
    
    Tolerance:
        residule < 0.01 rads
    """
    # Create distance array of shape (10,10) with a center of
    #  cx = 4, cy = 5
    distance_array = create_distance_array(4,5,(10,10))

    # Distance_array[5,4] is (cx=4,cy=5) because of numpy
    # indexing (i.e. [row,col]), therefore distance should be 0
    center_residual = np.abs(distance_array[5,4])
    assert center_residual < 0.01

    # The distance between point (1,1) and (4,5) is 5
    above_residual = distance_array[1,1] - 5
    assert above_residual < 0.01

    # The distance between point (8,8) and (4,5) is 5
    below_residual = distance_array[8,8] - 5
    assert below_residual < 0.01

    # The distance from (1,9) and (4,5) is equal to 
    # the distance from (7,1) and (4,5)
    equal_distance_residual = distance_array[9,1] - distance_array[1,7]
    assert equal_distance_residual < 0.01

    # The minimum distance should be 0
    assert np.min(distance_array) >= 0.0
    
@pytest.mark.parametrize(
    "cx, cy, theta, shape, expected_exception",
    [
        ([], 10, np.pi * 0.25, (100,100), ValueError),    #Case 1: cx not floatable
        (27, "a", 0.0, (50,50), ValueError),    #Case 2: cy not floatable
        (35, 47, "0.0", (75,75), ValueError),    #Case 3: theta not float
        (49, 61, 0.0, (50), ValueError),    #Case 4: shape not 2D
    ]
)
def test_create_angle_array_exceptions(cx, cy, theta, shape, expected_exception):
    with pytest.raises(expected_exception):
        create_angle_array(cx, cy, theta, shape)

def test_create_angle_array():
    """Create an angle array where each angle is measured
    counter clockwise from positive line specified by point
    (cx,cy) at slope theta (with respect to positive x-axis)
    
    Tolerance:
        residule < 0.01 rads
    """

    # Create angle array for a line with pi/4 slope, that goes through (25,25)
    angle_array = create_major_axis_angle_array(25, 25, np.pi/4, (50,50))

    # (28,28) is on the same diagonal line specified by theta
    # so value should be near 0 rads
    residual = np.abs(angle_array[28,28])
    assert residual < 0.01

    # [30,25] is right above center point so since slope of
    # line is pi/4, [30,25] should be near pi/4 rads
    diagonal_residual = np.abs(angle_array[30,25] - np.pi/4)
    assert diagonal_residual < 0.01

    # [20,30] is orthogonal to the given line from (cx,cy)
    #  so should be near pi/2 rads
    orthogonal_residual = np.abs(angle_array[20,30] + np.pi/2)
    assert orthogonal_residual < 0.01

def test_major_axis_angle_array_horizontal_symmetry():
    """Test symmetry of major axis angle array across minor axis

    Tolerance:
        max_residule < 0.01 rads
        mean_residule < 0.001 rads
    
    
    For this case we are using major axis alligned with positive x-axis
    Hence vertical_symmetry is really in reference to across major axis
    
    Code format note: asserts are 2 seperate lines for error message readability
    """
    # Middle of each entry treated as center, so x axis is shifted by 0.5 to ensure symmetry:
    major_axis_array = create_major_axis_angle_array(49.5, 49.5, 0.0, (100,100))

    left_side = np.abs(major_axis_array[:,0:50])
    right_side = np.abs(major_axis_array[:,50:100])

    # Across y-axis right side reflected both vertically & horizontally is left
    residual = left_side - np.flip(np.flip(right_side, axis = 0), axis = 1)

    # Checks symmetry by allowing max residual to be at most +/-0.01*pi rads
    max_residule = np.max(np.abs(residual)) 
    assert max_residule < 0.01

    # Checks symmetry by allowing avg. differenceto be at most +/-0.001*pi rads
    mean_residule = np.mean(np.abs(residual))
    assert mean_residule <= 0.001

def test_major_axis_angle_array_vertical_symmetry():
    """Test symmetry of major axis angle array across major axis

    Tolerance:
        max_residule < 0.01 rads
        mean_residule < 0.001 rads
    
    For this case we are using major axis alligned with positive x-axis
    Hence vertical_symmetry is really in reference to across major axis
    
    Code format note: asserts are 2 seperate lines for error message readability
    """
    # Middle of each entry treated as center, so x axis is shifted by 0.5 to ensure symmetry
    major_axis_array = create_major_axis_angle_array(49.5, 49.5, 0.0, (100,100))

    top_side = np.abs(major_axis_array[0:50,:])
    bottom_side = np.abs(major_axis_array[0:50,:])

    # Across y-axis right side reflected both vertically & horizontally is left
    residual = top_side - bottom_side

    # Checks symmetry by allowing max residual to be at most +/-0.01*pi rads
    max_residule = np.max(np.abs(residual))
    assert max_residule < 0.01

    # Checks symmetry by allowing avg. differenceto be at most +/-0.001*pi rads
    mean_residule = np.mean(np.abs(residual))
    assert mean_residule <= 0.001

def test_normalize_array():
    """Test normalize_array function

    Tolerance:
        max_residule < 0.01 rads
        mean_residule < 0.001 rads
    """
    array = np.array([[0.0, 1.0, 2.0], 
                      [4.0, 8.0, 16.0],
                      [32.0, 64.0, 128.0]],dtype=np.float32)

    normalize_all_expected = np.array([[0.0/128.0, 1.0/128.0, 2.0/128.0], 
                              [4.0/128.0, 8.0/128.0, 16.0/128.0],
                              [32.0/128.0, 64.0/128.0, 128.0/128.0]],dtype=np.float32)
    
    normalize_row_expected = np.array([[0.0, 0.0, 0.0], 
                                       [4.0/16.0, 8.0/16.0, 16.0/16.0],
                                       [0.0, 0.0, 0.0]],dtype=np.float32)
    
    # Normalize the whole array and calculate the residual
    normalized_all_array = normalize_array(array)
    all_residual = np.abs(normalized_all_array - normalize_all_expected)

    # Checks max difference to be at most +/-0.01
    max_all_residual = np.max(all_residual)
    assert max_all_residual < 0.01

    # Checks avg. difference to be at most +/-0.001
    mean_all_residual = np.max(all_residual)
    assert mean_all_residual < 0.001

    # Normalize only the second row and calculate the residual
    second_row_mask = np.broadcast_to([[False], [True], [False]], (3, 3))
    normalize_row_array = normalize_array(array,second_row_mask)
    row_residual = np.abs(normalized_all_array - normalize_all_expected)

    # Checks max difference to be at most +/-0.01
    max_row_residual = np.max(row_residual)
    assert max_row_residual < 0.01

    # Checks avg. difference to be at most +/-0.001
    mean_row_residual = np.max(row_residual)
    assert mean_row_residual < 0.001
    