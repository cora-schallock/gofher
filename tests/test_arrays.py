"""Test all functions in gofher/array.py

The script is run using the commend: python -m pytest
"""


import pytest
import numpy as np

from arrays import (
    create_distance_array, 
    create_angle_array, 
    create_major_axis_angle_array, 
    create_minor_axis_angle_array, 
    normalize_array
)

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
    """Tests exceptions expected from distance array"""
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
    """Test exceptions expected from angle array"""
    with pytest.raises(expected_exception):
        create_angle_array(cx, cy, theta, shape)

def test_create_angle_array():
    """Create an angle array where each angle is measured
    counter clockwise from positive line specified by point
    (cx,cy) at slope theta (with respect to positive x-axis)
    
    Tolerance:
        residule < 0.01 rads
    """
    angle_array = create_angle_array(30,30,np.pi/4,(60,60))

    # angle between line through (30,30) with slope pi/4 and (30,35) is pi/4:
    residual = abs(angle_array[35,30] - np.pi/4)
    assert residual < 0.01 

    # angle between line through (30,30) with slope pi/4 and (40,30) is -pi/4:
    residual = abs(angle_array[30,40] + np.pi/4)
    assert residual < 0.01 

    # angle between line through (30,30) with slope pi/4 and (15,15) is 0/2pi:
    residual = abs(angle_array[15,15] % np.pi)
    assert residual < 0.01 

@pytest.mark.parametrize(
    "cx, cy, theta, shape, expected_exception",
    [
        ("a", 5, np.pi * 0.25, (100,100), ValueError),    #Case 1: cx not floatable
        (27, {}, 0.0, (50,50), ValueError),    #Case 2: cy not floatable
        (35, 47, [], (75,75), ValueError),    #Case 3: theta not float
        (49, 61, 0.0, (50.25,40), ValueError),    #Case 4: shape not 2 ints
    ]
)
def test_create_major_axis_angle_array_exceptions(cx, cy, theta, shape, expected_exception):
    """Test exceptions expected from major axis angle array"""
    with pytest.raises(expected_exception):
        create_major_axis_angle_array(cx, cy, theta, shape)

def test_create_major_axis_angle_array():
    """Creates an array of angles from major axis of ellipse
    
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
    bottom_side = np.abs(major_axis_array[50:100,:])

    # Across y-axis right side reflected both vertically & horizontally is left
    residual = top_side - np.flip(bottom_side,axis=0)

    # Checks symmetry by allowing max residual to be at most +/-0.01*pi rads
    max_residule = np.max(np.abs(residual))
    assert max_residule < 0.01

    # Checks symmetry by allowing avg. differenceto be at most +/-0.001*pi rads
    mean_residule = np.mean(np.abs(residual))
    assert mean_residule <= 0.001

@pytest.mark.parametrize(
    "cx, cy, theta, shape, expected_exception",
    [
        ("a", 5, np.pi * 0.25, (100,100), ValueError),    #Case 1: cx not floatable
        (27, {}, 0.0, (50,50), ValueError),    #Case 2: cy not floatable
        (35, 47, [], (75,75), ValueError),    #Case 3: theta not float
        (49, 61, 0.0, (50.25,40), ValueError),    #Case 4: shape not 2 ints
    ]
)
def test_create_minor_axis_angle_array_exceptions(cx, cy, theta, shape, expected_exception):
    """Test exceptions expected from minor axis angle array"""
    with pytest.raises(expected_exception):
        create_major_axis_angle_array(cx, cy, theta, shape)

def test_create_minor_axis_angle_array():
    """Creates an array of angles from minor axis of ellipse
    
    Tolerance:
        residule < 0.01 rads
    """

    # Create angle array for a ellipse with -pi/4 slope
    # that goes through (25,25), hence minor axis at pi/4:
    angle_array = create_minor_axis_angle_array(25, 25, -np.pi/4, (50,50))

    # (28,28) is on the same diagonal line as minor axis
    # so value should be near 0 rads:
    residual = np.abs(angle_array[28,28])
    assert residual < 0.01

    # (25,30) is on y-axis, and minor axis is on diagonal line 
    # with np.pi/4, so it should be np.pi/4:
    diagonal_residual = np.abs(angle_array[30,25] - np.pi/4)
    assert diagonal_residual < 0.01

    # (30,20) is on major axis, counter clockwise from (25,25)
    #  so should be -np.pi/2:
    orthogonal_residual = np.abs(angle_array[20,30] + np.pi/2)
    assert orthogonal_residual < 0.01

def test_create_minor_axis_angle_array_horizontal_symmetry():
    """Test symmetry of minor axis angle array across minor axis

    Tolerance:
        max_residule < 0.01 rads
        mean_residule < 0.001 rads
    
    For this case we are using major axis is alligned with positive 
    x-axis so the minor axis is alligned with the positive y-axis.
    Hence horizontal_symmetry (i.e. across y-axis) is really in 
    reference to across mior axis.
    
    Code format note: asserts are 2 seperate lines for error message readability
    """
    # Middle of each entry treated as center, so x axis is shifted by 0.5 to ensure symmetry:
    minor_axis = create_minor_axis_angle_array(49.5, 49.5, 0.0, (100,100))

    left_side = minor_axis[:,0:50]
    right_side = minor_axis[:,50:100]

    # Right side is flipped vertically then horizontally
    residual = left_side - np.flip(np.flip(right_side,axis=0), axis=1)

    # Checks symmetry by allowing max residual to be at most +/-0.01*pi rads
    max_residule = np.max(np.abs(residual)) 
    assert max_residule < 0.01

    # Checks symmetry by allowing avg. differenceto be at most +/-0.001*pi rads
    mean_residule = np.mean(np.abs(residual))
    assert mean_residule <= 0.001
   
def test_create_minor_axis_angle_array_vertical_symmetry():
    """Test symmetry of minor axis angle array across major axis

    Tolerance:
        max_residule < 0.01 rads
        mean_residule < 0.001 rads
    
    For this case we are using major axis is alligned with positive 
    x-aaxis so the minor axis is alligned with the positive y-axis.
    Hence vertical symmetry (i.e. across x-axis) is really in 
    reference to across major axis.
    
    Code format note: asserts are 2 seperate lines for error message readability
    """
    # Middle of each entry treated as center, so x axis is shifted by 0.5 to ensure symmetry:
    minor_axis = create_minor_axis_angle_array(49.5, 49.5, 0.0, (100,100))

    left_side = minor_axis[0:50,:]
    right_side = minor_axis[50:100,:]

    # Right side is flipped vertically then horizontally
    residual = left_side - np.flip(np.flip(right_side,axis=0), axis=1)

    # Checks symmetry by allowing max residual to be at most +/-0.01*pi rads
    max_residule = np.max(np.abs(residual)) 
    assert max_residule < 0.01

    # Checks symmetry by allowing avg. differenceto be at most +/-0.001*pi rads
    mean_residule = np.mean(np.abs(residual))
    assert mean_residule <= 0.001

@pytest.mark.parametrize(
    "array, normalize_mask, expected_exception",
    [
        (np.ones((2,2)), np.ones((2,2),np.float32), ValueError),
        (np.ones((2,2)), np.ones((3,3),bool), ValueError),
        (np.array([[0,np.nan], [0,np.inf]]), np.ones((2,2),bool), ValueError),
        (np.array([[0,0.2], [1,"a"]]),np.ones((2,2),bool), ValueError),
    ]
)
def test_normalize_array_exceptions(array, normalize_mask, expected_exception):
    """Test exceptions expected from normalize array"""
    with pytest.raises(expected_exception):
        normalize_array(array, normalize_mask)

def test_normalize_array():
    """Test normalize_array function

    Tolerance:
        max_residule < 0.01 rads
        mean_residule < 0.001 rads

    3 cases:
        1. normalize all entries
        2. normalize only entries in a signle row
        3. normalize only a single value (i.e. min ~= max)
    """
    array = np.array([[0.0, 1.0, 2.0], 
                      [4.0, 8.0, 16.0],
                      [32.0, 64.0, 128.0]],dtype=np.float32)

    normalize_all_expected = np.array([[0.0/128.0, 1.0/128.0, 2.0/128.0], 
                              [4.0/128.0, 8.0/128.0, 16.0/128.0],
                              [32.0/128.0, 64.0/128.0, 128.0/128.0]],dtype=np.float32)
    
    normalize_row_expected = np.array([[0.0, 0.0, 0.0], 
                                       [0.0, (8.0-4.0)/(16.0-4.0), 1.0],
                                       [0.0, 0.0, 0.0]],dtype=np.float32)
    
    normalize_single_expected = np.array([[0.0, 0.0, 0.0], 
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0]],dtype=np.float32)
    
    # Case 1 - Normalize the whole array and calculate the residual
    normalized_all_array = normalize_array(array)
    all_residual = np.abs(normalized_all_array-normalize_all_expected)

    # Checks max difference to be at most +/-0.01
    max_all_residual = np.max(all_residual)
    assert max_all_residual < 0.01

    # Checks avg. difference to be at most +/-0.001
    mean_all_residual = np.max(all_residual)
    assert mean_all_residual < 0.001

    # Case 2 - Normalize only the second row and calculate the residual
    second_row_mask = np.broadcast_to([[False], [True], [False]], (3, 3))
    normalize_row_array = normalize_array(array,second_row_mask)
    row_residual = np.abs(normalize_row_array-normalize_row_expected)

    # Checks max difference to be at most +/-0.01
    max_row_residual = np.max(row_residual)
    assert max_row_residual < 0.01

    # Checks avg. difference to be at most +/-0.001
    mean_row_residual = np.max(row_residual)
    assert mean_row_residual < 0.001

    # Case 3 - Normalize a single value (this test case where min ~= max)
    single_value_mask = np.zeros((3,3), dtype=bool)
    single_value_mask[2][2] = True
    normalize_single_array = normalize_array(array,single_value_mask)
    single_residual = np.abs(normalize_single_array-normalize_single_expected)

    # Checks max difference to be at most +/-0.01
    max_single_residual = np.max(single_residual)
    assert max_single_residual < 0.01

    # Checks avg. difference to be at most +/-0.001
    mean_single_residual = np.max(single_residual)
    assert mean_single_residual < 0.001
