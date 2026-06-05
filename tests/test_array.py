import pytest
import numpy as np

from gofher.array import create_distance_array, create_angle_array, create_major_axis_angle_array

#TODO: test meshgrid code 

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

#TODO: write this:
def test_create_distance_array():
    pass

@pytest.mark.parametrize(
    "cx, cy, theta, shape, expected_exception",
    [
        ([], 10, np.pi * 0.25, (100,100), ValueError),    #Case 1: cx not floatable
        (27, "a", 0.0, (50,50), ValueError),    #Case 2: cy not floatable
        (35, 47, "0.0", (75,75), ValueError),    #Case 3: theta not float
        (49, 61, 0.0, (50), ValueError),    #Case 4: shape not 2D
    ]
)
def test_create_distance_array_exceptions(cx, cy, theta, shape, expected_exception):
    with pytest.raises(expected_exception):
        create_angle_array(cx, cy, theta, shape)

#TODO: write this:
def test_create_distance_array():
    pass

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