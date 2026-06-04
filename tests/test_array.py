import pytest
import numpy as np

from gofher.array import create_distance_array, create_angle_array

#TODO: test meshgrid code

@pytest.mark.parametrize(
    "cx, cy, shape, expected_exception",
    [
        (10, "a", (100,100), ValueError),    #Case 1: cx not float 
        ([], -10, (100,100), ValueError),   #Case 2: cy not float 
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

#TODO: update tests:
@pytest.mark.parametrize(
    "cx, cy, theta, shape, expected_exception",
    [
        (0, 0, "0.0", (50,50), ValueError),    #Case 1: theta not float
    ]
)
def test_create_distance_array_exceptions(cx, cy, theta, shape, expected_exception):
    with pytest.raises(expected_exception):
        create_angle_array(cx, cy, theta, shape)

#TODO: write this:
def test_create_distance_array():
    pass