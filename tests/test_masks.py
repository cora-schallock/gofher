"""Test all functions in gofher/mask.py

The script is run using the commend: python -m pytest
"""


import pytest
import numpy as np

from mask import (
    create_ellipse_mask, 
    create_near_major_axis_mask,
    create_near_minor_axis_mask
)

def _is_in_ellipse(h, k, a, b, theta, r, x, y):
    x_prime = (x-h)*np.cos(theta) + (y-k)*np.sin(theta)
    y_prime = -(x-h)*np.sin(theta) + (y-k)*np.cos(theta)

    ellipse_equation = (x_prime/(a*r))**2 + (y_prime/(b*r))**2
    return ellipse_equation <= 1

@pytest.mark.parametrize(
    "h, k, a, b, theta, r, shape, expected_exception",
    [
        ("a", 48, 20, 10, 0.0, (100, 100), 1.0, ValueError),
        (50, [48], 20, 10, 0.0, (100, 100), 1.0, ValueError),
        (50, 48, {}, 10, 0.0, (100, 100), 1.0, ValueError),
        (50, 48, -20, 10, 0.0, (100, 100), 1.0, ValueError),
        (50, 48, 20, {}, 0.0, (100, 100), 1.0, ValueError),
        (50, 48, 20, -10, 0.0, (100, 100), 1.0, ValueError),
        (50, 48, 20, 10, (), (100, 100), 1.0, ValueError),
        (50, 48, 20, 10, 0.0, (100), 1.0, ValueError),
        (50, 48, 20, 10, 0.0, (-100, 100), False, ValueError),
        (50, 48, 20, 10, 0.0, (100, 100), -1, ValueError)
    ]
)
def test_create_ellipse_mask_exceptions(h, k, a, b, theta, 
                                        shape, r, expected_exception):
    """Tests exceptions expected from distance array"""
    with pytest.raises(expected_exception):
        create_ellipse_mask(h, k, a, b, theta, shape, r)

@pytest.mark.parametrize(
    "h, k, a, b, theta, shape, r",
    [
        (50, 48, 20, 10, 0.0, (100, 100), 1.0),
        (50, 48, 20, 10, np.pi * 0.25, (100, 100), 1.0),
        (50, 48, 20, 10, np.pi * 0.5, (100, 100),  1.0)
    ]
)
def test_create_ellipse_mask(h, k, a, b, theta, shape, r):
    """Test ellipse mask creation
    
    Tolerance:
        residual == 0
    """
    expected = np.zeros((shape), bool)

    for y in range(shape[0]):
        for x in range(shape[1]):
            expected[y][x] = _is_in_ellipse(h, k, a, b, theta, r, x, y)

    ellipse_mask = create_ellipse_mask(h, k, a, b, theta, shape, r)
    residual = np.abs(np.bitwise_xor(ellipse_mask, expected))
    assert np.sum(residual) == 0

@pytest.mark.parametrize(
    "sweep, h, k, theta, shape, expected_exception",
    [
        ([10],49.5,49.5,0,(100,100), ValueError),
        (-np.pi,49.5,49.5,0,(100,100), ValueError),
        (np.pi/4,"a",49.5,0,(100,100), ValueError),
        (np.pi/4,49.5,[],0,(100,100), ValueError),
        (np.pi/4,49.5,49.5,{},(100,100), ValueError),
        (np.pi/4,49.5,49.5,0,(100,100,3), ValueError)
    ]
)
def test_create_near_major_axis_mask_excpetions(sweep, h, k, theta, 
                                                shape, expected_exception):
    """Tests exceptions expected from major axis array"""
    with pytest.raises(expected_exception):
        create_near_major_axis_mask(sweep,h,k,theta,shape)

@pytest.mark.parametrize(
    "sweep, h, k, theta, shape",
    [
        (np.pi/4,49.5,49.5,0,(100,100)),
        (np.pi/4,49.5,49.5,-np.pi/4,(100,100)),
        (np.pi/6,24.5,24.5,-np.pi/4,(50,50))
    ]
)
def test_create_near_major_axis_mask(sweep, h, k, theta, shape):
    """Test ellipse mask creation
    
    Tolerance:
        residual == 0
    """
    maj_axis_mask = create_near_major_axis_mask(sweep,h,k,theta,shape)
    expected_mask = np.zeros((shape))

    for y in range(shape[0]):
        for x in range(shape[1]):
            origin = np.array([h,k])
            end = np.array([x,y])

            dx, dy = end - origin
            
            #calculate angle in rads. from major axis
            # arctan2(dy, dx) is angle of ray from origin to end and x-axis
            # -theta accounts for major axis offset by theta from x-axis
            # % np.pi fixes range so that all angles positive inj range [0,pi]
            #     from minor axis
            angle_from_major = (np.arctan2(dy, dx) - theta)% (np.pi)
            expected_mask[y][x] = angle_from_major <= sweep or angle_from_major >= np.pi-sweep
    
    residual = expected_mask-maj_axis_mask
    assert np.sum(residual) == 0

@pytest.mark.parametrize(
    "sweep, h, k, theta, shape, expected_exception",
    [
        ([10],49.5,49.5,0,(100,100), ValueError),
        (-np.pi,49.5,49.5,0,(100,100), ValueError),
        (np.pi/4,"a",49.5,0,(100,100), ValueError),
        (np.pi/4,49.5,[],0,(100,100), ValueError),
        (np.pi/4,49.5,49.5,{},(100,100), ValueError),
        (np.pi/4,49.5,49.5,0,(100,100,3), ValueError)
    ]
)
def test_create_near_minor_axis_mask_excpetions(sweep, h, k, theta, 
                                                shape, expected_exception):
    """Tests exceptions expected from minor axis array"""
    with pytest.raises(expected_exception):
        create_near_minor_axis_mask(sweep,h,k,theta,shape)

@pytest.mark.parametrize(
    "sweep, h, k, theta, shape",
    [
        (np.pi/4,49.5,49.5,0,(100,100)),
        (np.pi/4,49.5,49.5,-np.pi/4,(100,100)),
        (np.pi/6,24.5,24.5,-np.pi/4,(50,50))
    ]
)
def test_create_near_minor_axis_mask(sweep, h, k, theta, shape):
    """Test ellipse mask creation
    
    Tolerance:
        residual == 0
    """
    maj_axis_mask = create_near_minor_axis_mask(sweep,h,k,theta,shape)
    expected_mask = np.zeros((shape))

    for y in range(shape[0]):
        for x in range(shape[1]):
            origin = np.array([h,k])
            end = np.array([x,y])

            dx, dy = end - origin
            
            #calculate angle in rads. from minor
            # arctan2(dy, dx) is angle of ray from origin to end and x-axis
            # +theta accounts for major axis offset by theta from x-axis
            #   note since we care about minor axis, it is + not -
            # % np.pi fixes range so that all angles positive inj range [0,pi]
            #     from minor axis
            angle_from_minor = (np.arctan2(dy, dx) + theta) % np.pi
            expected_mask[y][x] = angle_from_minor <= sweep or angle_from_minor >= np.pi-sweep
    
    residual = expected_mask-maj_axis_mask
    assert np.sum(residual) == 0
