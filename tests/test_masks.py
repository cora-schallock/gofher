"""Test all functions in gofher/mask.py

The script is run using the commend: python -m pytest
"""


import pytest
import numpy as np

from mask import (
    create_ellipse_mask, 
    create_near_major_axis_mask,
    create_near_minor_axis_mask,
    create_bisection_mask
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
        (50, 48, 20, 10, 0.0, (-100, 100), 1.0, ValueError),
        (50, 48, 20, 10, 0.0, (100, 100), False, ValueError),
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
        (np.pi/4,49.5,49.5,0,(100,100,3), ValueError),
        (np.pi/4,49.5,49.5,0,(0,100), ValueError),
        (np.pi/4,49.5,49.5,0,(100,10.25), ValueError)
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
        (np.pi/4,49.5,49.5,0,(100,100,3), ValueError),
        (np.pi/4,49.5,49.5,0,(-7,100), ValueError),
        (np.pi/4,49.5,49.5,0,(100,0.5), ValueError)
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

#TODO: test bisection mask and exceptions

@pytest.mark.parametrize(
    "h, k, theta, shape, expected_exception",
    [
        ("a",49.5,0,(100,100), ValueError),
        (49.5,[],0,(100,100), ValueError),
        ("a",49.5,np.nan,(100,100), ValueError),
        (49.5,49.5,0,(100,100,3), ValueError),
        (49.5,49.5,0,(0,100), ValueError),
        (49.5,49.5,0,(100,0), ValueError)
    ]
)
def test_create_bisection_mask_excpetions(h, k, theta, shape, expected_exception):
    """Tests exceptions expected from create_bisection_mask"""
    with pytest.raises(expected_exception):
        create_bisection_mask(h,k,theta,shape)

def _construct_expected_bisection_pos_mask(h,k,theta,shape):
    """Helper function to construct pos_mask"""
    expected = np.zeros((shape))
    for y in range(shape[0]):
        for x in range(shape[1]):
            origin = np.array([h,k])
            end = np.array([x,y])

            dx, dy = end - origin
            
            #calculate angle in rads. from major axis
            # arctan2(dy, dx) is angle of ray from origin to end and x-axis
            #
            # -theta accounts for major axis offset by theta from x-axis
            #
            # % 2*np.pi fixes range so that all angles positive 
            # in range [0,2*pi] and are measurment from positive x'-axis which is
            # standard x-axis roatetd by theta radians counter clockwise with center
            # through point (h,k). Note -theta means rotating clockwise by theta
            angle = (np.arctan2(dy, dx)- theta) % (2*np.pi)
            expected[y][x] = angle < np.pi

    return expected


def test_create_bisection_mask():
    #TODO: write docstring
    h, k, theta, shape = 49.5, 49.5, 0, (100,100)
    steps = 16

    for i in range(-2*steps,2*steps+1):
        theta = np.pi/steps * i
        expected_pos = _construct_expected_bisection_pos_mask(h,k,theta,shape)
        expected_neg = np.logical_not(expected_pos)

        (pos,neg) = create_bisection_mask(h,k,theta,shape)

        pos_residual = np.sum(np.logical_xor(expected_pos,pos))
        assert pos_residual == 0

        neg_residual = np.sum(np.logical_xor(expected_neg,neg))
        assert neg_residual == 0

        #write that this only works centered
        flipped_residual = np.sum(np.logical_xor(np.logical_not(pos),neg))
        assert flipped_residual == 0

        assert np.sum(pos) == np.sum(neg)
