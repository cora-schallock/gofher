import pytest
import numpy as np


from gofher.mask import create_ellipse_mask, create_near_major_axis_mask

def _is_in_ellipse(h, k, a, b, theta, r, x, y):
    x_prime = (x-h)*np.cos(theta) + (y-k)*np.sin(theta)
    y_prime = -(x-h)*np.sin(theta) + (y-k)*np.cos(theta)

    ellipse_equation = (x_prime/(a*r))**2 + (y_prime/(b*r))**2
    return ellipse_equation <= 1

@pytest.mark.parametrize(
    "h, k, a, b, theta, r, shape",
    [
        (50, 48, 20, 10, 0.0, 1.0, (100, 100)),
        (50, 48, 20, 10, np.pi * 0.25, 1.0, (100, 100)),
        (50, 48, 20, 10, np.pi * 0.5, 1.0, (100, 100))
    ]
)
def test_create_ellipse_mask(h, k, a, b, theta, r, shape):
    """Test exceptions expected from angle array"""
    expected = np.zeros((shape), bool)

    for x in range(shape[0]):
        for y in range(shape[1]):
            expected[y][x] = _is_in_ellipse(h, k, a, b, theta, r, x, y)

    ellipse_mask = create_ellipse_mask(h, k, a, b, theta, shape, r)
    residual = np.abs(np.bitwise_xor(ellipse_mask, expected))
    assert np.sum(residual) == 0

@pytest.mark.parametrize(
    "sweep, cx, cy, theta, shape",
    [
        (np.pi/4,49.5,49.5,0,(100,100))
    ]
)
def test_create_near_major_axis_mask(sweep, cx, cy, theta, shape):
    #todo: fix
    pass

@pytest.mark.parametrize(
    "sweep, cx, cy, theta, shape",
    [
        (np.pi/4, 50, 48, 0.0, (100, 100))
    ]
)
def test_create_near_minor_axis_mask(sweep, cx, cy, theta, shape):
    #todo: fix
    pass