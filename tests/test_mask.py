import pytest
import numpy as np


from gofher.utils import (
    create
)

def _is_in_ellipse(h, k, a, b, theta, r, x, y):
    x_prime = (x-h)*np.cos(theta) + (y-k)*np.sin(theta)
    y_prime = -(x-h)*np.sin(theta) + (y-k)*np.cos(theta)

    ellipse_equation = (x_prime/a)**2 + (y_prime/b)**2
    return ellipse_equation <= 1


@pytest.mark.parametrize(
    "h, k, a, b, theta, r, shape",
    [
        (50, 48, 20, 10, 0.0, 1.0, (100, 100))
    ]
)
def test_create_angle_array_exceptions(h, k, a, b, theta, r, shape):
    """Test exceptions expected from angle array"""
    expected = np.zeros((shape), bool)

    for x in range(shape[0]):
        for y in range(shape[1]):
            expected[y][x] = _is_in_ellipse(h, k, a, b, theta, r, x, y)

    residual = 


import matplotlib.pyplot as plt

sample = np.zeros((100,100), bool)
for x in range(100):
    for y in range(100):
        sample[y][x] = _is_in_ellipse(50, 50, 40, 10, np.pi/4, 1, x, y)

plt.imshow(sample,origin='lower')
plt.show()