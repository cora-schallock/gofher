"""Masks used by GOFHER

This module provides a collection of 2D binary masks 
that can be used on the data.
"""

from arrays import create_meshgrid

import numpy as np

def create_ellipse_mask(cx: float, cy: float, 
                        a: float, b: float, 
                        theta: float, shape: tuple,
                        r: float = 1.0) -> np.ndarray:
    """Create a binary ellipse mask"""

    #TODO: test params
    xx, yy = create_meshgrid(shape)

    xx_offset = xx-cx
    yy_offset = yy-cy
    x_prime = xx_offset*np.cos(theta) + yy_offset*np.sin(theta)
    y_prime = -xx_offset*np.sin(theta) + yy_offset*np.cos(theta)

    ellipse_equation = (x_prime/(a*r))**2 + (y_prime/(b*r))**2
    return ellipse_equation <= 1

def create_near_major_axis_mask(sweep: float, cx: float, cy: float, 
                                theta: float, shape: tuple) -> np.ndarray:
    pass

def create_near_minor_axis_mask(sweep: float, cx: float, cy: float, 
                                theta: float, shape: tuple) -> np.ndarray:
    pass


array = create_ellipse_mask(300, 400, 150, 100, 0, (1000,1000), 1)
import matplotlib.pyplot as plt

plt.imshow(array,origin='lower')
plt.show()