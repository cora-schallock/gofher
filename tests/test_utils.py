import pytest
import numpy as np

from gofher.utils import is_float, is_2d_array_shape

@pytest.mark.parametrize("value, expected", [
    (1, True),
    (2.5, True),
    ([], False),
    ("a", False),
    ({"key":"value"}, False),
    (np.float64(64), True),
    (np.int32(32.3), True)
])
def test_is_float(value, expected):
    assert is_float(value) == expected

@pytest.mark.parametrize("value, expected", [
    ((0, 0), False),
    ((1, 1), True),
    ((100, 100), True),
    ((-100, 100), False),
    ((100, 100, 50), False)
])
def test_is_2d_array_shape(value, expected):
    assert is_2d_array_shape(value) == expected