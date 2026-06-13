import pytest
import numpy as np

from gofher.utils import is_float_int, is_2d_array_shape, is_2d_float_int_array, is_2d_bool_array, is_2d_same_shape_arrays, is_2d_array, is_finite_array

@pytest.mark.parametrize("value, expected", [
    (1, True),
    (2.5, True),
    ([], False),
    ("a", False),
    ({"key":"value"}, False),
    (np.float64(64), True),
    (np.int32(32.3), True)
])
def test_is_float_int(value, expected):
    assert is_float_int(value) == expected

@pytest.mark.parametrize("value, expected", [
    ((0, 0), False),
    ((1, 1), True),
    ((100, 100), True),
    ((-100, 100), False),
    ((100, 100, 50), False)
])
def test_is_2d_array_shape(value, expected):
    assert is_2d_array_shape(value) == expected

@pytest.mark.parametrize("value, expected", [
    (np.ones((2,1)), True),
    (np.ones((2,)), False),
    ([False], False),
    (0, False)
])
def test_is_2d_array(value, expected):
    assert is_2d_array(value) == expected

@pytest.mark.parametrize("value, expected", [
    (np.array([[1.0,2.0],[3.0,4.0]]), True),
    (np.array([[1,2],[3,4]]), True),
    (np.array([1,2]), False),
    (np.float32(3), False),
    (np.int32(2.0), False),
    ([["1",1],["",False]], False),
    ("a", False)
])
def test_is_2d_float_int_array(value, expected):
    assert is_2d_float_int_array(value) == expected

@pytest.mark.parametrize("value, expected", [
    (np.array([[True,False],[True,True]]), True),
    (np.array([True,True]), False),
    (True, False),
    (1, False),
    ([["1",1],["",False]], False)
])
def test_is_2d_bool_array(value, expected):
    assert is_2d_bool_array(value) == expected

@pytest.mark.parametrize("value1, value2, expected", [
    (np.zeros((2,2)),np.ones((2,2)), True),
    (np.zeros((2,1)),np.ones((2,1)), True),
    (np.zeros((2,)),np.ones((2,)), False),
    ([[0,0]],np.ones((2,)), False)
])
def test_is_2d_same_shape_arrays(value1, value2, expected):
    assert is_2d_same_shape_arrays(value1, value2) == expected

@pytest.mark.parametrize("value, expected", [
    (np.array([[0.0,-125.6],[100,50]]), True),
    (np.array([np.nan,0]), False),
    (np.array([-np.inf,14]), False),
    (np.array([np.inf,14]), False),
    ([["1",1],["",False]], False)
])
def test_is_finite_array(value, expected):
    assert is_finite_array(value) == expected

