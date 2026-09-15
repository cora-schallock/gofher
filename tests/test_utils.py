"""Test all functions in gofher/utils.py

The script is run using the commend: python -m pytest
"""


import pytest
import numpy as np

from gofher.utils import (
    is_int,
    is_float_int, 
    is_2d_array_shape, 
    is_float_int_array, 
    is_2d_bool_array, 
    is_2d_same_shape_arrays, 
    is_2d_array, 
    is_finite_array,
    generate_band_pair_tuples,
    generate_all_band_pair_strings
)

@pytest.mark.parametrize("value, expected", [
    (1, True),
    (2.5, False),
    ([], False),
    ("a", False),
    ({"key":"value"}, False),
    (np.float64(64), False),
    (np.int32(32), True),
    (np.ones((2,2),int), False)
])
def test_is_int(value, expected):
    """Test exceptions expected from is_float_int"""
    assert is_int(value) == expected

@pytest.mark.parametrize("value, expected", [
    (1, True),
    (2.5, True),
    ([], False),
    ("a", False),
    ({"key":"value"}, False),
    (np.float64(64), True),
    (np.int32(32.3), True),
    (np.ones((2,2),float), False)
])
def test_is_float_int(value, expected):
    """Test exceptions expected from is_float_int"""
    assert is_float_int(value) == expected

@pytest.mark.parametrize("value, expected", [
    ((0, 0), False),
    ((1, 1), True),
    ((100, 100), True),
    ((-100, 100), False),
    ((100, 100, 50), False)
])
def test_is_2d_array_shape(value, expected):
    """Test exceptions expected from is_2d_array_shape"""
    assert is_2d_array_shape(value) == expected

@pytest.mark.parametrize("value, expected", [
    (np.ones((2,1)), True),
    (np.ones((2,)), False),
    ([False], False),
    (0, False)
])
def test_is_2d_array(value, expected):
    """Test exceptions expected from is_2d_array"""
    assert is_2d_array(value) == expected

@pytest.mark.parametrize("value, expected", [
    (np.array([[1.0,2.0],[3.0,4.0]]), True),
    (np.array([[1,2],[3,4]]), True),
    (np.array([1,2]), True),
    (np.float32(3), False),
    (np.int32(2.0), False),
    ([["1",1],["",False]], False),
    ("a", False)
])
def test_is_float_int_array(value, expected):
    """Test exceptions expected from is_float_int_array"""
    assert is_float_int_array(value) == expected

@pytest.mark.parametrize("value, expected", [
    (np.array([[True,False],[True,True]]), True),
    (np.array([True,True]), False),
    (True, False),
    (1, False),
    ([["1",1],["",False]], False)
])
def test_is_2d_bool_array(value, expected):
    """Test exceptions expected from is_2d_bool_array"""
    assert is_2d_bool_array(value) == expected

@pytest.mark.parametrize("value1, value2, expected", [
    (np.zeros((2,2)),np.ones((2,2)), True),
    (np.zeros((2,1)),np.ones((2,1)), True),
    (np.zeros((2,)),np.ones((2,)), False),
    ([[0,0]],np.ones((2,)), False)
])
def test_is_2d_same_shape_arrays(value1, value2, expected):
    """Test exceptions expected from is_2d_same_shape_arrays"""
    assert is_2d_same_shape_arrays(value1, value2) == expected

@pytest.mark.parametrize("value, expected", [
    (np.array([[0.0,-125.6],[100,50]]), True),
    (np.array([np.nan,0]), False),
    (np.array([-np.inf,14]), False),
    (np.array([np.inf,14]), False),
    ([["1",1],["",False]], False)
])
def test_is_finite_array(value, expected):
    """Test exceptions expected from def test_is_finite_array"""
    assert is_finite_array(value) == expected

@pytest.mark.parametrize("bands, expected_expectation", [
    (1.0,TypeError),
    ([1.0,{}], TypeError),
    (["g"], ValueError)
])
def generate_band_pair_tuples_and_strings_exceptions(bands, expected_expectation):
    """Test the excpetions expected for:
    generate_band_pair_tuples() and generate_band_pair_strings() 

    Programmer Note: The two functions have same behavior for exceptions
        hence the combined test.
    """

    # Validate excpetion is raised for generate_band_pair_tuples()
    with pytest.raises(expected_expectation):
        generate_band_pair_tuples(bands)

    # Validate excpetion is raised for generate_all_band_pair_strings()
    with pytest.raises(expected_expectation):
        generate_all_band_pair_strings(bands)

@pytest.mark.parametrize("bands, expected", [
    (["g","r"],
     [("g","r")]),
    (["g","r","i"],
     [("g","r"),("g","i"),("r","i")]),
    (["g","r","i","z"],
     [("g","r"),("g","i"),("g","z"),("r","i"),("r","z"),("i","z")]),
])
def test_generate_band_pair_tuples(bands, expected):
    """Test generate_band_pair_tuples()
    
    This function should return all ordered tuples are sorted
    in lexigraphical order according to original list
    """
    
    assert generate_band_pair_tuples(bands) == expected

@pytest.mark.parametrize("bands, expected", [
    (["g","r"],
     ["g-r"]),
    (["g","r","i"],
     ["g-r","g-i","r-i"]),
    (["g","r","i","z"],
     ["g-r","g-i","g-z","r-i","r-z","i-z"]),
])
def test_generate_band_pair_strings(bands, expected):
    """Test generate_band_pair_strings()
    
    This function should return all ordered strings are sorted
    in lexigraphical order according to original list
    """

    assert generate_all_band_pair_strings(bands) == expected