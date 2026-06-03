import pytest
import random

from gofher.array import create_distance_array

@pytest.mark.parametrize(
    "cx, cy, shape, expected_exception",
    [
        (-1, 10, (100,100), ValueError),    
        (50, -10, (100,100), ValueError)
    ]
)
def test_input_validation_errors(cx, cy, shape, expected_exception):
    with pytest.raises(expected_exception):
        assert create_distance_array(cx, cy, shape)