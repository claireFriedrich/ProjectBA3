import Checkerboard as cb
import pytest
import numpy as np


def test_init_checkerboard():
    # ensure that the size of the sub_matrix is not equal to 0
    with pytest.raises(Exception):
        checkerboard = cb.Checkerboard(0)
    
    # ensure that at the very end of the function the matrix checkerboard is a square matrix
    sub_matrix_size = 5
    checkerboard = cb.Checkerboard(sub_matrix_size)
    x, y = np.shape(checkerboard.checkerboard)
    assert x == y


if __name__ == '__main__':
    pytest.main()
