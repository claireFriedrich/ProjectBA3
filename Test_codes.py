import numpy as np
import timeit
import math


'''def distance(p1, p2):
    """ Computes the distance between two points.

    :param p1: a 2D vector, represents the x and y coordinates of the fist point
    :param p2: a 2D vector, represents the x and y coordinates of the second point
    :return: an integer representing the distance between the two points (p1 and p2)

    Examples
    -------
    >>> distance((1,2), (3,3))
    2.23606797749979
    >>> distance((0,0), (-7,0))
    7.0
    >>> distance((0,0), (0,0))
    0.0
    """
    distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return distance


if __name__ == '__main__':
    import doctest
    doctest.testmod()'''


def method_frobenius_norm(matrix):
    sum_terms = 0
    for i in range(0, np.shape(matrix)[0]):
        for j in range(0, np.shape(matrix)[1]):
            sum_terms += pow(abs(matrix[i][j]), 2)

    return pow(sum_terms, 1/2)


def method_frobenius_norm_time():
    setup_code = '''
import numpy as np
from __main__ import method_frobenius_norm
'''

    test_code = '''
matrix = matrix = np.random.rand(28, 28)
method_frobenius_norm(matrix)'''

    times = timeit.repeat(setup=setup_code,
                          stmt=test_code,
                          repeat=3,
                          number=10)
    print('Pure python method time: {}'.format(min(times)))


def numpy_frobenius_norm_time():
    setup_code = '''
import numpy as np
'''

    test_code = '''
matrix = matrix = np.random.rand(28, 28)
np.linalg.norm(matrix)'''

    # timeit.repeat statement
    times = timeit.repeat(setup=setup_code,
                          stmt=test_code,
                          repeat=3,
                          number=10)
    print('Numpy function time: {}'.format(min(times)))


if __name__ == '__main__':
    method_frobenius_norm_time()
    numpy_frobenius_norm_time()

