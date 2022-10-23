import numpy as np
import Patterns as pa
from PIL import Image


class Checkerboard:
    def __init__(self, sub_matrix_size=1, given_image=''):
        """
        This constructor initializes a Checkerboard beginning with a base matrix.
        Parameters
        ----
        :param sub_matrix_size: a matrix (numpy array) from which one the checkerboard will be created
        :param given_image: a local image that has to be converted in a matrix
        Returns
        ----
        :return: None
        """
        self.sub_matrix_size = sub_matrix_size

        if given_image == '':
            # creation of the checkerboard
            if self.sub_matrix_size == 0:
                raise Exception('The sub_matrix size cannot be 0')

            sub_matrix_white = np.ones((self.sub_matrix_size, self.sub_matrix_size), dtype=int)
            sub_matrix_black = np.full((self.sub_matrix_size, self.sub_matrix_size), -1)

            # construction of the first row
            sub_matrix = np.concatenate((sub_matrix_white, sub_matrix_black), axis=1)

            for i in range(0, self.sub_matrix_size - 1):
                sub_matrix = np.concatenate((sub_matrix, np.concatenate((sub_matrix_white, sub_matrix_black), axis=1)),
                                            axis=1)

            # construction of the full checkerboard
            checkerboard = np.concatenate((sub_matrix, np.flip(sub_matrix)), axis=0)
            inter_submatrix = sub_matrix
            for i in range(0, self.sub_matrix_size * 2 - 2):
                checkerboard = np.concatenate((checkerboard, inter_submatrix), axis=0)
                inter_submatrix = np.flip(inter_submatrix)

            self.checkerboard = checkerboard

            checkerboard_new_shape = np.reshape(checkerboard.flatten(order='C'), (1, 2500))

            self.vector = pa.Patterns(patterns=checkerboard_new_shape)

        else:
            image_matrix = image_to_matrix(given_image)
            self.checkerboard = np.reshape(image_matrix, (100, 100))

            image_matrix_new_shape = np.reshape(image_matrix, (1, 10000))

            self.vector = pa.Patterns(patterns=image_matrix_new_shape)


def image_to_matrix(image_file='image.png'):
    """
    This function converts an image to a matrix containing the corresponding pixels.
    If the colour of one pixel is closer to black, the pixel will take the value -1
    and if the colour is closer to white, the pixel will take the value 1.
    Parameters
    ----
    :param image_file: the image which has to be converted in a matrix
    Returns
    ----
    :return: a numpy matrix containing -1 and 1
    """
    im = Image.open(image_file, 'r')
    pixel_values = list(im.getdata())
    final_pixel_values = []

    for mini_list in pixel_values:
        total_indexes = 0
        for index in mini_list:
            total_indexes += index
        if total_indexes > 255:
            final_pixel_values.append(-1)
        else:
            final_pixel_values.append(1)
    return final_pixel_values

