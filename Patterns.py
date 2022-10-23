import numpy as np
import random


class Patterns:
    def __init__(self, num_patterns=1, size_pattern=2500, patterns=None):
        """
        This constructor initializes an object of type Patterns with a fixed shape and if wanted with already constructed patterns
        Parameters
        ----
        :param num_patterns: the number of lines in Patterns
        :param size_pattern: the number of rows in Patterns
        :param patterns: a matrix (numpy array) to initialize directly Patterns
        Returns
        ----
        :return: None
        """
        if patterns is None:
            if num_patterns <= 0 or size_pattern <= 0:
                raise Exception('The values for creating a matrix have to be positive')
            self.patterns = np.random.choice(np.array([-1, 1]), (num_patterns, size_pattern))
        else:
            # allow the generation of a instance Patterns from a vector
            for element in patterns:
                if element.all() != -1 and element.all() != 1:
                    raise Exception('The pattern can only contain 1 or -1')

            patterns_copy = patterns.copy()
            if len(patterns) == 1:
                patterns_copy.reshape((1, -1))
            self.patterns = patterns_copy

    def reset(self):
        """
        This function resets the entire object of class Patterns by emptying its argument, so putting it to an
        empty array.
        Returns
        ----
        :return: None
        """
        self.patterns = np.array([])

    def perturb_pattern(self, num_perturb):
        """
        This function perturbs a given pattern.
        It samples num_perturb elements of the input pattern uniformly at random and changes their sign.
        Each element of the pattern is at most changed once for a given num_perturb.
        Also if num_perturb is bigger than the length of pattern, then the function perturbs each element of the pattern,
        it means that num_perturb becomes the number of elements in pattern.
        Parameters
        ----
        :param num_perturb: integer which is the number of perturbation(s) wanted in the pattern
        Returns
        ----
        :return: returns the perturbed pattern
        """
        pattern = self.patterns[np.random.randint(0, np.shape(self.patterns)[0])]
        pattern_copy = pattern.copy()

        # the pattern cannot be empty
        if pattern.size == 0:
            raise Exception('The pattern cannot be an empty vector')

        # the number of perturbations need to be positive or equal to 0
        if num_perturb < 0:
            raise Exception('The number of perturbations need to be >=0')

        # ensure that the number of perturbations is an integer and not a float
        if int(num_perturb) != num_perturb:
            raise Exception('The number of perturbations need to be an integer')

        # the pattern can contain only 1 or -1
        for element in pattern_copy:
            if element != 1 and element != -1:
                raise Exception('The pattern should contain only 1 or -1')

        if num_perturb > len(pattern_copy):
            num_perturb = len(pattern_copy)

        history_n = []
        for i in range(num_perturb):
            n = random.randint(0, len(pattern_copy) - 1)

            # to avoid that a certain position of a pattern can change more than once for a total number of perturbation
            while n in history_n:
                n = random.randint(0, len(pattern_copy) - 1)

            history_n.append(n)

            if pattern_copy[n] == -1:
                pattern_copy[n] = 1
            elif pattern_copy[n] == 1:
                pattern_copy[n] = -1

        return pattern_copy

    def replace_one_random_state(self, state):
        """
        This function replaced one random state in the matrix of patterns with the state provided in the arguments
        Parameters
        ----
        :param state: np.array, the state with which we replace one random pattern in the matrix of multiple patterns
        Returns
        ----
        :return: None
        """
        # ensure that state is not empty
        if state.size == 0:
            raise Exception('State cannot be empty')

        i = np.random.randint(0, np.shape(self.patterns)[0])
        self.patterns[i] = state

    def pattern_match(self, pattern):
        """
        Match a pattern with the corresponding memorized one.
        In others words, this function checks if pattern is in the memorized_patterns matrix.
        Parameters
        ----
        :param pattern: a vector (numpy array) which represents the pattern that we want to know if it is in the self pattern
        Returns
        ----
        :return: returns None if no memorized pattern (in memorized_patterns) matches,
                 otherwise it returns the index of the row corresponding to the matching pattern (in self)
        """
        # the patterns or the memorized patterns cannot be empty
        if len(self.patterns) == 0 or np.shape(self.patterns)[0] == 0 or np.shape(self.patterns)[1] == 0:
            raise Exception('The patterns or the memorized patterns cannot be empty')

        # the length of pattern should be the same as the length of the axis 1 in memorized patterns
        if np.shape(self.patterns)[1] != np.shape(self.patterns)[1]:
            raise Exception('The length of axis 1 in pattern and the length of the axis 1 in memorized patterns should be the same')

        for i in range(0, np.shape(self.patterns)[0]):
            if np.allclose(pattern, self.patterns[i]):
                return i

