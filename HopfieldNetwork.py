import numpy as np


class HopfieldNetwork:
    def __init__(self, patterns, rule="hebbian"):
        """
        This constructor initializes a HopfieldNetwork with patterns and a rule which is hebbian by default.
        Parameters
        ----
        :param patterns: the patterns for the network, type array
        :param rule: the update rule to use (by default hebbian)
        Returns
        ----
        :return: None
        """
        self.patterns = patterns
        if rule == 'hebbian' or rule == 'storkey' or rule == 'Hebbian' or rule == 'Storkey':
            self.rule = rule
        else:
            raise Exception('The learning rule needs to be either Hebbian or Storkey')

    def reset(self, patterns=np.array([]), rule="hebbian"):
        """
        This function resets the parameters of an object of class HopfieldNetwork to the values provided.
        Parameters
        ----
        :param patterns: np.array, the pattern matrix we want to assign to the patterns argument of the HopfieldNetwork
        object. Default value is an empty array
        :param rule: a string, represents the name of the rule used to compute the HopfieldNetwork.
        Default value is 'hebbian'
        Returns
        ----
        :return: None
        """
        # the rule must be either hebbian or storkey
        if rule != 'hebbian' and rule != 'Hebbian' and rule != 'storkey' and rule != 'Storkey':
            raise Exception('The rule must be either hebbian or storkey')

        self.patterns = patterns
        self.rule = rule

    def hebbian_weights(self):
        """
        This function applies the Hebbian learning rule on some given patterns to create the weight matrix (numpy array).
        Returns
        ----
        :return: returns a 2-dimensional numpy array which is the Hebbian weights matrix
        """
        # The patterns cannot be an empty matrix, otherwise there is a division by 0
        if self.patterns.size == 0:
            raise Exception('The patterns cannot be an empty matrix')

        weights_matrix = 1 / np.shape(self.patterns)[0] * np.dot(self.patterns.T, self.patterns)
        np.fill_diagonal(weights_matrix, 0.)
        return weights_matrix

    def create_h_matrix(self, pattern, old_weights):
        """
        This function creates the matrix h that is used for the Storkey learning rule.
        Parameters
        ----
        :param pattern: 2-dimensional numpy array for which the matrix h should be created
        :param old_weights: the old weight matrix
        Returns
        ----
        :return: matrix h for a specific pattern
        """
        if len(pattern) == 0 or np.shape(old_weights)[0] == 0 or np.shape(old_weights)[1] == 0:
            raise Exception('The patterns or the memorized patterns cannot be empty')

        np.fill_diagonal(old_weights, 0)
        pattern_matrix = np.broadcast_to(pattern[:, None], (len(pattern), len(pattern))).copy()
        np.fill_diagonal(pattern_matrix, 0)
        return np.dot(old_weights, pattern_matrix)

    def create_pre_synaptic_matrix(self, pattern, h_matrix):
        """
        This function creates one term for the pre-synaptic matrix
        Parameters
        ----
        :param pattern: 2-dimensional numpy array for which the pre-synaptic matrix should be created
        :param h_matrix: the matrix h that is used to create the pre-synaptic matrix
        Returns
        ----
        :return: pre-synaptic matrix
        """

        if len(pattern) == 0 or np.shape(h_matrix)[0] == 0 or np.shape(h_matrix)[1] == 0:
            raise Exception('The pattern and the h_matrix should not be empty matrices')

        if len(pattern) != np.shape(h_matrix)[0]:
            raise Exception('The matrix multiplication is not possible due to incompatible matrix sizes')

        pre_synaptic_matrix = np.zeros((len(pattern), len(pattern)))
        for i in range(len(pattern)):
            pre_synaptic_matrix[i] = np.multiply(pattern[i], h_matrix[i, :])
        return pre_synaptic_matrix

    def storkey_weights(self):
        """
        This function applies the Storkey learning rule on some given patterns to create the weight matrix (numpy array).
        Returns
        ----
        :return: returns a 2-dimensional numpy array which is the Storkey weights matrix
        """
        # The patterns cannot be an empty matrix, otherwise there is a division by 0
        if np.shape(self.patterns)[0] == 0 or np.shape(self.patterns)[1] == 0:
            raise Exception('The patterns cannot be an empty matrix')

        number_patterns, size_patterns = self.patterns.shape
        old_weights = np.zeros((size_patterns, size_patterns))
        new_weights = np.zeros((size_patterns, size_patterns))

        for pattern in self.patterns:
            outer_matrix = np.outer(pattern, pattern)
            h_matrix = self.create_h_matrix(pattern, old_weights.copy())
            pre_synaptic_matrix = self.create_pre_synaptic_matrix(pattern, h_matrix.T)
            post_synaptic_matrix = pre_synaptic_matrix.T

            new_weights = old_weights + (1./size_patterns) * (outer_matrix - pre_synaptic_matrix - post_synaptic_matrix)
            old_weights = new_weights.copy()

        return new_weights

    def weights_creation(self):
        """
        Creates the weights matrix of the Hopfield Network
        Returns
        ----
        :return: weights matrix (a numpy array)
        """
        if self.rule == 'hebbian' or self.rule == 'Hebbian':
            weights = self.hebbian_weights()
        elif self.rule == 'storkey' or self.rule == 'Storkey':
            weights = self.storkey_weights()
        else:
            raise Exception('Learning rule has to be hebbian (or Hebbian) or storkey (or Storkey)')

        return weights

    def sigma(self, vector):
        """
        This functions allows to compute a binary vector from a vector containing floats.
        All the vector's elements bigger or equal to 0 become 1 and the elements strictly smaller than 0 become -1.
        Parameters
        ----
        :param vector: a vector (numpy array)
        Returns
        ----
        :return: returns the binary vector corresponding to the initial vector
        """
        condition_vector = (vector >= 0).astype(int)
        condition_vector[condition_vector == 0] = -1
        return condition_vector

    def update(self, state, weight):
        """
        This function applies the update rule to a state pattern.
        It computes the dot product between state and weights. Then it calls the sigma function
        to make a binary vector from the initial one.
        Parameters
        ----
        :param state: a vector (numpy array) which represents the network state
        :param weight: a matrix (numpy array) which represents the weight matrix
        Returns
        ----
        :return: return a vector (numpy array) representing the new state of the network
        after having apply the update rule.
        """
        # the state and the weight cannot be empty
        if len(state) == 0 or weight.size == 0:
            raise Exception('The state or the weights cannot be empty')

        return self.sigma(np.dot(state, weight))

    def update_async(self, state, weight):
        """
        This function applies the asynchronous update rule to a state pattern.
        However, in-stead of computing the full update p(t+1) = Wp(t), it just updates the i-th component
        of the state vector (with i sampled uniformly at random) by computing the new value p[i](t+1) = w[i] · p(t)
        In the previous expression, w[i] denotes the i-th row of the matrix weights.
        So in others words it computes the dot product between a random element of state and weights.Then it calls
        the sigma function to make a binary vector from the initial one.
        Parameters
        ----
        :param state: a vector (numpy array) which represents the network state.
        :param weight: a matrix (numpy array) which represents the weight matrix of the Hopfield network
        Returns
        ----
        :return: return a vector (numpy array) representing the new state of the network after having apply
                 the asynchronous update rule.
        """
        # the state and the weight cannot be empty
        if len(state) == 0 or weight.size == 0:
            raise Exception('The state or the weights cannot be empty')

        i = np.random.randint(0, len(weight))
        # a copy of state is necessary to modify it and return the state with these modifications
        state_copy = state.copy()
        weights_one_line_reshape = np.reshape(weight[i], (1, len(state)))
        state_copy[i] = self.sigma(np.dot(weights_one_line_reshape, state_copy))
        return state_copy

    def dynamics(self, state, saver, max_iter=20):
        """
        This function runs the dynamical system from an initial state until convergence
        or until a maximum number of steps is reached using the update rule.
        Convergence is achieved when two consecutive updates return the same state.
        The name of the dynamic system is stored in the attribute 'rule' of saver.
        The states computes until convergence or the maximal number of iterations is achieved is stored in the
        attribute 'states' of saver.
        Parameters
        ----
        :param state: a vector (numpy array) which represents the network state
        :param max_iter: an integer that represents the maximum number of iterations that could be done
        :param saver: object of class DataSaver in which the history of the states computed until convergence is stored
        Returns
        ----
        :return None
        """
        weight = self.weights_creation()

        if len(state) == 0:
            raise Exception('The state cannot be an empty state')

        if max_iter < 0:
            raise Exception('The maximum number of iterations must be positive')

        history_state = [state]
        old_state = state
        new_state = self.update(old_state, weight)
        history_state.append(new_state)
        counter_iter = 1
        while counter_iter < max_iter and not np.allclose(old_state, new_state):
            old_state = new_state
            new_state = self.update(old_state, weight)
            history_state.append(new_state)
            counter_iter = counter_iter + 1

        saver.rule = self.rule + ' synchronous'
        saver.states = history_state

    def dynamics_async(self, state, saver, max_iter=20000, convergence_num_iter=3000, skip=100):
        """
        This function runs the dynamical system from an initial state until a maximum number of steps is reached
        using the asynchronous update rule.
        With the asynchronous update rule, we can set a softer convergence criterion :
        If the solution does not change for convergence_num_iter steps in a row, then we can say
        that the algorithm has reached convergence.
        Parameters
        ----
        :param state: a vector (numpy array) which represents the network state
        :param saver: an object of class DataSaver in which we store the states computed until convergence is achieved
        :param max_iter: an integer that represents the maximum number of iteration that could be done
        :param convergence_num_iter: an integer that represents the minimal number of iterations during which,
               the solution must not change to reached convergence.
        :param skip: an integer that represents every how many states we store in the history_list
        Returns
        ----
        :return None
        """
        weight = self.weights_creation()

        if len(state) == 0:
            raise Exception('The state cannot be an empty matrix')

        if max_iter < 0 or convergence_num_iter < 0 or skip < 0:
            raise Exception('The maximum iteration number, convergence number and step must be positive')
        history_state = [state]
        old_state = state
        new_state = self.update_async(old_state, weight)
        history_state.append(new_state)
        counter_iter = 1
        convergence_counter = 0

        while counter_iter < max_iter and convergence_counter < convergence_num_iter:
            old_state = new_state
            new_state = self.update_async(old_state, weight)
            history_state.append(new_state)
            counter_iter = counter_iter + 1

            if np.allclose(old_state, new_state):
                convergence_counter += 1
            else:
                convergence_counter = 0

        saver.rule = self.rule + ' asynchronous'
        saver.states = history_state[::skip]

