import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class DataSaver:
    def __init__(self):
        """
        This constructor initializes a DataSaver.
        Returns
        ----
        :return: None
        """
        self.rule = ''
        self.states = []
        self.energies = []

    def reset(self):
        """
        This functions resets an object of type DataSaver ny emptying all its arguments
        Returns
        ----
        :return: None
        """
        self.rule = ''
        self.states = []
        self.energies = []

    def compute_energy(self, state, weights):
        """
        This function computes the energy of a state.
        This calculation is called energy because the patterns which are memorized, either with
        the Hebbian or with the Storkey rule, are local minima of this function.
        Furthermore, the energy is a non-increasing quantity of the dynamical system, meaning that the energy
        at time step t is always greater or equal than the energy at any subsequent step t′ > t.
        Parameters
        ----
        :param state: a vector (numpy array) which represents the network state.
        :param weights: a 2-dimensional numpy array corresponding to the weights matrix of the memorized patterns
        Returns
        ----
        :return None
        """
        # the weights or the state cannot be empty
        if len(state) == 0 or np.shape(weights)[0] == 0 or np.shape(weights)[1] == 0:
            raise Exception('The weights or the state cannot be empty')

        # the length of the state and the length of axis 0 of weights must be equal
        if len(state) != np.shape(weights)[0]:
            raise Exception('The length of the state and the length of axis 0 of weights must be equal')

        trace_matrix = np.trace(np.dot(np.outer(state, state.T), weights))

        self.energies.append((-1 / 2) * trace_matrix)

    def compute_energies(self, weights, step=1):
        """
        This function computes the energy of each state contained in the states (matrix)
        Parameters
        ----
        :param weights: a 2-dimensional numpy array corresponding to the weights matrix of the memorized patterns
        :param step: an integer which represents every how many states we compute the energy (default value is 1,
               i.e we compute the energy for every state)
        Returns
        ----
        :return None
        """
        # the weights or the states cannot be empty
        if len(self.states) == 0 or np.shape(weights)[0] == 0 or np.shape(weights)[1] == 0:
            raise Exception('The weights or the state cannot be empty')

        # the length of axis-1 of states and the length of axis 0 of weights must be equal
        if np.shape(self.states)[1] != np.shape(weights)[0]:
            raise Exception('The length of the state and the length of axis 0 of weights must be equal')

        # test if the step is strictly positive
        if step <= 0:
            raise Exception('The step must be greater than 0')

        state_counter = 0
        while state_counter < np.shape(self.states)[0]:
            self.compute_energy(self.states[state_counter], weights)
            state_counter += step

    def get_data(self):
        """
        This function returns the object.
        Returns
        ----
        :return: the object itself (self)
        """
        return self

    def save_video(self, img_shape=(50, 50), title=''):
        """
        This function saves a video of the convergence process of a pattern on the laptop.
        Parameters
        ----
        :param img_shape: the size of the images forming the video
        :param title: a string, corresponds to the title of the video
        Returns
        ----
        :return: NONE
        """
        if len(self.states) == 0:
            raise Exception('The state_list cannot be empty')

        out_path = "video_" + title + self.rule + ".gif"

        for i in range(0, len(self.states)):
            pattern = self.states[i].copy()
            pattern = np.reshape(pattern, img_shape)
            self.states[i] = pattern

        list_frames = []
        fig = plt.figure(figsize=img_shape)
        plt.title(self.rule, fontsize=150)
        for state in self.states:
            list_frames.append([plt.imshow(state, cmap=plt.cm.get_cmap('Greys'), animated=True)])

        video = animation.ArtistAnimation(fig, list_frames)
        writer_video = animation.PillowWriter(fps=3)

        video.save(out_path, writer=writer_video)

    def plot_energy(self):
        """
        This function allows to plot the different values of the energies of the states necessary until convegence.
        It also stores the plot on your computer.
        Returns
        ----
        :return: None
        """
        x = self.energies
        fig = plt.figure()
        plt.plot(x, 'm')
        plt.title('Time-energy plot ' + self.rule)
        plt.xlabel('Time')
        plt.ylabel('Energy')

        # save the plot on the computer
        out_path = "plot_" + self.rule + ".png"
        fig.savefig(out_path)

