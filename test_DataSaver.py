import pytest
import numpy as np
import DataSaver as ds
import Patterns as pa
import HopfieldNetwork as hn


def test_reset_datasaver():
    datasaver = ds.DataSaver()
    datasaver.reset()
    assert datasaver.rule == ''
    assert np.allclose(datasaver.states, np.array([]))
    assert np.allclose(datasaver.energies, np.array([]))


def test_compute_energy(benchmark):
    # benchmark in order to measure duration of the function energy for one pattern
    # weights_for_benchmark is created by calling the function hebbian_weights
    pattern_for_benchmark = pa.Patterns(50, 2500)
    hopfield_network = hn.HopfieldNetwork(pattern_for_benchmark.patterns)
    weights_for_benchmark = hopfield_network.weights_creation()
    data_saver = ds.DataSaver()
    benchmark.pedantic(data_saver.compute_energy, args=(pattern_for_benchmark.patterns[0], weights_for_benchmark), iterations=5)

    # ensure that the state or the weights are not empty
    with pytest.raises(Exception):
        data_saver.compute_energy(np.array([]), np.ones((1, 2)))

    with pytest.raises(Exception):
        data_saver.compute_energy(np.array((1, 0)), np.ones((1, 2)))

    with pytest.raises(Exception):
        data_saver.compute_energy(np.array((0, 1)), np.ones((1, 2)))

    with pytest.raises(Exception):
        data_saver.compute_energy(np.ones((1, 2)), np.array([]))

    with pytest.raises(Exception):
        data_saver.compute_energy(np.ones((1, 2)), np.array((1, 0)))

    with pytest.raises(Exception):
        data_saver.compute_energy(np.ones((1, 2)), np.array((0, 1)))

    # ensure that the length of the state and the length of axis 0 of weights are equal
    state_created_0 = np.array([1, 2, 3])
    weights_created_0 = np.ones((1, 3))
    with pytest.raises(Exception):
        data_saver.compute_energy(state_created_0, weights_created_0)


def test_compute_energies():
    # ensure that the weights matrix is not empty
    data_saver = ds.DataSaver()
    with pytest.raises(Exception):
        data_saver.compute_energies(np.array([]))

    # ensure that the length of axis 1 of states and the length of axis-0 of weights are equal
    data_saver.states = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    with pytest.raises(Exception):
        data_saver.compute_energies(np.array([[1, 2, 3], [1, 2, 3]]))

    # test if the length of the returned state_history is right
    data_saver.reset()
    data_saver.states = np.ones((4, 8))
    weights_created = np.ones((8, 8))
    data_saver.compute_energies(weights_created, 1)

    assert len(data_saver.energies) == 4

    # ensure that the step given is bigger than 0
    data_saver.reset()
    states_created = np.ones((4, 8))
    data_saver.states = states_created
    with pytest.raises(Exception):
        data_saver.compute_energies(np.array([1, 2, 3]), -1)

    with pytest.raises(Exception):
        ds.DataSaver.compute_energies(np.array([1, 2, 3]), 0)

    # if the length of one state is divisible by the step,
    # than the length of the returned list is length(state)/ step
    data_saver.reset()
    data_saver.states = states_created

    data_saver.compute_energies(weights_created, 2)
    assert len(data_saver.energies) == np.shape(states_created)[0] / 2

    data_saver.energies = []
    data_saver.compute_energies(weights_created, 4)
    assert len(data_saver.energies) == np.shape(states_created)[0] / 4

    # if the length of one state is not divisible by the step,
    # than the length of the returned list is length(state)//step +1
    data_saver.energies = []
    data_saver.compute_energies(weights_created, 3)
    assert len(data_saver.energies) == (np.shape(states_created)[0] // 3 + 1)

    data_saver.energies = []
    data_saver.compute_energies(weights_created, 4)
    assert len(data_saver.energies) == (np.shape(states_created)[0] // 5 + 1)


def test_save_video():
    saver = ds.DataSaver()
    saver.states = np.array([])
    # ensure that state_list is not empty
    with pytest.raises(Exception):
        saver.save_video()


if __name__ == '__main__':
    pytest.main()
