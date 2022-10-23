import pytest
import numpy as np
import HopfieldNetwork as hn
import Patterns as pa
import DataSaver as ds


def test_init_hopfieldnetwork():
    # ensure that the rule is either hebbian or storkey
    with pytest.raises(Exception):
        hn.HopfieldNetwork(np.array([1, 1]), 'Non')

    # test the object has the expected values
    hopfieldnetwork = hn.HopfieldNetwork(np.array([1, -1]))
    assert hopfieldnetwork.rule == 'hebbian'
    assert np.allclose(hopfieldnetwork.patterns, np.array([1, -1]))

    hopfieldnetwork = hn.HopfieldNetwork(np.ones((1, 5)), 'storkey')
    assert hopfieldnetwork.rule == 'storkey'
    assert np.allclose(hopfieldnetwork.patterns, np.ones((1, 5)))


def test_reset_hopfieldnetwork():
    # ensure that the rule is either hebbian or storkey
    hopfieldnetwork = hn.HopfieldNetwork(np.array([1, -1]), 'hebbian')
    with pytest.raises(Exception):
        hopfieldnetwork.reset(rule='None')

    # test if the function sets the object with the right values
    hopfieldnetwork.reset(rule='storkey')
    assert hopfieldnetwork.rule == 'storkey'

    hopfieldnetwork.reset(patterns=np.array([1, 1]))
    assert np.allclose(hopfieldnetwork.patterns, np.array([1, 1]))

    hopfieldnetwork.reset(np.array([-1, -1]), 'hebbian')
    assert hopfieldnetwork.rule == 'hebbian'
    assert np.allclose(hopfieldnetwork.patterns, np.array([-1, -1]))


def test_hebbian_weights(benchmark):
    # benchmark in order to measure generation time of function hebbian_weights
    patterns_for_benchmark = pa.Patterns(50, 2500)
    benchmark.pedantic(hn.HopfieldNetwork.hebbian_weights, args=(patterns_for_benchmark,), iterations=5)

    # ensure that the patterns is not an empty matrix
    with pytest.raises(Exception):
        hn.HopfieldNetwork.hebbian_weights(np.array([]))

    with pytest.raises(Exception):
        hn.HopfieldNetwork.hebbian_weights(np.array((1, 0)))

    with pytest.raises(Exception):
        hn.HopfieldNetwork.hebbian_weights(np.array((0, 1)))

    # test if the hebbian matrix has the correct size
    weight_test = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1], [1, 1, 1, 1]])
    hopfielnetwork = hn.HopfieldNetwork(pa.Patterns(4, 4).patterns, 'hebbian')
    hebbian_matrix_created = hopfielnetwork.hebbian_weights()
    x1, y1 = np.shape(weight_test)
    x2, y2 = np.shape(hebbian_matrix_created)
    assert y1 == x2
    assert y1 == y2

    # test if the values of the hebbian matrix are in the correct range
    assert np.where(np.logical_and(hebbian_matrix_created >= -1, hebbian_matrix_created <= 1))

    # test if the hebbian matrix is symmetric
    assert np.allclose(hebbian_matrix_created.T, hebbian_matrix_created)

    # test that all the elements on the diagonal are 0
    diagonal_values_created = np.diag(hebbian_matrix_created)
    diagonal_values_optimal = np.zeros((1, np.shape(hebbian_matrix_created)[0]))
    assert np.allclose(diagonal_values_created, diagonal_values_optimal)

    # test if the created matrix is equal to the desired matrix
    hopfielnetwork = hn.HopfieldNetwork(np.array([[1, 1, -1, -1],[1, 1, -1, 1],[-1, 1, -1, 1]]))
    hebbian_matrix_test = np.array([[0., 0.33333333, -0.33333333, -0.33333333],
                                    [0.33333333, 0., -1, 0.33333333],
                                    [-0.33333333, -1, 0., -0.33333333],
                                    [-0.33333333, 0.33333333, -0.33333333, 0.]])
    hebbian_matrix_created = hopfielnetwork.hebbian_weights()
    assert np.allclose(hebbian_matrix_created, hebbian_matrix_test)


def test_create_h_matrix():
    # ensure that the the patterns (argument pattern)
    # and the memorized patterns (argument old_weights) are both not empty

    hopfielnetwork = hn.HopfieldNetwork(np.array([]), 'hebbian')
    old_weights_test = np.ones((2, 2))
    with pytest.raises(Exception):
        hopfielnetwork.create_h_matrix(hopfielnetwork.patterns, old_weights_test)

    old_weights_test = np.array(())
    with pytest.raises(Exception):
        hopfielnetwork.create_h_matrix(hopfielnetwork.patterns, old_weights_test)

    hopfielnetwork = hn.HopfieldNetwork(np.ones((2, 5)), 'hebbian')
    with pytest.raises(Exception):
        hopfielnetwork.create_h_matrix(hopfielnetwork.patterns, old_weights_test)

    # ensure that the h matrix created by the function is a square matrix
    hopfielnetwork = hn.HopfieldNetwork(np.array([-1, -1, 1, -1]), 'hebbian')
    length = len(hopfielnetwork.patterns)
    old_weights_test = np.zeros((length, length))
    x, y = np.shape(hopfielnetwork.create_h_matrix(hopfielnetwork.patterns, old_weights_test))
    assert x == y


def test_create_pre_synaptic_matrix():
    # test if the pattern and the h_matrix aren't empty
    hopfielnetwork = hn.HopfieldNetwork(np.array([]), 'hebbian')
    h_matrix = np.array([])
    with pytest.raises(Exception):
        hopfielnetwork.create_pre_synaptic_matrix(hopfielnetwork.patterns, h_matrix)

    with pytest.raises(Exception):
        hopfielnetwork.create_pre_synaptic_matrix(hopfielnetwork.patterns, np.array([1, 2]))

    hopfielnetwork = hn.HopfieldNetwork(np.array([1, 2]), 'hebbian')
    with pytest.raises(Exception):
        hopfielnetwork.create_pre_synaptic_matrix(hopfielnetwork.patterns, h_matrix)

    # test if the pattern and the h_matrix have the right dimensions to perform a matrix multiplication
    hopfielnetwork = hn.HopfieldNetwork(np.array((1, 3)), 'hebbian')
    h_matrix_created_0 = np.array((4, 4))
    with pytest.raises(Exception):
        hopfielnetwork.create_pre_synaptic_matrix(hopfielnetwork.patterns, h_matrix_created_0)

    # ensure that the pre-synaptic matrix at the end is a square matrix
    hopfielnetwork = hn.HopfieldNetwork(np.array([1, 2]), 'hebbian')
    weight_matrix_created_1 = np.ones((2, 2))
    x, y = np.shape(hopfielnetwork.create_h_matrix(hopfielnetwork.patterns, weight_matrix_created_1))
    assert x == y


def test_storkey_weights(benchmark):
    # benchmark in order to measure generation time of function storkey_weights
    patterns_for_benchmark = pa.Patterns(50, 500)
    hopfieldnetwork = hn.HopfieldNetwork(patterns_for_benchmark.patterns, 'storkey')
    benchmark.pedantic(hopfieldnetwork.storkey_weights, args=(), iterations=5)

    # ensure that the patterns are not an empty matrix
    hopfieldnetwork.reset()
    with pytest.raises(Exception):
        hopfieldnetwork.storkey_weights()

    hopfieldnetwork.reset(patterns=np.array([1, 0]))
    with pytest.raises(Exception):
        hopfieldnetwork.storkey_weights()

    # test if the storkey matrix has the correct size
    patterns = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
    hopfieldnetwork.reset(patterns)
    storkey_matrix_created = hopfieldnetwork.storkey_weights()
    x1, y1 = np.shape(patterns)
    x2, y2 = np.shape(storkey_matrix_created)
    assert y1 == x2
    assert y1 == y2

    # test if the created matrix is equal to the desired matrix
    storkey_matrix_test = np.array([[1.125, 0.25, -0.25, -0.5],
                                    [0.25, 0.625, -1, 0.25],
                                    [-0.25, -1, 0.625, -0.25],
                                    [-0.5, 0.25, -0.25, 1.125]])
    assert np.allclose(storkey_matrix_created, storkey_matrix_test)


def test_sigma():
    hopfieldnetwork = hn.HopfieldNetwork(np.array([1, 1]))
    # test sigma with an empty vector, normally the function will return an empty vector
    assert np.allclose(hopfieldnetwork.sigma(np.array([])), np.array([]))

    # ensure that the returned vector has the same length that the initial vector
    assert len(hopfieldnetwork.patterns) == len(hopfieldnetwork.sigma(hopfieldnetwork.patterns))

    # test if sigma works as wanted
    hopfieldnetwork.reset(np.array([1, -1, -5]))
    assert np.allclose(hopfieldnetwork.sigma(hopfieldnetwork.patterns), np.array([1, -1, -1]))

    hopfieldnetwork.reset(np.array([0, 0, 0]))
    assert np.allclose(hopfieldnetwork.sigma(hopfieldnetwork.patterns), np.array([1, 1, 1]))

    hopfieldnetwork.reset(np.array([-100, 1.9, -3.5]))
    assert np.allclose(hopfieldnetwork.sigma(hopfieldnetwork.patterns), np.array([-1, 1, -1]))


def test_update(benchmark):
    # benchmark in order to measure duration of the function update_async
    # the matrix weights_for_benchmark is created by calling the hebbian_weights function
    patterns_for_benchmark = pa.Patterns(50, 2500)
    hopfieldnetwork = hn.HopfieldNetwork(patterns_for_benchmark.patterns)
    weight = hopfieldnetwork.weights_creation()
    benchmark.pedantic(hopfieldnetwork.update, args=(patterns_for_benchmark.patterns[0], weight,), iterations=100)

    # ensure that the state and the weight are not empty
    with pytest.raises(Exception):
        hopfieldnetwork.update(np.array([]), np.ones((2, 2)))

    with pytest.raises(Exception):
        hopfieldnetwork.update(np.array([]), np.array([]))

    with pytest.raises(Exception):
        hopfieldnetwork.update(np.array([1, 1]), np.array([]))

    # ensure that the returned vector at the end has the same shape as the vector state passed as argument
    hopfieldnetwork.reset(np.ones((1, 2)))
    weight = hopfieldnetwork.weights_creation()
    assert hopfieldnetwork.patterns.size == hopfieldnetwork.update(hopfieldnetwork.patterns[0], weight).size


def test_update_async(benchmark):
    # benchmark in order to measure duration of the function update_async
    # the matrix weights_for_benchmark is created by calling the hebbian_weights function
    patterns_for_benchmark = pa.Patterns(50, 2500)
    hopfieldnetwork = hn.HopfieldNetwork(patterns_for_benchmark.patterns)
    weight = hopfieldnetwork.weights_creation()
    benchmark.pedantic(hopfieldnetwork.update_async, args=(patterns_for_benchmark.patterns[0], weight), iterations=100)

    # ensure that the state and the weight are not empty
    with pytest.raises(Exception):
        hopfieldnetwork.update_async(np.array([]), np.ones(2,2))

    with pytest.raises(Exception):
        hopfieldnetwork.update_async(np.array([]), np.array([]))

    with pytest.raises(Exception):
        hopfieldnetwork.update_async(np.array([1, 2]), np.array([]))

    # ensure that the returned vector at the end has the same shape as the vector state passed as argument
    hopfieldnetwork.reset(np.ones((1, 3)))
    weight = hopfieldnetwork.weights_creation()
    assert hopfieldnetwork.patterns.size == hopfieldnetwork.update_async(hopfieldnetwork.patterns[0], weight).size


def test_dynamics():
    # test that the state is not an empty pattern
    datasaver = ds.DataSaver()
    hopfieldnetwork = hn.HopfieldNetwork(np.ones((1,3)))
    with pytest.raises(Exception):
        hopfieldnetwork.dynamics(np.array([]), datasaver, 4)

    # test that the number of maximum iterations in positive
    with pytest.raises(Exception):
        hopfieldnetwork.dynamics(np.array(1, -1, 1), datasaver, -2)

    # ensure that the list of the whole state history at the end is not empty if max_iter ≠ 0
    state_created = np.ones((1, 3))
    max_iter = 10
    hopfieldnetwork.dynamics(state_created, datasaver, max_iter)
    assert np.array([]) != datasaver.states


def test_dynamics_async():
    datasaver = ds.DataSaver()
    hopfieldnetwork = hn.HopfieldNetwork(np.ones((1, 3)))
    # test that the state is not empty
    with pytest.raises(Exception):
        hopfieldnetwork.dynamics_async(np.array([]), datasaver, 2, 10)

    # test that the number of maximum iterations in positive
    with pytest.raises(Exception):
        hopfieldnetwork.dynamics_async(np.array(1, -1, 1), datasaver, -2, 20)

    # test that the convergence number is positive
    with pytest.raises(Exception):
        hopfieldnetwork.dynamics_async(np.array(1, -1, 1), datasaver, 2, -2)

    # test that the step is positive
    with pytest.raises(Exception):
        hopfieldnetwork.dynamics_async(np.array(1, -1, 1), datasaver, 2, 10, -1)

    # ensure that the list of the whole state history at the end is not empty
    state_created = np.array([1, 1, -1])
    max_iter = 10
    hopfieldnetwork.dynamics_async(state_created, datasaver, max_iter)
    assert np.array(()) != datasaver.states


if __name__ == '__main__':
    pytest.main()
