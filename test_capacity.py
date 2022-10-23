import capacity as ca
import pytest
import math


def test_network_capacity():
    # ensure that the number of neurons is >1 and an integer
    with pytest.raises(Exception):
        ca.network_capacity(0)
    with pytest.raises(Exception):
        ca.network_capacity(-2)
    with pytest.raises(Exception):
        ca.network_capacity(-1.3)
    with pytest.raises(Exception):
        ca.network_capacity(3.5)
    with pytest.raises(Exception):
        ca.network_capacity(1)

    # ensure that the rule used is either hebbian or storkey
    with pytest.raises(Exception):
        ca.network_capacity(3, 'None')


def test_experiment():
    size = 0
    num_patterns = 0
    num_perturb = -1
    num_trials = 0
    rule = ''
    # ensure that the size is bigger than 0
    with pytest.raises(Exception):
        ca.experiment(size, 3, 2, 4, 'hebbian')
    # ensure that the number of patterns is bigger than 0
    with pytest.raises(Exception):
        ca.experiment(2, num_patterns, 2, 4, 'storkey')
    # ensure that number of perturbation is not negative
    with pytest.raises(Exception):
        ca.experiment(2, 3, num_perturb, 4, 'Hebbian')
    # ensure that the number of trials is bigger than 0
    with pytest.raises(Exception):
        ca.experiment(2, 3, 2, num_trials, 'Storkey')
    # ensure that the rule is hebbian (or Hebbian) or storkey (or Storkey)
    with pytest.raises(Exception):
        ca.experiment(2, 3, 2, 4, rule)


def test_capacity_curve():
    with pytest.raises(Exception):
        ca.capacity_vs_neurons([], [])

    with pytest.raises(Exception):
        ca.capacity_vs_neurons([], [{'key1', 1}])

    with pytest.raises(Exception):
        ca.capacity_vs_neurons([{'key1', 1}], [])


def test_capacity_vs_neurons():
    with pytest.raises(Exception):
        ca.capacity_vs_neurons([], [])

    with pytest.raises(Exception):
        ca.capacity_vs_neurons([], [{'key1', 1}])

    with pytest.raises(Exception):
        ca.capacity_vs_neurons([{'key1', 1}], [])