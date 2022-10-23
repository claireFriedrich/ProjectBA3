import pytest
import numpy as np
import Patterns as pa


def test_init_patterns():
    # test if the matrix has the good shape for 3 and 50
    num_pattern = 3
    pattern_size = 50
    patterns = pa.Patterns(num_pattern, pattern_size)
    x, y = np.shape(patterns.patterns)
    assert (x == num_pattern)
    assert (y == pattern_size)

    # test if the matrix has the good shape for 80 and 1000
    num_pattern = 80
    pattern_size = 1000
    patterns = pa.Patterns(num_pattern, pattern_size)
    x, y = np.shape(patterns.patterns)
    assert (x == num_pattern)
    assert (y == pattern_size)

    # ensure that the values passed as arguments are positive
    with pytest.raises(Exception):
        pa.Patterns(-1, 1)

    with pytest.raises(Exception):
        pa.Patterns(1, -1)

    with pytest.raises(Exception):
        pa.Patterns(-1, -1)

    with pytest.raises(Exception):
        pa.Patterns(0, 1)

    with pytest.raises(Exception):
        pa.Patterns(1, 0)

    with pytest.raises(Exception):
        pa.Patterns(0, 0)

    # ensure that if Patterns is created with a given pattern, it only contains 1 or -1
    with pytest.raises(Exception):
        pa.Patterns(patterns=np.two((1, 3)))

    # ensure that if Patterns is created with given patterns, the final Patterns are equal to the given ones
    patterns = np.array([1, -1, 1])
    patterns_created = pa.Patterns(patterns=patterns)
    assert np.allclose(patterns_created.patterns, patterns)


def test_reset_patterns():
    patterns = pa.Patterns()
    patterns.reset()
    assert np.allclose(patterns.patterns, np.array([]))


def test_perturb_pattern():
    # ensure that the pattern given is not empty
    pattern_created_1 = np.array([])

    with pytest.raises(Exception):
        pa.Patterns.perturb_pattern(pattern_created_1, 4)

    # ensure that one element is not modified more than once
    pattern_created_2 = pa.Patterns(patterns=np.ones((1, 3)))
    assert np.allclose(np.array([-1, -1, -1]), pattern_created_2.perturb_pattern(5))

    pattern_created_2 = pa.Patterns(patterns=np.ones((1, 3)))
    assert np.allclose(np.array([-1, -1, -1]), pattern_created_2.perturb_pattern(3))

    # ensure that the elements of pattern are only 1 or -1
    pattern_created_3 = np.array([-1, 2, 1])

    with pytest.raises(Exception):
        pa.Patterns.perturb_pattern(pattern_created_3, 2)

    # ensure that the number of perturbations is >=0
    with pytest.raises(Exception):
        pa.Patterns.perturb_pattern(pattern_created_3, -2)

    # ensure that the number of perturbations is an integer
    with pytest.raises(Exception):
        pa.Patterns.perturb_pattern(pattern_created_3, 1.2)

    with pytest.raises(Exception):
        pa.Patterns.perturb_pattern(pattern_created_3, -10.2)

    # test that if number of perturbations is 0 the function return the same pattern
    pattern_created_4 = pa.Patterns(patterns=np.ones((1, 3)))
    assert np.allclose(pattern_created_4.patterns, pattern_created_4.perturb_pattern(0))


def test_replace_one_random_state():
    # ensure that state is not empty
    patterns = np.ones((2,3))
    state = np.array([])
    with pytest.raises(Exception):
        patterns.replace_one_random_state(state)


def test_pattern_match():
    # ensure that the pattern or the memorized patterns are not empty
    pattern_created_0 = np.array([])
    patterns = pa.Patterns(patterns=pattern_created_0)

    with pytest.raises(Exception):
        patterns.pattern_match(pattern_created_0)

    with pytest.raises(Exception):
        patterns.pattern_match(np.array([1, 2]))

    patterns.patterns = np.ones((1, 3))
    with pytest.raises(Exception):
        patterns.pattern_match(pattern_created_0)

    # ensure that the length of pattern is the same as the length of axis 1 of memorized_patterns
    pattern_created_1 = np.array([0, 0, 0])
    patterns.patterns = np.zeros((3, 4))

    with pytest.raises(Exception):
        patterns.pattern_match(pattern_created_1)

    # test if pattern_match returns None if pattern is not in memorized_patterns
    pattern_created_2 = np.array([-1, -1, -1, -1])
    patterns.patterns = np.ones((3, 4))
    assert patterns.pattern_match(pattern_created_2) is None

    # test if pattern match return the right index if pattern is in memorized_patterns
    # if the pattern is contained more than one time in memorized_patterns, the function returns the
    # index of the first
    pattern_created_3 = np.array([1, 1, 1])
    patterns.patterns = np.array([[1, 1, 1], [1, 1, 1]])
    assert patterns.pattern_match(pattern_created_3) == 0

    pattern_created_4 = np.array([1, 1, 1])
    patterns.patterns = np.array([[1, -1, -1], [1, 1, 1]])
    assert patterns.pattern_match(pattern_created_4) == 1


def test_replace_one_random_state():
    patterns = pa.Patterns()
    # ensure that the pattern to perturb (state) is not empty
    with pytest.raises(Exception):
        patterns.replace_one_random_state(np.array([]))

    # ensure that the pattern has been modified
    patterns_copy = patterns.patterns.copy()
    patterns.replace_one_random_state(np.ones((1, 2500)))
    assert not np.allclose(patterns.patterns, patterns_copy)


if __name__ == '__main__':
    pytest.main()

