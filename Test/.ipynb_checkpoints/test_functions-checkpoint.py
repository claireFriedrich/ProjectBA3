from unittest import TestCase
import numpy as np
import functions as func
import additional_functions as add_func


class Test(TestCase):
    def test_generate_patterns(self):
        # test if the matrix has the good shape for 3 and 50
        num_pattern = 3
        pattern_size = 50
        patterns_generated = func.generate_patterns(num_pattern, pattern_size)
        x, y = np.shape(patterns_generated)
        self.assertEqual(x, num_pattern)
        self.assertEqual(y, pattern_size)

        # test if the matrix has the good shape for 80 and 1000
        num_pattern = 80
        pattern_size = 1000
        patterns_generated = func.generate_patterns(num_pattern, pattern_size)
        x, y = np.shape(patterns_generated)
        self.assertEqual(x, num_pattern)
        self.assertEqual(y, pattern_size)

        # ensure that the values passed as arguments are positive
        with self.assertRaises(Exception):
            func.generate_patterns(-1, 1)

        with self.assertRaises(Exception):
            func.generate_patterns(1, -1)

        with self.assertRaises(Exception):
            func.generate_patterns(-1, -1)

        with self.assertRaises(Exception):
            func.generate_patterns(0, 1)

        with self.assertRaises(Exception):
            func.generate_patterns(1, 0)

        with self.assertRaises(Exception):
            func.generate_patterns(0, 0)

    def test_perturb_pattern(self):
        # ensure that the pattern given is not empty
        pattern_created_1 = np.array([])

        with self.assertRaises(Exception):
            func.perturb_pattern(pattern_created_1, 4)

        # ensure that one element is not modify more than once
        pattern_created_2 = np.array([1, 1, 1, 1, 1, 1, 1])
        self.assertTrue(
            np.allclose(np.array([-1, -1, -1, -1, -1, -1, -1]), func.perturb_pattern(pattern_created_2, 10)))

        # ensure that the elements of pattern are only 1 or -1
        pattern_created_3 = np.array([-1, 2, 1])

        with self.assertRaises(Exception):
            func.perturb_pattern(pattern_created_3, 2)

        # test that if number of perturbations is 0 the function return the same pattern
        pattern_created_4 = np.array([1, 1, 1])
        self.assertTrue(np.allclose(pattern_created_4, func.perturb_pattern(pattern_created_4, 0)))

    def test_pattern_match(self):
        # ensure that the pattern or the memorized patterns are not empty
        pattern_created_0 = np.array([])
        memorized_patterns_created_0 = np.array([])

        with self.assertRaises(Exception):
            func.pattern_match(memorized_patterns_created_0, pattern_created_0)

        with self.assertRaises(Exception):
            func.pattern_match(np.array([1, 2]), pattern_created_0)

        with self.assertRaises(Exception):
            func.pattern_match(memorized_patterns_created_0, np.array([1, 2]))

        with self.assertRaises(Exception):
            func.pattern_match(np.ones((1, 0)), np.array([1, 2]))

        with self.assertRaises(Exception):
            func.pattern_match(np.ones((0, 1)), np.array([1, 2]))

        # ensure that the length of pattern is the same as the length of axis 1 of memorized_patterns
        pattern_created_1 = np.array([0, 0, 0])
        memorized_patterns_created_1 = np.zeros((10, 9))

        with self.assertRaises(Exception):
            func.pattern_match(memorized_patterns_created_1, pattern_created_1)

        # test if pattern_match return None if pattern is not in memorized_patterns
        pattern_created_2 = np.array([0, 0, 0, 0])
        memorized_patterns_created_2 = np.ones((3, 4))
        self.assertEqual(func.pattern_match(memorized_patterns_created_2, pattern_created_2), None)

        # test if pattern match return the right index if pattern is in memorized_patterns
        pattern_created_3 = np.array([1, 2, 3])
        memorized_patterns_created_3 = np.array([[1, 2, 0], [1, 2, 3]])
        self.assertEqual(func.pattern_match(memorized_patterns_created_3, pattern_created_3), 1)

    def test_hebbian_weights(self):
        # ensure that the patterns is not an empty matrix
        with self.assertRaises(Exception):
            func.hebbian_weights(np.array([]))

        with self.assertRaises(Exception):
            func.hebbian_weights(np.array((1, 0)))

        with self.assertRaises(Exception):
            func.hebbian_weights(np.array((0, 1)))

        # test if the hebbian matrix has the correct size
        matrix_test = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
        hebbian_matrix_created = func.hebbian_weights(matrix_test)
        x1, y1 = np.shape(matrix_test)
        x2, y2 = np.shape(hebbian_matrix_created)
        self.assertEqual(y1, x2)
        self.assertEqual(y1, y2)

        # test if the values of the hebbian matrix are in the correct range
        self.assertTrue(np.where(np.logical_and(hebbian_matrix_created >= -1, hebbian_matrix_created <= 1)))

        # test if the hebbian matrix is symmetric
        self.assertTrue(np.allclose(hebbian_matrix_created.T, hebbian_matrix_created))

        # test that all the elements on the diagonal are 0
        diagonal_values_created = np.diag(hebbian_matrix_created)
        diagonal_values_optimal = np.zeros((1, np.shape(hebbian_matrix_created)[0]))
        self.assertTrue(np.allclose(diagonal_values_created, diagonal_values_optimal))

        # test if the created matrix is equal to the desired matrix
        hebbian_matrix_test = np.array([[0., 0.33333333, -0.33333333, -0.33333333],
                                        [0.33333333, 0., -1, 0.33333333],
                                        [-0.33333333, -1, 0., -0.33333333],
                                        [-0.33333333, 0.33333333, -0.33333333, 0.]])
        hebbian_matrix_created = func.hebbian_weights(matrix_test)
        self.assertTrue(np.allclose(hebbian_matrix_created, hebbian_matrix_test))

    def test_sigma(self):
        # test sigma with an empty vector, normally the function will return an empty vector
        self.assertTrue(np.allclose(func.sigma(np.array([])), func.sigma(np.array([]))))

        # ensure that the returned vector has the same length that the initial vector
        vector_created_0 = np.array([1, 2])
        self.assertEqual(len(vector_created_0), len(func.sigma(vector_created_0)))

        # test if sigma works as wanted
        vector_created_1 = np.array([1, -1, -5])
        self.assertTrue(np.allclose(func.sigma(vector_created_1), np.array([1, -1, -1])))

        vector_created_2 = np.array([0, 0, 0])
        self.assertTrue(np.allclose(func.sigma(vector_created_2), np.array([1, 1, 1])))

        vector_created_3 = np.array([-100, 1.9, -3.5])
        self.assertTrue(np.allclose(func.sigma(vector_created_3), np.array([-1, 1, -1])))

    def test_update(self):
        # ensure that the length of state and the length of the axis-1 of weights are equal
        state_created_0 = np.array([1, 2, 3])
        weights_created_0 = np.array([[1, 2], [1, 2]])
        with self.assertRaises(Exception):
            func.update(state_created_0, weights_created_0)

        # ensure that the state or the weights are not empty
        with self.assertRaises(Exception):
            func.update(np.array([]), weights_created_0)

        with self.assertRaises(Exception):
            func.update(state_created_0, np.array([]))

        with self.assertRaises(Exception):
            func.update(np.array([]), np.array([]))

    def test_update_async(self):
        # ensure that the length of state and the length of the axis-1 of weights are equal
        state_created_0 = np.array([1, 2, 3])
        weights_created_0 = np.array([[1, 2], [1, 2]])
        with self.assertRaises(Exception):
            func.update_async(state_created_0, weights_created_0)

        # ensure that the state or the weights are not empty
        with self.assertRaises(Exception):
            func.update_async(np.array([]), weights_created_0)

        with self.assertRaises(Exception):
            func.update_async(state_created_0, np.array([]))

        with self.assertRaises(Exception):
            func.update_async(np.array([]), np.array([]))

    def test_dynamics(self):
        # test that the state is not an empty pattern
        with self.assertRaises(Exception):
            func.dynamics(np.array([]), np.array((3, 3)), 4)

        # test that the wight matrix is not an empty matrix
        with self.assertRaises(Exception):
            func.dynamics(np.array(-1, 1, -1), np.array([]), 4)

        # test that the number of maximum iterations in positive
        with self.assertRaises(Exception):
            func.dynamics(np.array(1, -1, 1), np.array((3, 3)), -2)

    def test_dynamics_async(self):
        # test that the state is not empty
        with self.assertRaises(Exception):
            func.dynamics_async(np.array([]), np.array((3,3)), 2, 10)

        # test that the wight matrix is not an empty matrix
        with self.assertRaises(Exception):
            func.dynamics_async(np.array(-1, 1, -1), np.array([]), 4, 20)

        # test that neither state nor weights are empty
        with self.assertRaises(Exception):
            func.dynamics_async(np.array([]), np.array([]), 4, 20)

        # test that the number of maximum iterations in positive
        with self.assertRaises(Exception):
            func.dynamics_async(np.array(1, -1, 1), np.array((3, 3)), -2, 20)

        # test that the convergence number is positive
        with self.assertRaises(Exception):
            func.dynamics_async(np.array(1, -1, 1), np.array((3, 3)), 2, -2)

        # test that the step is positive
        with self.assertRaises(Exception):
            func.dynamics_async(np.array(1, -1, 1), np.array((3, 3)), 2, 10, -1)

    def test_storkey_weights(self):
        # ensure that the patterns are not an empty matrix
        with self.assertRaises(Exception):
            func.storkey_weights(np.array([]))

        with self.assertRaises(Exception):
            func.storkey_weights(np.array((1, 0)))

        with self.assertRaises(Exception):
            func.storkey_weights(np.array((0, 1)))

        # test if the storkey matrix has the correct size
        matrix_test = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
        storkey_matrix_created = func.storkey_weights(matrix_test)
        x1, y1 = np.shape(matrix_test)
        x2, y2 = np.shape(storkey_matrix_created)
        self.assertEqual(y1, x2)
        self.assertEqual(y1, y2)

        # TODO do we have values for the range?? -> PIAZZA réponse
        # test if the values of the storkey matrix are in the correct range
        # self.assertTrue(np.where(np.logical_and(storkey_matrix_created >= -1, storkey_matrix_created <= 1)))

        # test if the created matrix is equal to the desired matrix
        storkey_matrix_test = np.array([[1.125, 0.25, -0.25, -0.5],
                                        [0.25, 0.625, -1, 0.25],
                                        [-0.25, -1, 0.625, -0.25],
                                        [-0.5, 0.25, -0.25, 1.125]])
        self.assertTrue(np.allclose(storkey_matrix_created, storkey_matrix_test))

    def test_energy(self):
        # ensure that the state or the weights are not empty
        with self.assertRaises(Exception):
            func.energy(np.array([]), np.ones((1, 2)))

        with self.assertRaises(Exception):
            func.energy(np.array((1, 0)), np.ones((1, 2)))

        with self.assertRaises(Exception):
            func.energy(np.array((0, 1)), np.ones((1, 2)))

        with self.assertRaises(Exception):
            func.energy(np.ones((1, 2)), np.array([]))

        with self.assertRaises(Exception):
            func.energy(np.ones((1, 2)), np.array((1, 0)))

        with self.assertRaises(Exception):
            func.energy(np.ones((1, 2)), np.array((0, 1)))

        # ensure that the length of the state and the length of axis 0 of weights are equal
        state_created = np.array([1, 2, 3])
        weights_created = np.ones((1, 3))
        with self.assertRaises(Exception):
            func.energy(state_created, weights_created)

    def test_save_video(self):
        # ensure that state_list is not empty
        with self.assertRaises(Exception):
            func.save_video(np.array([]), 'title', 'out_path')

    # TESTS FOR THE EVOLUTION OF THE SYSTEM (NO SPECIFIC FUNCTION TESTS)

    def test_convergence_hebbian_sync(self):
        print("TEST CONVERGENCE HEBBIAN SYNC")
        num_pattern = 80
        pattern_size = 100
        nbr_perturbation = 10
        patterns_generated = func.generate_patterns(num_pattern, pattern_size)
        print("Start of creation of the hebbian weights matrix")
        weight_matrix_created = func.hebbian_weights(patterns_generated)
        print("Hebbian weights matrix created")

        base_pattern = patterns_generated[np.random.randint(0, len(patterns_generated))]

        base_pattern_modif = func.perturb_pattern(base_pattern, nbr_perturbation)

        print("Start of the sync dynamical system")
        patterns_new_list = func.dynamics(base_pattern_modif, weight_matrix_created, 20)
        print("Sync dynamical system completed")

        print("Start of testing if the original pattern is retrieved")
        if func.pattern_match(patterns_generated, patterns_new_list[-1]) is not None:
            print("--> Original network retrieved! [synchronous update]")
        else:
            print("--> Original network NOT retrieved...")

        print("Testing if the energy function is non-increasing")
        energy_list_created = add_func.energies(patterns_new_list, weight_matrix_created)
        for i in range(1, len(energy_list_created)):
            self.assertGreaterEqual(energy_list_created[i - 1], energy_list_created[i])
        print("Energy function test completed")

    def test_convergence_hebbian_async(self):
        print("TEST CONVERGENCE HEBBIAN ASYNC")
        num_pattern = 3
        pattern_size = 50
        nbr_perturbation = 10
        patterns_generated = func.generate_patterns(num_pattern, pattern_size)
        print("Start of creation of the hebbian weights matrix")
        weight_matrix_created = func.hebbian_weights(patterns_generated)
        print("Hebbian weights matrix created")

        base_pattern = patterns_generated[np.random.randint(0, len(patterns_generated))]

        base_pattern_modif = func.perturb_pattern(base_pattern, nbr_perturbation)

        print("Start of the async dynamical system")
        patterns_new_list = func.dynamics_async(base_pattern_modif, weight_matrix_created, 20000, 3000)
        print("Async dynamical system completed")

        print("Start of testing if the original pattern is retrieved")
        if func.pattern_match(patterns_generated, patterns_new_list[-1]) is not None:
            print("--> Original network retrieved! [asynchronous update]")
        else:
            print("--> Original network NOT retrieved...")

        print("Testing if the energy function is non-increasing")
        energy_list_created = add_func.energies(patterns_new_list, weight_matrix_created)
        for i in range(1, len(energy_list_created)):
            self.assertGreaterEqual(energy_list_created[i - 1], energy_list_created[i])
        print("Energy function test completed")

    def test_convergence_storkey_sync(self):
        print("TEST CONVERGENCE STORKEY SYNC")
        num_pattern = 3
        pattern_size = 50
        nbr_perturbation = 10
        patterns_generated = func.generate_patterns(num_pattern, pattern_size)
        print("Start of creation of the storkey weights matrix")
        weight_matrix_created = func.storkey_weights(patterns_generated)
        print("Storkey weights matrix created")

        base_pattern = patterns_generated[np.random.randint(0, len(patterns_generated))]

        base_pattern_modif = func.perturb_pattern(base_pattern, nbr_perturbation)

        print("Start of the sync dynamical system")
        patterns_new_list = func.dynamics(base_pattern_modif, weight_matrix_created, 20)
        print("Sync dynamical system completed")

        print("Start of testing if the original pattern is retrieved")
        if func.pattern_match(patterns_generated, patterns_new_list[-1]) is not None:
            print("--> Original network retrieved! [synchronous update]")
        else:
            print("--> Original network NOT retrieved...")

        print("Testing if the energy function is non-increasing")
        energy_list_created = add_func.energies(patterns_new_list, weight_matrix_created)
        for i in range(1, len(energy_list_created)):
            self.assertGreaterEqual(energy_list_created[i - 1], energy_list_created[i])
        print("Energy function test completed")

    def test_convergence_storkey_async(self):
        print("TEST CONVERGENCE STORKEY ASYNC")
        num_pattern = 3
        pattern_size = 50
        nbr_perturbation = 10
        patterns_generated = func.generate_patterns(num_pattern, pattern_size)
        print("Start of creation of the storkey weights matrix")
        weight_matrix_created = func.storkey_weights(patterns_generated)
        print("Storkey weights matrix created")

        base_pattern = patterns_generated[np.random.randint(0, len(patterns_generated))]

        base_pattern_modif = func.perturb_pattern(base_pattern, nbr_perturbation)

        print("Start of the async dynamical system")
        patterns_new_list = func.dynamics_async(base_pattern_modif, weight_matrix_created, 20000, 3000)
        print("Async dynamical system completed")

        print("Start of testing if the original pattern is retrieved")
        if func.pattern_match(patterns_generated, patterns_new_list[-1]) is not None:
            print("--> Original network retrieved! [asynchronous update]")
        else:
            print("--> Original network NOT retrieved...")

        print("Testing if the energy function is non-increasing")
        energy_list_created = add_func.energies(patterns_new_list, weight_matrix_created)
        for i in range(1, len(energy_list_created)):
            self.assertGreaterEqual(energy_list_created[i - 1], energy_list_created[i])
        print("Energy function test completed")





