import HopfieldNetwork as hn
import Patterns as pa
import DataSaver as ds


def test_convergence_hebbian_sync():
    print("TEST CONVERGENCE HEBBIAN SYNC")
    num_pattern = 3
    pattern_size = 50
    nbr_perturbation = 9
    patterns_generated = pa.Patterns(num_pattern, pattern_size)
    hopfield_network = hn.HopfieldNetwork(patterns_generated.patterns, 'hebbian')
    saver = ds.DataSaver()

    print("Start of the sync dynamical system")
    hopfield_network.dynamics(patterns_generated.perturb_pattern(nbr_perturbation), saver)
    print("Sync dynamical system completed")

    print("Start of testing if the original pattern is retrieved")
    if patterns_generated.pattern_match(saver.states[-1]) is not None:
        print("--> Original network retrieved! [synchronous update]")
    else:
        print("--> Original network NOT retrieved...")

    print("Testing if the energy function is non-increasing")
    saver.compute_energies(hopfield_network.weights_creation())
    energy_list_created = saver.energies
    for i in range(1, len(energy_list_created)):
        assert energy_list_created[i - 1] >= energy_list_created[i]
    print("Energy function test completed")


def test_convergence_storkey_sync():
    print("TEST CONVERGENCE STORKEY SYNC")
    num_pattern = 3
    pattern_size = 50
    nbr_perturbation = 9
    patterns_generated = pa.Patterns(num_pattern, pattern_size)
    hopfield_network = hn.HopfieldNetwork(patterns_generated.patterns, 'storkey')
    saver = ds.DataSaver()

    print("Start of the sync dynamical system")
    hopfield_network.dynamics(patterns_generated.perturb_pattern(nbr_perturbation), saver)
    print("Sync dynamical system completed")

    print("Start of testing if the original pattern is retrieved")
    if patterns_generated.pattern_match(saver.states[-1]) is not None:
        print("--> Original network retrieved! [synchronous update]")
    else:
        print("--> Original network NOT retrieved...")

    print("Testing if the energy function is non-increasing")
    saver.compute_energies(hopfield_network.weights_creation())
    energy_list_created = saver.energies
    for i in range(1, len(energy_list_created)):
        assert energy_list_created[i - 1] >= energy_list_created[i]
    print("Energy function test completed")


def test_convergence_hebbian_async():
    print("TEST CONVERGENCE HEBBIAN ASYNC")
    num_pattern = 3
    pattern_size = 50
    nbr_perturbation = 9
    patterns_generated = pa.Patterns(num_pattern, pattern_size)
    hopfield_network = hn.HopfieldNetwork(patterns_generated.patterns, 'hebbian')
    saver = ds.DataSaver()

    print("Start of the async dynamical system")
    hopfield_network.dynamics_async(patterns_generated.perturb_pattern(nbr_perturbation), saver)
    print("Sync async dynamical system completed")

    print("Start of testing if the original pattern is retrieved")
    if patterns_generated.pattern_match(saver.states[-1]) is not None:
        print("--> Original network retrieved! [asynchronous update]")
    else:
        print("--> Original network NOT retrieved...")

    print("Testing if the energy function is non-increasing")
    saver.compute_energies(hopfield_network.weights_creation())
    energy_list_created = saver.energies
    for i in range(1, len(energy_list_created)):
        assert energy_list_created[i - 1] >= energy_list_created[i]
    print("Energy function test completed")


def test_convergence_storkey_async():
    print("TEST CONVERGENCE STORKEY ASYNC")
    num_pattern = 3
    pattern_size = 50
    nbr_perturbation = 9
    patterns_generated = pa.Patterns(num_pattern, pattern_size)
    hopfield_network = hn.HopfieldNetwork(patterns_generated.patterns, 'storkey')
    saver = ds.DataSaver()

    print("Start of the sync async dynamical system")
    hopfield_network.dynamics_async(patterns_generated.perturb_pattern(nbr_perturbation), saver)
    print("Sync async dynamical system completed")

    print("Start of testing if the original pattern is retrieved")
    if patterns_generated.pattern_match(saver.states[-1]) is not None:
        print("--> Original network retrieved! [asynchronous update]")
    else:
        print("--> Original network NOT retrieved...")

    print("Testing if the energy function is non-increasing")
    saver.compute_energies(hopfield_network.weights_creation())
    energy_list_created = saver.energies
    for i in range(1, len(energy_list_created)):
        assert energy_list_created[i - 1] >= energy_list_created[i]
    print("Energy function test completed")

