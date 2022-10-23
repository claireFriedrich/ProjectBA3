import capacity as capacity
import Checkerboard as cb
import Patterns as pa
import HopfieldNetwork as hn
import DataSaver as ds


def main():
    # energy plots
    '''pattern_size = 1000
    num_pattern = 80
    num_perturb = 80

    patterns = pa.Patterns(num_pattern, pattern_size)

    network_hebbian = hn.HopfieldNetwork(patterns.patterns, 'hebbian')
    saver_hebbian = ds.DataSaver()

    network_storkey = hn.HopfieldNetwork(patterns.patterns, 'storkey')
    saver_storkey = ds.DataSaver()

    print('---ENERGY PLOTS FOR DYNAMICAL SYNCHRONOUS AND ASYNCHRONOUS SYSTEM FOR BOTH HEBBIAN AND STORKEY RULES---')

    print('Start of dynamical synchronous system with the hebbian rule')
    network_hebbian.dynamics(patterns.perturb_pattern(num_perturb), saver_hebbian)
    print('End of dynamical synchronous system with hebbian rule')
    print('Start of computation of energies for dynamical synchronous system with hebbian rule')
    saver_hebbian.compute_energies(network_hebbian.weights_creation())
    print('Energies for dynamical synchronous system with hebbian rule computed')
    print('Start of process of saving plot_hebbian synchronous')
    saver_hebbian.plot_energy()
    print('Plot_hebbian synchronous saved')

    print('Start of dynamical synchronous system with the storkey rule')
    network_storkey.dynamics(patterns.perturb_pattern(num_perturb), saver_storkey)
    print('End of dynamical synchronous system with storkey rule')
    print('Start of computation of energies for dynamical synchronous system with storkey rule')
    saver_storkey.compute_energies(network_storkey.weights_creation())
    print('Energies for dynamical synchronous system with storkey rule computed')
    print('Start of process of saving plot_storkey synchronous')
    saver_storkey.plot_energy()
    print('Plot_storkey synchronous saved')

    saver_hebbian.reset()
    saver_storkey.reset()

    print('Start of dynamical asynchronous system with the hebbian rule')
    network_hebbian.dynamics_async(patterns.perturb_pattern(num_perturb), saver_hebbian)
    print('End of dynamical asynchronous system with hebbian rule')
    print('Start of computation of energies for dynamical asynchronous system with hebbian rule')
    saver_hebbian.compute_energies(network_hebbian.weights_creation())
    print('Energies for dynamical asynchronous system with hebbian rule computed')
    print('Start of process of saving plot_hebbian asynchronous')
    saver_hebbian.plot_energy()
    print('Plot_hebbian asynchronous saved')

    print('Start of dynamical asynchronous system with the storkey rule')
    network_storkey.dynamics_async(patterns.perturb_pattern(num_perturb), saver_storkey)
    print('End of dynamical asynchronous system with storkey rule')
    print('Start of computation of energies for dynamical asynchronous system with storkey rule')
    saver_storkey.compute_energies(network_storkey.weights_creation())
    print('Energies for dynamical asynchronous system with storkey rule computed')
    print('Start of process of saving plot_storkey asynchronous')
    saver_storkey.plot_energy()
    print('Plot_storkey asynchronous saved')

    saver_hebbian.reset()
    saver_storkey.reset()

    # checkerboard
    pattern_size = 2500
    num_pattern = 50
    num_perturb = 1000
    sub_matrix_size = 5

    patterns = pa.Patterns(num_pattern, pattern_size)

    print('')
    print('---CHECKERBOARDS FOR DYNAMICAL SYNCHRONOUS AND ASYNCHRONOUS SYSTEM FOR BOTH HEBBIAN AND STORKEY RULES---')

    print('Start creation of Checkerboard')
    checkerboard = cb.Checkerboard(sub_matrix_size)
    print('Checkerboard created -> start of perturbation of Checkerboard')
    patterns.replace_one_random_state(checkerboard.vector.patterns[0])
    print('Checkerboard perturbed')

    network_hebbian.reset(patterns.patterns, "hebbian")
    print('Start of dynamical synchronous system with hebbian rule')
    network_hebbian.dynamics(checkerboard.vector.perturb_pattern(num_perturb), saver_hebbian)
    print('End of dynamical synchronous system with hebbian rule')
    print('Start of process of saving video_hebbian_synchronous')
    saver_hebbian.save_video()
    print('Video_hebbian_synchronous saved')

    network_storkey.reset(patterns.patterns, "storkey")
    print('Start of dynamical synchronous system with storkey rule')
    network_storkey.dynamics(checkerboard.vector.perturb_pattern(num_perturb), saver_storkey)
    print('End of dynamical synchronous system with storkey rule')
    print('Start of process of saving video_storkey_synchronous')
    saver_storkey.save_video()
    print('Video_storkey_synchronous saved')

    saver_hebbian.reset()
    saver_storkey.reset()

    network_hebbian.reset(patterns.patterns, "hebbian")
    print('Start of dynamical asynchronous system with hebbian rule')
    network_hebbian.dynamics_async(checkerboard.vector.perturb_pattern(num_perturb), saver_hebbian, skip=1000)
    print('End of dynamical asynchronous system with hebbian rule')
    print('Start of process of saving video_hebbian_asynchronous')
    saver_hebbian.save_video()
    print('Video_hebbian_asynchronous saved')

    network_storkey.reset(patterns.patterns, "storkey")
    print('Start of dynamical asynchronous system with storkey rule')
    network_storkey.dynamics_async(checkerboard.vector.perturb_pattern(num_perturb), saver_storkey, skip=1000)
    print('End of dynamical asynchronous system with storkey rule')
    print('Start of process of saving video_storkey_asynchronous')
    saver_storkey.save_video()
    print('Video_storkey_asynchronous saved')'''

    # capacity and robustness
    print('---CAPACITY AND ROBUSTNESS CURVES---')
    results_hebbian, results_storkey = capacity.experiments()
    capacity.capacity_vs_neurons(results_hebbian, results_storkey)
    print('Begin robustness')
    capacity.robustness()

    # checkerboard with an image
    pattern_size = 10000
    num_pattern = 50
    num_perturb = 1000

    patterns = pa.Patterns(num_pattern, pattern_size)

    saver_hebbian = ds.DataSaver()

    print('')
    print('---CHECKERBOARDS OF A GIVEN IMAGE FOR DYNAMICAL SYNCHRONOUS AND ASYNCHRONOUS SYSTEM '
          'FOR BOTH HEBBIAN AND STORKEY RULES---')

    print('Start creation of Checkerboard of image')
    checkerboard_image = cb.Checkerboard(given_image='star_100_100.png')
    print('Checkerboard of image created -> start of perturbation of Checkerboard of image')
    patterns.replace_one_random_state(checkerboard_image.vector.patterns[0])
    print('Checkerboard of image perturbed')

    network_hebbian = hn.HopfieldNetwork(patterns.patterns, "hebbian")
    print('Start of dynamical synchronous system with hebbian rule')
    network_hebbian.dynamics(checkerboard_image.vector.perturb_pattern(num_perturb), saver_hebbian)
    print('End of dynamical synchronous system with hebbian rule')
    print('Start of process of saving video_hebbian_synchronous of image')
    saver_hebbian.save_video(img_shape=(100, 100), title='image ')
    print('Video_hebbian_synchronous of image saved')


if __name__ == '__main__':
    main()
