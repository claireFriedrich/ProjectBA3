import numpy as np
import math
import Patterns as pa
import HopfieldNetwork as hn
import DataSaver as ds
import matplotlib.pyplot as plt
import pandas as pd


def network_capacity(num_neurons, rule='hebbian'):
    """
    This function calculates the capacity of a network given its number of neurons
    Parameters
    ----
    :param num_neurons: the number of neurons of the network (int>0)
    :param rule: the learning rule to use, hebbian per default
    Returns
    ----
    :return: number which corresponds to the network capacity
    """
    # ensure that num_neurons is an integer and >1
    if num_neurons <= 1:
        raise Exception('The number of neurons needs to be >0')
    if int(num_neurons) != num_neurons:
        raise Exception('The number of neurons needs to be an integer')

    # ensure that the rule is either hebbian or storkey
    if rule != 'hebbian' and rule != 'Hebbian' and rule != 'storkey' and rule != 'Storkey':
        raise Exception('The rule used needs to be either hebbian/Hebbian or storkey/Storkey')

    if rule == 'hebbian' or rule == 'Hebbian':
        return num_neurons/(2*math.log(num_neurons))
    elif rule == 'storkey' or rule == 'Storkey':
        return num_neurons/math.sqrt(2*math.log(num_neurons))


def generate_num_patterns(capacity):
    """
    This function generates a list of 10 numbers of patterns
    Parameters
    ----
    :param capacity: the capacity of the considered number of neurons.
    Returns
    ----
    :return: a list containing 10 linearly distributed pattern numbers
    """
    return np.linspace(0.5 * capacity, 2 * capacity, 10).astype(int)


# for one specific neuron size
def experiment(size, num_patterns, num_perturb, rule='hebbian', num_trials=10, max_iter=100):
    """
    This function does an experiment for one specific size of neurons.
    Parameters
    ----
    :param size: the size of neurons
    :param num_patterns: the number of patters
    :param num_perturb: the number of perturbation
    :param rule: the learning to use
    :param num_trials: the number of trials
    :param max_iter: the maximal number of iterations
    Returns
    ----
    :return: a dictionary with all the results for the experiment
    """
    result_dict = {'network_size': size, 'weight_rule': rule, 'num_patterns': num_patterns, 'num_perturb': num_perturb,
                   'match_frac': 0}
    counter_retrieved = 0
    frac = []
    for num_pattern in num_patterns:
        patterns = pa.Patterns(num_pattern, size)
        network = hn.HopfieldNetwork(patterns.patterns, rule)
        saver = ds.DataSaver()
        for i in range(0, num_trials):
            network.dynamics(patterns.perturb_pattern(num_perturb), saver)
            if patterns.pattern_match(saver.states[-1]) is not None:
                counter_retrieved += 1
        fraction = counter_retrieved / num_trials
        frac.append(fraction)
        counter_retrieved = 0
    result_dict['match_frac'] = frac
    return result_dict


def experiments():
    """
    This function carries out the capacity experiements on a Hopfield network, once using the Hebbian learning and once
    using the Storkey learning rule.
    Returns
    ----
    :return: two lists of dictionaries (one for Hebbian and one for Storkey)
    """
    # original list for sizes
    # sizes = [10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500]
    sizes = [10, 18, 34, 63, 116, 215, 397, 733]
    results_hebbian = []
    results_storkey = []
    percentage_perturb = 20/100
    for size in sizes:
        print("Start of process for n = ", size)
        capacity = network_capacity(size, 'hebbian')
        list_number_of_patterns = generate_num_patterns(capacity)
        results = experiment(size, list_number_of_patterns, int(percentage_perturb * size), 'hebbian')
        results_hebbian.append(results)
        capacity = network_capacity(size, 'storkey')
        list_number_of_patterns = generate_num_patterns(capacity)
        results = experiment(size, list_number_of_patterns, int(percentage_perturb * size), 'storkey')
        results_storkey.append(results)
        print("Finish of process for n = ", size)
    capacity_curve(results_hebbian, results_storkey)
    return results_hebbian, results_storkey


def capacity_curve(dict_results_hebbian, dict_results_storkey):
    """
    This function calculates the capacities in function of the pattern number for different network sizes.
    It stores the resulting plot on a chosen path on the computer

    Parameters
    ----
    :param dict_results_hebbian: a list of dictionaries
    :param dict_results_storkey: a list of dictionaries
    Returns
    ----
    :return: None
    """
    if len(dict_results_hebbian) == 0 or len(dict_results_storkey) == 0:
        raise Exception('Both dictionaries cannot be empty')

    fig, axs = plt.subplots(2, 4)
    fig.suptitle('Capacity curves')
    i = 0
    for ax in axs.flat:
        x = dict_results_storkey[i]['num_patterns']
        y_hebbian = dict_results_hebbian[i]['match_frac']
        y_storkey = dict_results_storkey[i]['match_frac']

        ax.plot(x, y_hebbian, color='g', label='hebbian')
        ax.plot(x, y_storkey, color='m', label='storkey')
        ax.set_title('n = ' + str(dict_results_hebbian[i]['network_size']), fontsize=8, fontweight='bold')

        ax.set_xlabel(xlabel='Number of patterns', fontsize=8)
        ax.set_ylabel(ylabel='Fraction retrieved', fontsize=8)
        ax.legend(prop={"size": 6})

        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

        i += 1

    fig.tight_layout()

    # save the curve on the computer
    out_path = "Capacity_curves.png"
    fig.savefig(out_path)


def capacity_vs_neurons(dict_results_hebbian, dict_results_storkey):
    """
    This function calculates the capacity in function of the number of neurons. 
    It allows also to compare the empirical capacity to the theoretical one.
    It also saves the plot on the computer.
    Parameters 
    ----
    :param dict_results_hebbian: a list of dictionaries
    :param dict_results_storkey: a list of dictionaries
    Returns 
    ----
    :return: None
    """

    capacity_exp_hebbian = []
    capacity_th_hebbian = []
    capacity_exp_storkey = []
    capacity_th_storkey = []

    for size in range(0, len(dict_results_hebbian)):
        for i in range(0, len(dict_results_hebbian[0]['match_frac'])):
            if dict_results_hebbian[size]['match_frac'][i] < 0.9:
                capacity_exp_hebbian.append(dict_results_hebbian[size]['num_patterns'][i])
                break
        capacity_th_hebbian.append(network_capacity(dict_results_hebbian[size]['network_size'], 'hebbian'))

    for size in range(0, len(dict_results_storkey)):
        for i in range(0, len(dict_results_storkey[0]['match_frac'])):
            if dict_results_storkey[size]['match_frac'][i] < 0.9:
                capacity_exp_storkey.append(dict_results_hebbian[size]['num_patterns'][i])
                break
        capacity_th_storkey.append(network_capacity(dict_results_storkey[size]['network_size'], 'storkey'))

    fig, axs = plt.subplots(1, 2)

    # original list for sizes
    # x = [10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500]
    x = [10, 18, 34, 63, 116, 215, 397, 733]

    axs[0].plot(x, capacity_th_hebbian, color='r', label='theoretical capacity')
    axs[0].plot(x, capacity_exp_hebbian, color='b', label='experimental capacity')
    axs[0].set_title('Capacity vs number of neurons (Hebbian)', fontsize=8)

    axs[1].plot(x, capacity_th_storkey, color='r', label='theoretical capacity')
    axs[1].plot(x, capacity_exp_storkey, color='b', label='experimental capacity')
    axs[1].set_title('Capacity vs number of neurons (Storkey)', fontsize=8)

    for ax in axs.flat:
        ax.set_xlabel(xlabel='Number of neurons', fontsize=8)
        ax.set_ylabel(ylabel='Limit capacity value', fontsize=8)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.legend(prop={"size": 6})

    # save the curve on the computer
    out_path = "Comparison_capacities.png"
    fig.savefig(out_path)


def robustness():
    """
    This function computes how much one can perturb a network (with different neuron numbers) in order to retrieve the
    original pattern
    It also stores the table of maximum percentage of perturbations for each network size and the plot of retrieved
    fraction for a given percentage of perturbation for each network size on the computer.
    Returns 
    ----
    :return: None
    """
    # original list for sizes
    # sizes = [10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500]
    sizes = [10, 18, 34, 63, 116, 215, 397, 733]
    results_hebbian = []
    results_storkey = []
    percentage_perturb = 0
    num_pattern = [2]

    for size in sizes:
        # hebbian
        dict_results_hebbian = experiment(size, num_pattern, int(percentage_perturb * size), 'hebbian')

        while dict_results_hebbian['match_frac'][0] >= 0.9:
            percentage_perturb += 5/100
            dict_results_hebbian = experiment(size, num_pattern, int(percentage_perturb * size), 'hebbian')
        results_hebbian.append(percentage_perturb*100)
        percentage_perturb = 0

        # storkey
        dict_results_storkey = experiment(size, num_pattern, int(percentage_perturb * size), 'storkey')

        while dict_results_storkey['match_frac'][0] >= 0.9:
            percentage_perturb += 5/100
            dict_results_storkey = experiment(size, num_pattern, int(percentage_perturb * size), 'storkey')
        results_storkey.append(percentage_perturb*100)
        percentage_perturb = 0

    fig = plt.figure()
    plt.plot(sizes, results_hebbian, color='g', label='hebbian')
    plt.plot(sizes, results_storkey, color='m', label='storkey')
    plt.title('Robustness', fontsize=10)
    plt.xlabel(xlabel='Number of neurons', fontsize=8)
    plt.ylabel(ylabel='Maximum percentage of perturbation (%)', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.legend(prop={"size": 6})

    # save the curve on the computer
    out_path = "Robustness_plot.png"
    fig.savefig(out_path)
    data = {'Number of neurons': sizes, '% of perturbation (Hebbian)': results_hebbian, '% of perturbation (Storkey)': results_storkey}
    df = pd.DataFrame(data=data, index=['10', '18', '34', '63', '116', '215', '397', '733'])
    gfg = df.to_markdown(index=False)
    print(gfg)
    
    # save the table on computer
    out_path = "Robustness_table.h5"
    df.to_hdf(out_path, key='df')

    # plot fraction of retrieved patterns vs perturb percentage
    frac_retrived_pattern_hebbian = []
    frac_retrived_pattern_storkey = []
    global_frac_hebbian = []
    global_frac_storkey = []

    for size in sizes:
        for i in range(0, 100, 5):
            frac_hebbian = experiment(size, num_pattern, int(i/100 * size), 'hebbian')
            frac_retrived_pattern_hebbian.append(frac_hebbian['match_frac'][0])

            frac_storkey = experiment(size, num_pattern, int(i / 100 * size), 'storkey')
            frac_retrived_pattern_storkey.append(frac_storkey['match_frac'][0])

        global_frac_hebbian.append(frac_retrived_pattern_hebbian)
        global_frac_storkey.append(frac_retrived_pattern_storkey)

        frac_retrived_pattern_hebbian = []
        frac_retrived_pattern_storkey = []

    fig, axs = plt.subplots(2, 4)
    fig.suptitle('Robustness curves')
    i = 0
    x = np.linspace(0, 100, 20, dtype=int)

    for ax in axs.flat:
        ax.plot(x, global_frac_hebbian[i], color='g', label='hebbian')
        ax.plot(x, global_frac_storkey[i], color='m', label='storkey')
        ax.set_title('n = ' + str(sizes[i]), fontsize=8, fontweight="bold")


        ax.set_xlabel(xlabel='Percentage of perturbation', fontsize=6)
        ax.set_ylabel(ylabel='Fraction retrieved', fontsize=6)
        ax.legend(prop={"size": 6})

        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

        i += 1

    fig.tight_layout()

    # save the curve on the computer
    out_path = "Robustness_curves.png"
    fig.savefig(out_path)
