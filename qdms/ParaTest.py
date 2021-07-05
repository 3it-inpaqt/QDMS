from Circuit import Circuit
from MemristorSimulation import MemristorSimulation
from HelperFunction import is_square
import numpy as np
from QDSimulation import QDSimulation
from Plot import plot_everything
from Data_Driven import Data_Driven
from PulsedProgramming import PulsedProgramming
from HelperFunction import *
from Log import save_everything_hdf5
from Log import save_memristor_simulation_hdf5
import os
import time


def parametric_test_resolution_memristor(path, configurations=None):
    """
    This function generates many memristor simulations and saves them in a folder with proper names to use
    load_resolution_memristor_data() on it.

    Parameters
    ----------
    path : string
        The path to the saving folder.

    configurations : list of list
        The configurations simulated with this disposition: [number of memristor, distribution type, number of states]

    Returns
    -------

    """
    if configurations is None:
        conf_4_linear = [[4, 'linear', i] for i in range(2, 13)]
        # conf_4_half_spread = [[4, 'half_spread', i] for i in range(2, 13)]
        conf_4_full_spread = [[4, 'full_spread', i] for i in range(2, 13)]
        conf_9_linear = [[9, 'linear', i] for i in range(2, 10)]
        # conf_9_half_spread = [[9, 'half_spread', i] for i in range(2, 8)]
        conf_9_full_spread = [[9, 'full_spread', i] for i in range(2, 7)]
        configurations = conf_9_full_spread

    config_done = 0
    res = Data_Driven()
    for configuration in configurations:
        print(f'{len(configurations) - config_done} configurations left')
        circuit = Circuit(memristor_model=res, number_of_memristor=configuration[0], is_new_architecture=True, v_in=1e-3
                          , gain_resistance=0, R_L=1)

        pulsed_programming = PulsedProgramming(circuit, configuration[2], distribution_type=configuration[1]
                                               , max_voltage=2, tolerance=10, variance_read=0,variance_write=0
                                               , lrs=1000, hrs=3000, pulse_algorithm='log', number_of_reading=1)
        pulsed_programming.simulate()
        memristor_sim = MemristorSimulation(pulsed_programming, is_using_conductance=False, verbose=True)
        memristor_sim.simulate()

        directory_name = f'{int(np.sqrt(configuration[0]))}x{int(np.sqrt(configuration[0]))}_{configuration[1]}_{configuration[2]}_states'

        save_everything_hdf5(path, directory_name, memristor=res, pulsed_programming=pulsed_programming, circuit=circuit
                        , memristor_sim=memristor_sim, verbose=True)
        config_done += 1


def parametric_test_resolution_variance(path, configuration, variability=None, nb_occurences=100, verbose=False, light=False):
    """
    This function generates many memristor simulations and saves them in a folder with proper names to use
    load_resolution_variance() on it.

    Parameters
    ----------
    path : string
        The path to the saving folder.

    configuration : iterable
        The configuration simulated with this disposition: [number of memristor, distribution_type, number of states, tolerance].

    variability : list of float
        Contains the variability used. Default: [0, 0.1, 0.5, 1, 2, 5, 10, 20] %

    nb_occurences : int
        Number of occurences per point.

    verbose : bool
        If true, output timers in console.

    light : bool
        If true, the memristor_sim_data.txt file will change for memristor_sim_data_light.txt and will not contains all
        voltages and resistances, but only the std and resolution which is the necessary information for create_resolution_variance_plot.
    Returns
    -------

    """

    if variability is None:
        variability = np.array([0, 0.1, 0.5, 1, 2, 3, 4, 5])
    variances = variability / 300
    config_done = 0
    res = Data_Driven()
    circuit = Circuit(memristor_model=res, number_of_memristor=configuration[0], is_new_architecture=True, v_in=1e-3
                      , gain_resistance=0, R_L=1)

    nb_memristor = [int(np.sqrt(configuration[0])),int(np.sqrt(configuration[0]))] if is_square(configuration[0]) else [configuration[0], 1]
    directory_name = f'{path}//{nb_memristor[0]}x{nb_memristor[1]}_{configuration[1]}_{configuration[2]}_states_{configuration[3]}_tolerance'

    if not os.path.isdir(f'{path}'):
        os.mkdir(f'{path}')
    if not os.path.isdir(f'{directory_name}'):
        os.mkdir(f'{directory_name}')
    if verbose:
        start = time.time()
        start_ = start

    for variance in variances:
        sub_directory_name = directory_name + f'//{round(variance * 300, 2)}'
        if not os.path.isdir(f'{sub_directory_name}'):
            os.mkdir(f'{sub_directory_name}')

        for index in range(nb_occurences):
            if verbose:
                print(f'{len(variances) - config_done} variances left {index} simulation -> {(((len(variances) - config_done) * nb_occurences) - index) * (time.time() - start_)} s left')
                start_ = time.time()

            pulsed_programming = PulsedProgramming(circuit, configuration[2], distribution_type=configuration[1]
                                                   , max_voltage=2, tolerance=configuration[3], variance_read=variance, variance_write=variance
                                                   , lrs=1000, hrs=3000, pulse_algorithm='log', number_of_reading=1)
            pulsed_programming.simulate()
            memristor_sim = MemristorSimulation(pulsed_programming, is_using_conductance=False, verbose=False)
            memristor_sim.simulate()
            while os.path.isdir(f'{sub_directory_name}\\{str(index)}'):
                index += 1

            if light:
                save_everything_hdf5(sub_directory_name, str(index), memristor=res, pulsed_programming=pulsed_programming, circuit=circuit
                               , verbose=False)
                save_memristor_simulation_hdf5(memristor_sim, f'{sub_directory_name}\\{index}', light=True)
            else:
                save_everything_hdf5(sub_directory_name, str(index), memristor=res, pulsed_programming=pulsed_programming, circuit=circuit
                                , memristor_sim=memristor_sim, verbose=False)

        config_done += 1

    if verbose:
        print(f'For {len(variances) * nb_occurences} simulations -> {time.time() - start} s')


def parametric_test_pulsed_programming(path, configurations=None, verbose=False, nb_occurences=10, plot=False):
    """
    This function generates many pulsed programming simulations and saves them in a folder with proper names to use
    load_resolution_variance() on it.

    Parameters
    ----------
    path : string
        The path to the saving folder.

    configurations : list of list
        The configurations simulated with this disposition: [number_of_reading, pulse_algorithm, tolerance, is_relative_tolerance, variance_read, variance_write, lrs_hrs]
        Where lrs_hrs is a list of list : [[0, 0.2], [0.2, 0.4], [0.6, 0.8], [0.8, 1]], where 0=r_on and 1=r_off
        variability_read and variability_write is in % and will change variance_read and variance_write

    verbose : bool
        If true, output timers in console.
    Returns
    -------

    """
    if configurations is None:
        configurations = []
        number_of_reading = [20]
        pulse_algorithm = ['fabien', 'log']
        tolerance = [0.5, 1, 2, 5, 10]
        is_relative_tolerance = [True]
        variance_read = list(np.array([0, 1, 5])/300)
        # variance_write = list(np.array([0, 0.1, 0.5, 1, 2, 3, 4, 5])/300)
        # variance_read = list(np.array(([0, 1, 5]))/300)
        variance_write = [0.5/300]
        lrs_hrs = [[0.4, 0.6]]
        for number_of_reading_ in number_of_reading:
            for pulse_algorithm_ in pulse_algorithm:
                for tolerance_ in tolerance:
                    for is_relative_tolerance_ in is_relative_tolerance:
                        for variance_read_ in variance_read:
                            for variance_write_ in variance_write:
                                for lrs_hrs_ in lrs_hrs:
                                    configuration_ = [number_of_reading_, pulse_algorithm_, tolerance_, is_relative_tolerance_, variance_read_, variance_write_, lrs_hrs_]
                                    configurations.append(configuration_)
    config_done = 0

    offset = 0
    for index in range(len(configurations)):
        index -= offset
        if configurations[index][6][0] >= configurations[index][6][1]:
            configurations.pop(index)
            offset += 1
    res = Data_Driven()
    circuit = Circuit(memristor_model=res, number_of_memristor=1, is_new_architecture=True, v_in=1e-3
                      , gain_resistance=0, R_L=1)

    if not os.path.isdir(f'{path}'):
        os.mkdir(f'{path}')
    for configuration in configurations:
        directory_name = f'{configuration[0]}_{configuration[1]}_{configuration[2]}_{configuration[3]}_{round(configuration[4] * 300, 1)}_{round(configuration[5] * 300, 1)}_{configuration[6][0]}_{configuration[6][1]}'
        if not os.path.isdir(f'{path}\\{directory_name}'):
            os.mkdir(f'{path}\\{directory_name}')
        if verbose:
            start = time.time()
        for index in range(nb_occurences):
            delta_r = res.r_off - res.r_on
            lrs = configuration[6][0] * delta_r + res.r_on
            hrs = configuration[6][1] * delta_r + res.r_on
            res.g = 1 / lrs

            pulsed_programming = PulsedProgramming(circuit, 2, distribution_type='linear'
                               , max_voltage=2.8, tolerance=configuration[2], is_relative_tolerance=configuration[3],variance_read=configuration[4],variance_write=configuration[5]
                               , lrs=lrs, hrs=hrs, pulse_algorithm=configuration[1], number_of_reading=configuration[0])
            pulsed_programming.simulate()

            save_everything_hdf5(f'{path}\\{directory_name}', str(index), memristor=res, pulsed_programming=pulsed_programming, circuit=circuit, verbose=False)

        if plot:
            plot_everything(None, None, pulsed_programming, f'{path}\\{directory_name}', plots=['pulsed_programming', 'amplitude'])

        if verbose:
            print(f'{len(configurations) - config_done} configurations left\tCurrent: {directory_name}\tTook: {time.time()-start}')
            config_done += 1


def parametric_test_voltage_min_max(path, configurations=None, verbose=False):
    """
    This function generates memristor simulations with different configuration.

    Parameters
    ----------
    path : string
        The path to the saving folder.

    configurations : list of list
        The configurations simulated with this disposition: [nb_memristor, is_new_architecture, v_in]

    verbose : bool
        If true, output timers in console.
    Returns
    -------

    """
    if configurations is None:
        configurations = []
        nb_memristor = [3, 4, 5, 6, 7, 8, 9, 10]
        architecture = [True]
        gain_resistance = [[0]]
        v_in = [[0.5e-3, 1e-3, 4e-3]]
        resistance_load = [1]
        lrs_hrs = [[1000, 3000]]
        for nb_memristor_ in nb_memristor:
            for lrs_hrs_ in lrs_hrs:
                for architecture_ in architecture:
                    if architecture_:
                        for v_in_ in v_in[0]:
                            for gain_resistance_ in gain_resistance[0]:
                                for resistance_load_ in resistance_load:
                                    configurations.append([nb_memristor_, architecture_, v_in_, gain_resistance_, resistance_load_, lrs_hrs_])
                    else:
                        for v_in_ in v_in[1]:
                            for gain_resistance_ in gain_resistance[1]:
                                configurations.append([nb_memristor_, architecture_, v_in_, gain_resistance_, 0, lrs_hrs_])

    config_done = 0
    if not os.path.isdir(f'{path}'):
        os.mkdir(f'{path}')
    for configuration in configurations:
        if verbose:
            start = time.time()
        res = Data_Driven()

        circuit = Circuit(memristor_model=res, number_of_memristor=configuration[0], is_new_architecture=configuration[1], v_in=configuration[2]
                          , gain_resistance=configuration[3], R_L=configuration[4])

        pulsed_programming = PulsedProgramming(circuit, 2, distribution_type='linear'
                           , max_voltage=2.8, tolerance=10, is_relative_tolerance=False,variance_read=1/300,variance_write=0,
                           lrs=configuration[5][0], hrs=configuration[5][1], pulse_algorithm='log', number_of_reading=5)
        pulsed_programming.simulate()

        memristor_simulation = MemristorSimulation(pulsed_programming)
        memristor_simulation.simulate()

        if is_square(configuration[0]):
            nb_memristor = [int(np.sqrt(configuration[0])), int(np.sqrt(configuration[0]))]
        else:
            nb_memristor = [configuration[0], 1]
        architecture = 'new' if configuration[1] else 'old'
        directory_name = f'{nb_memristor[0]}x{nb_memristor[1]}_{architecture}_{configuration[2]}_{configuration[3]}_{configuration[4]}'

        save_everything_hdf5(f'{path}', f'{directory_name}', memristor=res, pulsed_programming=pulsed_programming,
                             circuit=circuit, memristor_sim=memristor_simulation)

        if verbose:
            print(f'{len(configurations) - config_done} configurations left\tCurrent: {directory_name}\tTook: {time.time()-start}')
            config_done += 1



def load_resolution_memristor_data(path, verbose=False):
    """
    This function load a folder of saved folders for the create_resolution_memristor_plot function. The folders name need
    to follow this configuration: NxN_distribution_M_states, where N * N is the number of memristor, distribution is
    linear, half_spread or full_spread, and M is the number of states.

    Parameters
    ----------
    path : string
        The path to the folder to load.

    verbose : bool
        Output in console the timers.

    Returns
    ----------
    memristor_simulations : list of list of MemristorSimulation.MemristorSimulation
        Contains the memristor simulations that will be plot. They are divided in list by the configuration, like 2x2 half_spread.
    """
    directories = os.listdir(path)
    offset = 0
    for index in range(len(directories)):
        index -= offset
        if directories[index].endswith('.jpg'):
            directories.pop(index)
            offset += 1
    previous_conf = directories[0].split('_')
    previous_conf = str(previous_conf[0] + '_' + previous_conf[1])

    confs = []
    index_conf = 0
    for current_dir in directories:
        current_conf = current_dir.split('_')
        current_conf = str(current_conf[0] + '_' + current_conf[1])
        confs.append(current_conf)
    diff_conf = len(np.unique(confs))
    memristor_simulations = [[] for _ in range(diff_conf)]
    conf_left = len(directories)
    if verbose:
        start = time.time()
    for current_dir in directories:
        if verbose:
            print(f'{conf_left} configuration left')
        current_conf = current_dir.split('_')
        current_conf = str(current_conf[0] + '_' + current_conf[1])
        if current_conf != previous_conf:
            index_conf += 1
        memristor_simulations[index_conf].append(load_everything_hdf5(path + '\\' + current_dir, qd_simulation=0, verbose=False)[3])
        previous_conf = current_conf
        if verbose:
            conf_left -= 1
    if verbose:
        print(f'To load all data : {time.time() - start} s')

    # for i in range(len(memristor_simulations)):
    #     for j in range(len(memristor_simulations[i])):
    #         print(i, j, memristor_simulations[i][j].list_resistance)
    return memristor_simulations


def load_resolution_variance_data(path, verbose=False):
    """
    This function load a folder of saved folders of folders for the create_resolution_variance_plot function. The folders name need
    to be the variances, and than the index of the simulation.

    Parameters
    ----------
    path : string
        The path to the folder to load.

    verbose : bool
        Output in console the timers.

    Returns
    ----------
    memristor_simulations : list of list of MemristorSimulation.MemristorSimulation
        Contains the memristor simulations that will be plot. They are divided in list by the variance
    """
    directories_variance = os.listdir(path)
    offset = 0
    for index in range(len(directories_variance)):
        index -= offset
        if directories_variance[index].endswith('.jpg') or directories_variance[index].endswith('.png'):
            directories_variance.pop(index)
            offset += 1

    memristor_simulations = [[] for _ in range(len(directories_variance))]

    index_var = 0
    if verbose:
        start = time.time()
    for current_dir_var in directories_variance:
        for current_dir_sim in os.listdir(f'{path}\\{current_dir_var}'):
            if verbose:
                print(f'{current_dir_var} variance {current_dir_sim} simulation')
            path_ = f'{path}\\{current_dir_var}\\{current_dir_sim}'
            sim = load_everything_hdf5(path_, qd_simulation=0, verbose=False, light=False)[3]
            sim.resolution = np.mean(np.diff(sim.voltages))
            sim.std = np.std(np.diff(sim.voltages))
            sim.voltages.clear()
            sim.resistances.clear()
            memristor_simulations[index_var].append(sim)
        index_var += 1
        if verbose:
            print()
    if verbose:
        print(f'To load all data : {time.time() - start} s')

    return memristor_simulations


def load_pulsed_programming_efficiency(path, verbose=False):
    """
    This function load a folder of saved folders for  create_pulsed_programming_efficiency_plot().

    Parameters
    ----------
    path : string
        The path to the folder to load.

    verbose : bool
        Output in console the timers.

    Returns
    ----------
    pulsed_programmings : list of list of PulsedProgramming.PulsedProgramming
        Contains the pulsed programming simulations that will be plot. They are divided in list by the configuration.
    """
    directories = os.listdir(path)
    offset = 0
    for index in range(len(directories)):
        index -= offset
        if directories[index].endswith('.jpg'):
            directories.pop(index)
            offset += 1
    index_conf = 0
    pulsed_programmings = [[] for _ in range(len(directories))]
    if verbose:
        start = time.time()
    for current_dir in directories:
        for current_conf in os.listdir(path + '\\' + current_dir):
            if verbose:
                pass
            path_ = path + '\\' + current_dir + '\\' + current_conf
            pulsed_programmings[index_conf].append(load_everything_hdf5(path_, qd_simulation=0, memristor_sim=0, verbose=verbose)[2])
            if verbose:
                pass
        index_conf += 1
    if verbose:
        print(f'To load all data : {time.time() - start} s')

    # for i in range(len(pulsed_programmings)):
    #     for j in range(len(pulsed_programmings[i])):
    #         print(i, j, pulsed_programmings[i][j].variance_read)
    return pulsed_programmings


def load_voltage_min_max_data(path, verbose=False):
    """
    This function load a folder of saved folders for the create_voltage_min_max_plot function.

    Parameters
    ----------
    path : string
        The path to the folder to load.

    verbose : bool
        Output in console the timers.

    Returns
    ----------
    memristor_simulations : list of MemristorSimulation.MemristorSimulation
        Contains the memristor simulations that will be plot.
    """
    memristor_simulations = []
    directories = os.listdir(path)
    for directory in directories:
        if directory.endswith('.jpg'):
            continue
        memristor_simulation = load_everything_hdf5(f'{path}\\{directory}', qd_simulation=0)[3]
        memristor_simulations.append(memristor_simulation)

    return memristor_simulations
