import qdms
import numpy as np
import pickle
import os
import time


########################################################################################################################
# Creating values functions
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
    if not os.path.isdir(f'{path}'):
        os.mkdir(f'{path}')

    if configurations is None:
        conf_4_linear = [[4, 'linear', i] for i in range(2, 13)]
        # conf_4_half_spread = [[4, 'half_spread', i] for i in range(2, 13)]
        conf_4_full_spread = [[4, 'full_spread', i] for i in range(2, 13)]
        conf_9_linear = [[9, 'linear', i] for i in range(2, 10)]
        # conf_9_half_spread = [[9, 'half_spread', i] for i in range(2, 8)]
        conf_9_full_spread = [[9, 'full_spread', i] for i in range(2, 7)]
        configurations = conf_9_full_spread + conf_9_linear + conf_4_linear + conf_4_full_spread

    config_done = 0
    res = qdms.Data_Driven()
    for configuration in configurations:

        print(f'{len(configurations) - config_done} configurations left')
        circuit = qdms.Circuit(memristor_model=res, number_of_memristor=configuration[0], is_new_architecture=True, v_in=1e-3
                          , gain_resistance=0, R_L=1)

        memristor_sim = qdms.MemristorSimulation(circuit, configuration[2], verbose=True)
        memristor_sim.simulate()

        directory_name = f'{int(np.sqrt(configuration[0]))}x{int(np.sqrt(configuration[0]))}_{configuration[1]}_{configuration[2]}_states'

        if not os.path.isdir(f'{path}'):
            os.mkdir(f'{path}\\{directory_name}')
        qdms.Log.compressed_pickle(f'{path}\\{directory_name}', memristor_sim)
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

def parametric_test_quality_diagram(path, configuration, variability=None, nb_occurences=10, verbose=False):
    """
    This function generates numerous complete simulations to identify the impact of resolution and memristor variability
    on the quality of stability diagram

    Parameters
    ----------
    path : string
        The path to the saving folder.

    configuration : iterable
        The configuration simulated with this disposition: [number of memristor, number of states, tolerance, voltage resolution].

    variability : list of float
        Contains the variability used. Default: [0, 0.1, 0.5, 1, 2, 5, 10, 20] %

    nb_occurences : int
        Number of occurences per stability diagram.

    verbose : bool
        If true, output timers in console.
    Returns
    -------

    """

    if variability is None:
        variability = np.array([0, 0.1, 0.5, 1, 2, 3, 4, 5])
    variances = variability / 300
    config_done = 0
    rmss = []
    results = {}
    directory_name = f'{path}//parametric_stability_diagram_quality'
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

        for config in configuration:
            start_ = time.time()
            memristor = Data_Driven(is_variability_on=False)
            circuit = Circuit(memristor_model=memristor, number_of_memristor=config[0], is_new_architecture=True,
                              v_in=1e-3, gain_resistance=0, R_L=1)
            memristor_sim = MemristorSimulation(circuit, config[1], distribution_type='full_spread', verbose=True) #CHANGE WHEN VARIA AVAILABLE
            memristor_sim.simulate()
            voltages_target = algorithm(config[3], memristor_sim)
            pulsed_programming = PulsedProgramming(memristor_sim, verbose=True, pulse_algorithm='fabien',
                                                   tolerance=config[2], is_relative_tolerance=True,
                                                   number_of_reading=1)
            voltages_target_ = pulsed_programming.simulate(voltages_target, [[50, False], [10, False]])
            quantum_sim = QDSimulation(list(voltages_target_.keys()))
            quantum_sim.simulate()

            rms = 0
            v_shift = []
            for i in range(nb_occurences):
                if verbose:
                    print(f'Config {config_done+1}/{len(variances)*len(configuration)}: loop {i+1}/{nb_occurences}')
                voltage_target = (memristor_sim.voltages_memristor.max() - memristor_sim.voltages_memristor.min())*np.random.random()
                ind = np.where(memristor_sim.voltages_memristor>=voltage_target)[0][0]
                diagram_slice = quantum_sim.stability_diagram[:, ind]
                tol = (memristor_sim.voltages_memristor.max() - memristor_sim.voltages_memristor.min()) / (20 * int(diagram_slice.max()) - int(diagram_slice.min()))
                if 2*tol/1e-8<1000:
                    nb_points = 2*tol/1e-8
                else:
                    nb_points = 1000
                derivative_slice = np.gradient(diagram_slice)
                i_left = 0
                for j in range(int(diagram_slice.max())-int(diagram_slice.min())):
                    i_right = np.where(diagram_slice < diagram_slice.min()+j+1)[0][-1]
                    v_trans_varia = memristor_sim.voltages_memristor[i_left + np.armax(derivative_slice[i_left:i_right])]
                    i_left = i_right

                    sweep_x = np.linspace(v_trans_varia - tol, v_trans_varia + tol, nb_points)
                    sweep_y = np.linspace(memristor_sim.voltages_memristor[ind], memristor_sim.voltages_memristor[ind], 1)
                    x_mesh, y_mesh = np.meshgrid(sweep_x, sweep_y)
                    high_res_diagram = N_moy_DQD(x_mesh, y_mesh, Cg1=quantum_sim.Cg1, Cg2=quantum_sim.Cg2, Cm=quantum_sim.Cm,
                                                 CL=quantum_sim.CL, CR=quantum_sim.CR, N_min=quantum_sim.N_min,
                                                 N_max=quantum_sim.N_max, kBT=2 * quantum_sim.kB * quantum_sim.T, e=1.602e-19)
                    derivative_high_res = np.gradient(high_res_diagram[0])
                    v_trans_high_res = sweep_x[np.argmax(derivative_high_res)]
                    v_shift.append(v_trans_varia-v_trans_high_res)
                    rms += v_shift[-1]**2
            rmss.append(rms/len(v_shift))
            results[f'{config_done}'] = {'RMS': rmss, 'voltage_shift': v_shift, 'config': config, 'variance': variance}
            config_done += 1
    with open(f'{directory_name}\\data.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f'For {len(variances) * len(configuration)} simulations -> {time.time() - start} s')


########################################################################################################################
# Loading functions
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


########################################################################################################################
# Plot functions

def create_resolution_memristor_plot(memristor_simulations, directory_name, resolution_goal=100e-6):
    """
    This function creates a plot showing the impact of the number of states of multiple pulsed simulation on the resolution.
    They shouldn't have variability and be between the same LRS and HRS.

    Parameters
    ----------
    memristor_simulations : list of list of MemristorSimulation.MemristorSimulation
        Contains the memristor simulations that will be plot. They are divided in list by the configuration, like 2x2 half_spread.

    directory_name : string
        The directory name where the plots will be save

    resolution_goal : float
        The resolution goal. Will be plot as a black solid line across the graphic.

    Returns
    ----------
    """

    fig, ax = plt.subplots()
    x = []
    y = []
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan', 'gray']
    counter = 0
    biggest_nb_state = 0
    distribution_type = ''
    for current_conf in memristor_simulations:
        for current in current_conf:
            biggest_nb_state = current.pulsed_programming.nb_states if current.pulsed_programming.nb_states > biggest_nb_state else biggest_nb_state
            x.append(current.pulsed_programming.nb_states)
            y.append(np.mean(np.diff(current.voltages)))

        temp = []
        for index in range(len(x)):
            temp.append([x[index], y[index]])
        temp = list(np.unique(temp, axis=0))
        x, y = zip(*temp)
        x = list(x)
        y = list(y)

        if current_conf[0].pulsed_programming.distribution_type == 'linear':
            distribution_type = 'No spreading'
        elif current_conf[0].pulsed_programming.distribution_type == 'half_spread':
            distribution_type = 'Line state spreading'
        elif current_conf[0].pulsed_programming.distribution_type == 'full_spread':
            distribution_type = 'Full state spreading'
        number_of_memristor = int(np.sqrt(current_conf[0].pulsed_programming.circuit.number_of_memristor))
        label = f'{distribution_type} ({number_of_memristor}x{number_of_memristor})'
        marker = 'o' if number_of_memristor == 2 else 's'
        ax.plot(x, y, color=colors[counter], label=label, marker=marker, linestyle='dotted')
        counter += 1
        x.clear()
        y.clear()

    ax.plot([1, biggest_nb_state],[resolution_goal, resolution_goal], label='Goal', color='black')
    ax.set_yscale('log')
    ax.legend()
    title = f'LRS: {round(memristor_simulations[0][0].pulsed_programming.lrs/1000, 1)}k\u03A9 & HRS: {round(memristor_simulations[0][0].pulsed_programming.hrs/1000, 1)}k\u03A9'
    plt.title(title)

    plt.ylabel('Resolution (V)')
    plt.xlabel('# of states for a memristor')
    filename = f'resolution_memristor_plot.jpg'
    plt.tight_layout()
    plt.savefig(f'{directory_name}\\{filename}', dpi=1200)

    plt.close('all')
    # plt.show()


def create_resolution_memristor_cut_plot(memristor_simulations, directory_name, resolution_goal=100e-6, cut_extremes=None):
    """
    This function creates a plot showing the impact of the number of states of multiple pulsed simulation on the resolution.
    They shouldn't have variability and be between the same LRS and HRS.

    Parameters
    ----------
    memristor_simulations : list of list of MemristorSimulation.MemristorSimulation
        Contains the memristor simulations that will be plot. They are divided in list by the configuration, like 2x2 half_spread.

    directory_name : string
        The directory name where the plots will be save

    resolution_goal : float
        The resolution goal. Will be plot as a black solid line across the graphic.

    cut_extremes : float
        From the data generated by memristor_simulation.simulate(), cut the <cut_extremes> highest and lowest values.
        Between 0 (0%) and 0.5 (50%, on bottom and top, which is 100%)


    Returns
    ----------
    """
    def cut_extreme(voltages_list, value):
        """
        This function cut the <value> highest and lowest values.
        Parameters
        ----------
        voltages_list : iterable[float]
            Contains the voltages to cut.

        value : float
            Percentage to cut. Between 0 (0%) and 0.5 (50%, on bottom and top, which is 100%)

        Returns
        -------
        cut_voltages : iterable[float]
            Contains the cut voltages.
        """
        low_value = int(len(voltages_list) * value)
        high_value = int(len(voltages_list) * (1-value))
        return voltages_list[low_value:high_value]

    if cut_extremes is None:
        cut_extremes = [0, 1, 5, 10]

    fig, ax = plt.subplots()
    x = []
    y = []
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan', 'gray']
    markers = ['o', '^', 's', 'P']
    counter = 0
    biggest_nb_state = 0
    for cut_extremes_ in cut_extremes:
        for current in memristor_simulations[0]:
            nb_state = current.pulsed_programming.nb_states
            biggest_nb_state = nb_state if biggest_nb_state < nb_state else biggest_nb_state
            x.append(nb_state)
            voltages = current.voltages if cut_extremes == 0 else cut_extreme(current.voltages, cut_extremes_ / 100)
            y.append(np.mean(np.diff(voltages)))

        # temp = []
        # for index in range(len(x)):
        #     temp.append([x[index], y[index]])
        # temp = list(np.unique(temp, axis=0))
        # x, y = zip(*temp)
        # x = list(x)
        # y = list(y)

        label = f'{cut_extremes_} %'
        ax.plot(x, y, color=colors[counter], label=label, marker='o', linestyle='dotted')
        counter += 1
        x.clear()
        y.clear()

    ax.plot([2, biggest_nb_state],[resolution_goal, resolution_goal], label='Goal', color='black')
    ax.set_yscale('log')
    ax.legend()
    title = f'LRS: {round(memristor_simulations[0][0].pulsed_programming.lrs/1000, 1)}k\u03A9 & HRS: {round(memristor_simulations[0][0].pulsed_programming.hrs/1000, 1)}k\u03A9'

    plt.title(f'{title}')
    plt.ylabel('Resolution (V)')
    plt.xlabel('# of states for a memristor')
    filename = f'resolution_memristor_plot_{cut_extremes}.jpg'
    plt.tight_layout()
    plt.savefig(f'{directory_name}\\{filename}', dpi=1200)

    plt.close('all')
    # plt.show()


def create_voltage_min_max_plot_1(memristor_simulations, directory_name):
    """
    This function creates a plot showing the impact of the number of memristor on the maximum
    and minimum voltage possible.

    Parameters
    ----------
    memristor_simulations : list of MemristorSimulation.MemristorSimulation
        Contains the memristor simulations that will be plot. They are divided in list by the configuration, like 2x2 half_spread.

    directory_name : string
        The directory name where the plots will be save

    Returns
    ----------
    """
    fig, ax = plt.subplots()

    counter = 0
    for current in memristor_simulations:
        x = [int(current.pulsed_programming.circuit.number_of_memristor), int(current.pulsed_programming.circuit.number_of_memristor)]
        voltages_min = min(current.voltages)
        voltages_max = max(current.voltages)
        y = [voltages_min, voltages_max]
        ax.plot(x, y, color='black', marker='o', linestyle='dotted')

        counter += 1

    title = f'LRS: {round(memristor_simulations[0].pulsed_programming.lrs/1000, 1)}k\u03A9 & HRS: {round(memristor_simulations[0].pulsed_programming.hrs/1000, 1)}k\u03A9'
    plt.title(title)
    plt.ylabel('Voltage output (V)')
    plt.xlabel('Number of memristors')
    filename = f'voltage_min_max_plot_1.jpg'
    plt.tight_layout()
    plt.savefig(f'{directory_name}\\{filename}', dpi=1200)

    plt.close('all')
    # plt.show()


def create_voltage_min_max_plot_2(memristor_simulations, directory_name):
    """
    This function creates a plot showing the impact of lrs and hrs on the maximum and minimum voltage possible.

    Parameters
    ----------
    memristor_simulations : list of MemristorSimulation.MemristorSimulation
        Contains the memristor simulations that will be plot. They are divided in list by the configuration, like 2x2 half_spread.

    directory_name : string
        The directory name where the plots will be save

    Returns
    ----------
    """
    fig, ax = plt.subplots()

    counter = 0
    for current in memristor_simulations:
        x = [int(current.pulsed_programming.lrs), int(current.pulsed_programming.hrs)]
        voltages_min = min(current.voltages)
        voltages_max = max(current.voltages)

        v_in = current.pulsed_programming.circuit.v_in
        gain_resistance = current.pulsed_programming.circuit.gain_resistance
        if current.pulsed_programming.circuit.is_new_architecture:
            color = 'blue'
            linestyle = (0, (3, 1, 1, 1))
            label = f'New v_in={v_in} gain={gain_resistance} r_l={current.pulsed_programming.circuit.R_L}'
            y = [voltages_min, voltages_max]

        else:
            color = 'red'
            linestyle = (0, (1, 1))
            label = f'Old v_in={v_in} gain={gain_resistance}'
            y = [voltages_max, voltages_min]

        handles, labels = ax.get_legend_handles_labels()
        if label in labels:
            ax.plot(x, y, color=color, marker='o', linestyle=linestyle)
        else:
            ax.plot(x, y, color=color, label=label, marker='o', linestyle=linestyle)
        counter += 1
    ax.legend()

    # plt.title('title')
    plt.ylabel('Voltage output (V)')
    plt.xlabel('LRS and HRS (Ohm)')
    filename = f'voltage_min_max_plot_2.jpg'
    plt.tight_layout()
    plt.savefig(f'{directory_name}\\{filename}', dpi=1200)

    plt.close('all')
    # plt.show()


def create_voltage_min_max_plot_3(memristor_simulations, directory_name):
    """
    This function creates a plot showing the impact of lrs and hrs on the maximum and minimum voltage possible.

    Parameters
    ----------
    memristor_simulations : list of MemristorSimulation.MemristorSimulation
        Contains the memristor simulations that will be plot. They are divided in list by the configuration, like 2x2 half_spread.

    directory_name : string
        The directory name where the plots will be save

    Returns
    ----------
    """
    fig, ax = plt.subplots()

    counter = 0
    for current in memristor_simulations:
        x = [int(current.pulsed_programming.circuit.number_of_memristor), int(current.pulsed_programming.circuit.number_of_memristor)]
        voltages_min = min(current.voltages)
        voltages_max = max(current.voltages)
        y = [voltages_min, voltages_max]
        lns1 = ax.plot(x, y, color='blue', marker='o', label='Voltage output', linestyle='dotted')

        counter += 1
    ax.set_ylabel('Voltage output (V)')
    x.clear()
    y.clear()

    counter = 0
    ax2 = ax.twinx()
    for current in memristor_simulations:
        x.append(current.pulsed_programming.circuit.number_of_memristor)
        y.append(np.mean(np.diff(current.voltages)))

    temp = []
    for index in range(len(x)):
        temp.append([x[index], y[index]])
    temp = list(np.unique(temp, axis=0))
    x, y = zip(*temp)
    x = list(x)
    y = list(y)

    lns2 = ax2.plot(x, y, color='red', marker='^', label='Resolution', linestyle='dotted')

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    ax2.set_yscale('log')
    # plt.title('title')
    ax2.set_ylabel('Resolution (V)')
    ax.set_xlabel('# of states for a memristor')
    filename = f'resolution_memristor_plot_min_max_.jpg'
    plt.tight_layout()
    plt.savefig(f'{directory_name}\\{filename}', dpi=1200)
    plt.close('all')
    # plt.show()


def create_resolution_variance_plot(memristor_simulations, directory_name, resolution_goal=None):
    """
    This function creates a plot showing the impact of the number of states of multiple pulsed simulation on the resolution.
    They shouldn't have variability and be between the same LRS and HRS.

    Parameters
    ----------
    memristor_simulations : list of list of MemristorSimulation.MemristorSimulation
        Contains the memristor simulations that will be plot. They are divided in list by the configuration, like 2x2 half_spread.

    directory_name : string
        The directory name where the plots will be save

    Returns
    ----------
    """
    fig, ax = plt.subplots()
    variance = []
    variance_ = []
    resolution = []
    resolution_ = []
    std = []
    std_ = []
    biggest_variance = 0

    for current_var in memristor_simulations:
        for current_sim in current_var:
            variance_.append(current_sim.pulsed_programming.variance_read * 300)
            resolution_.append(current_sim.resolution)
        biggest_variance = variance_[0] if variance_[0] > biggest_variance else biggest_variance
        variance.append(variance_[0])
        # std.append(np.mean(std_[:]))
        resolution.append(np.mean(resolution_[:])*1e9)
        std.append(np.std(resolution_[:])*1e9)
        variance_.clear()
        resolution_.clear()
        std_.clear()

    for index in range(len(variance)):
        plt.errorbar(variance[index], resolution[index], std[index], label=str(round(variance[index],2))+' %', marker='o', capsize=5, color='black')
    if resolution_goal is not None:
        plt.plot([0,biggest_variance],[resolution_goal, resolution_goal], label='Goal', color='black')

    nb_memristor = memristor_simulations[0][0].pulsed_programming.circuit.number_of_memristor
    nb_states = memristor_simulations[0][0].pulsed_programming.nb_states
    distribution = memristor_simulations[0][0].pulsed_programming.distribution_type
    tolerance = memristor_simulations[0][0].pulsed_programming.tolerance
    plt.title(f'{int(np.sqrt(nb_memristor))}x{int(np.sqrt(nb_memristor))} {distribution} {nb_states} states {tolerance} tolerance {len(memristor_simulations[0])} simulations per point')
    plt.ylabel('Resolution (nV)')
    plt.xlabel('Variability (%)')
    # plt.xscale('symlog')
    filename = f'resolution_variance_plot.jpg'
    plt.tight_layout()
    plt.savefig(f'{directory_name}\\{filename}', dpi=1600)

    # plt.close('all')
    plt.show()


def create_std_variance_plot(memristor_simulations, directory_name, resolution_goal=None):
    """
    This function creates a plot showing the impact of the number of states of multiple pulsed simulation on the resolution.
    They shouldn't have variability and be between the same LRS and HRS.

    Parameters
    ----------
    memristor_simulations : list of list of MemristorSimulation.MemristorSimulation
        Contains the memristor simulations that will be plot. They are divided in list by the configuration, like 2x2 half_spread.

    directory_name : string
        The directory name where the plots will be save

    Returns
    ----------
    """
    fig, ax = plt.subplots()
    variance = []
    variance_ = []
    resolution_ = []
    std = []
    biggest_variance = 0

    for current_var in memristor_simulations:
        for current_sim in current_var:
            variance_.append(current_sim.pulsed_programming.variance_read * 300)
            resolution_.append(current_sim.resolution)
        biggest_variance = variance_[0] if variance_[0] > biggest_variance else biggest_variance
        variance.append(variance_[0])
        std.append(np.std(resolution_[:])*1e9)
        variance_.clear()

    plt.scatter(variance, std, color='black')
    if resolution_goal is not None:
        plt.plot([0,biggest_variance],[resolution_goal, resolution_goal], label='Goal', color='black')

    nb_memristor = memristor_simulations[0][0].pulsed_programming.circuit.number_of_memristor
    nb_states = memristor_simulations[0][0].pulsed_programming.nb_states
    distribution = memristor_simulations[0][0].pulsed_programming.distribution_type
    tolerance = memristor_simulations[0][0].pulsed_programming.tolerance
    plt.title(f'{int(np.sqrt(nb_memristor))}x{int(np.sqrt(nb_memristor))} {distribution} {nb_states} states {tolerance} tolerance {len(memristor_simulations[0])} simulations per point')
    plt.ylabel('Std (nV)')
    plt.xlabel('Variability (%)')
    # plt.xscale('symlog')
    filename = f'std_variance_plot.jpg'
    plt.tight_layout()
    plt.savefig(f'{directory_name}\\{filename}', dpi=1600)

    # plt.close('all')
    plt.show()


def create_pulsed_programming_efficiency_plot_1(pulsed_programmings, directory_name):
    """
    This function creates a plot showing the impact of the variance on the efficiency of the pulsed programming with
    different % of lrs and hrs.

    Parameters
    ----------
    pulsed_programmings : list of list of PulsedProgramming.PulsedProgramming

    directory_name : string
        The directory name where the plots will be save

    Returns
    ----------
    """

    if not os.path.isdir(f'{directory_name}'):
        os.mkdir(f'{directory_name}')

    ax = plt.subplot(1, 1, 1)
    x = []
    y = []
    error_bar = []
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan', 'gray']
    linestyle = [(0, (1, 10)), (0, (1, 5)), (0, (1, 1)),
                 (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]
    counter = 0
    dict_ = {}
    for current_conf in pulsed_programmings:
        for current in current_conf:
            delta_r = current.circuit.memristor_model.r_off - current.circuit.memristor_model.r_on
            lrs = (current.lrs - current.circuit.memristor_model.r_on) / delta_r
            hrs = (current.hrs - current.circuit.memristor_model.r_on) / delta_r
            if dict_.get(f'{lrs}_{hrs}') is None:
                dict_[f'{lrs}_{hrs}'] = {}
            if dict_.get(f'{lrs}_{hrs}').get(f'{current.variance_read*300}') is None:
                dict_[f'{lrs}_{hrs}'][f'{current.variance_read*300}'] = []
            dict_[f'{lrs}_{hrs}'][f'{current.variance_read * 300}'].append(current)

    y_ = []
    for key in dict_.keys():
        for current_key in dict_.get(key).keys():
            for current in dict_.get(key).get(current_key):
                second_final_read = first_final_read = 0
                flag_first = True
                for current_ in current.graph_resistance:
                    if current_[3] and flag_first:
                        first_final_read = current_[1]
                        flag_first = not flag_first
                    elif current_[3]:
                        second_final_read = current_[1]
                y_.append(second_final_read - first_final_read)
            x.append(dict_.get(key).get(current_key)[0].variance_read * 300)
            y.append(np.mean(y_[:]))
            error_bar.append(np.std(y_[:]))
            y_.clear()
        lrs, hrs = key.split('_')
        label = f'{int(round(float(lrs)*100))}-{int(round(float(hrs)*100))}%'
        # ax.errorbar(x, y, error_bar, color=colors[counter], marker='o', label=label, linestyle=linestyle[counter],capsize=5)
        ax.plot(x, y, color=colors[counter], marker='o', label=label, linestyle=linestyle[counter])
        counter += 1
        x.clear()
        y.clear()
        error_bar.clear()

    for key in dict_.keys():
        pulsed_programmings_ = dict_.get(key).get('0.0')
        algorithm = pulsed_programmings_[0].pulse_algorithm
        nb_simulations = len(pulsed_programmings_)
        textstr = 'Constants\n---------------\n'
        # textstr += f'Pulse algorithm : {algorithm}\n'
        textstr += f'Number of reading : {pulsed_programmings_[0].number_of_reading}\n'
        textstr += f'Variability write : {pulsed_programmings_[0].variance_write * 300} %\n'
        textstr += f'Tolerance : {pulsed_programmings_[0].tolerance} '
        textstr += '%\n' if pulsed_programmings_[0].is_relative_tolerance else 'Ohm\n'
        break

    plt.figtext(0.70, 0.35, textstr, fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    hl = sorted(zip(labels, handles))
    labels2, handles2 = zip(*hl)
    ax.legend(handles2, labels2, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f'{algorithm} algorithm with {nb_simulations} simulations per point')
    ax.set_ylabel('Number of pulses')
    ax.set_xlabel('Variability read (%)')
    filename = f'efficiency_1_plot.jpg'
    plt.tight_layout()
    plt.savefig(f'{directory_name}\\{filename}', dpi=1200)

    plt.close('all')
    # plt.show()


def create_pulsed_programming_efficiency_plot_2(pulsed_programmings, directory_name):
    """
    This function creates a plot showing the impact of the tolerance on the efficiency of the pulsed programming with
    varying pulse algorithm and read variability.

    Parameters
    ----------
    pulsed_programmings : list of list of PulsedProgramming.PulsedProgramming

    directory_name : string
        The directory name where the plots will be save

    Returns
    ----------
    """
    if not os.path.isdir(f'{directory_name}'):
        os.mkdir(f'{directory_name}')

    ax = plt.subplot(1, 1, 1)
    x = []
    y = []
    error_bar = []
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan', 'gray']
    linestyle = [(0, (1, 10)), (0, (1, 5)), (0, (1, 1)),
                 (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]
    counter = 0
    dict_ = {}
    for current_conf in pulsed_programmings:
        for current in current_conf:
            key = f'{current.pulse_algorithm}_{current.variance_read*300}'
            if dict_.get(f'{key}') is None:
                dict_[f'{key}'] = {}
            if dict_.get(f'{key}').get(float(current.tolerance)) is None:
                dict_[f'{key}'][float(current.tolerance)] = []
            dict_[f'{key}'][float(current.tolerance)].append(current)
    for i in dict_.keys():
        dict_[i] = dict(sorted(dict_.get(i).items()))

    # for i in dict_.keys():
    #     print(i)
    #     for j in dict_.get(i).keys():
    #         print(j, dict_.get(i).get(j))
    y_ = []
    for key in dict_.keys():
        for current_key in dict_.get(key).keys():
            for current in dict_.get(key).get(current_key):
                second_final_read = first_final_read = 0
                flag_first = True
                for current_ in current.graph_resistance:
                    if current_[3] and flag_first:
                        first_final_read = current_[1]
                        flag_first = not flag_first
                    elif current_[3]:
                        second_final_read = current_[1]
                y_.append(second_final_read - first_final_read)
            x.append(dict_.get(key).get(current_key)[0].tolerance)
            y.append(np.mean(y_[:]))
            error_bar.append(np.std(y_[:]))
            y_.clear()
        label = f'{key}'
        # ax.errorbar(x, y, error_bar, color=colors[counter], marker='o', label=label, linestyle='solid',capsize=5)
        ax.plot(x, y, color=colors[counter], marker='o', label=label, linestyle=linestyle[counter])
        counter += 1
        x.clear()
        y.clear()
        error_bar.clear()

    for key in dict_.keys():
        pulsed_programmings_ = dict_.get(key).get(1)
        nb_simulations = len(pulsed_programmings_)
        textstr = 'Constants\n---------------\n'
        break

    plt.figtext(0.75, 0.35, textstr, fontsize=8)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'{nb_simulations} simulations per point')
    ax.set_ylabel('Number of pulses')
    ax.set_xlabel('Tolerance (%)')
    filename = f'efficiency_2_plot.jpg'
    plt.tight_layout()
    plt.savefig(f'{directory_name}\\{filename}', dpi=1200)

    plt.close('all')
    # plt.show()


def create_pulsed_programming_accuracy_plot_1(pulsed_programmings, directory_name):
    """
    This function creates a plot showing the impact of the variance on the accuracy of the pulsed programming.

    Parameters
    ----------
    pulsed_programmings : list of list of PulsedProgramming.PulsedProgramming

    directory_name : string
        The directory name where the plots will be save

    Returns
    ----------
    """

    if not os.path.isdir(f'{directory_name}'):
        os.mkdir(f'{directory_name}')

    ax = plt.subplot(1, 1, 1)
    x = []
    y = []
    error_bar = []
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan', 'gray']
    linestyle = [(0, (1, 10)), (0, (1, 5)), (0, (1, 1)),
                 (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]
    counter = 0
    dict_ = {}
    for current_conf in pulsed_programmings:
        for current in current_conf:
            delta_r = current.circuit.memristor_model.r_off - current.circuit.memristor_model.r_on
            lrs = (current.lrs - current.circuit.memristor_model.r_on) / delta_r
            hrs = (current.hrs - current.circuit.memristor_model.r_on) / delta_r
            if dict_.get(f'{lrs}_{hrs}') is None:
                dict_[f'{lrs}_{hrs}'] = {}
            if dict_.get(f'{lrs}_{hrs}').get(f'{current.variance_read*300}') is None:
                dict_[f'{lrs}_{hrs}'][f'{current.variance_read*300}'] = []
            dict_[f'{lrs}_{hrs}'][f'{current.variance_read * 300}'].append(current)

    y_ = []
    for key in dict_.keys():
        for current_key in dict_.get(key).keys():
            for current in dict_.get(key).get(current_key):
                accuracy = 100 * (current.res_states_practical[0][1] - current.res_states[0][1]) / current.res_states[0][1]
                y_.append(accuracy)
            x.append(dict_.get(key).get(current_key)[0].variance_read * 300)
            y.append(np.mean(y_[:]))
            error_bar.append(np.std(y_[:]))
            y_.clear()
        lrs, hrs = key.split('_')
        label = f'{int(round(float(lrs)*100))}-{int(round(float(hrs)*100))}%'
        # ax.errorbar(x, y, error_bar, color=colors[counter], marker='o', label=label, linestyle=linestyle[counter],capsize=5)
        ax.plot(x, y, color=colors[counter], marker='o', label=label, linestyle=linestyle[counter])
        counter += 1
        x.clear()
        y.clear()
        error_bar.clear()

    for key in dict_.keys():
        pulsed_programmings_ = dict_.get(key).get('0.0')
        algorithm = pulsed_programmings_[0].pulse_algorithm
        nb_simulations = len(pulsed_programmings_)

        textstr = 'Constants\n---------------\n'
        textstr += f'Pulse algorithm : {algorithm}\n'
        textstr += f'Number of reading : {pulsed_programmings_[0].number_of_reading}\n'
        textstr += f'Variability write : {pulsed_programmings_[0].variance_write * 300} %\n'
        textstr += f'Tolerance : {pulsed_programmings_[0].tolerance} '
        textstr += '%\n' if pulsed_programmings_[0].is_relative_tolerance else 'Ohm\n'
        break

    handles, labels = ax.get_legend_handles_labels()
    hl = sorted(zip(labels, handles))
    labels2, handles2 = zip(*hl)
    ax.legend(handles2, labels2, bbox_to_anchor=(1.1, 1), loc='upper left')

    plt.figtext(0.75, 0.35, textstr, fontsize=8)

    plt.title(f'{nb_simulations} simulations per point')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Variability read (%)')
    filename = f'accuracy_1_plot.jpg'
    plt.tight_layout()
    plt.savefig(f'{directory_name}\\{filename}', dpi=1200)

    plt.close('all')
    # plt.show()


def create_pulsed_programming_accuracy_plot_2(pulsed_programmings, directory_name):
    """
    This function creates a plot showing the impact of the tolerance on the accuracy of the pulsed programming.

    Parameters
    ----------
    pulsed_programmings : list of list of PulsedProgramming.PulsedProgramming

    directory_name : string
        The directory name where the plots will be save

    Returns
    ----------
    """
    if not os.path.isdir(f'{directory_name}'):
        os.mkdir(f'{directory_name}')

    ax = plt.subplot(1, 1, 1)
    x = []
    y = []
    error_bar = []
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan', 'gray']
    linestyle = [(0, (1, 10)), (0, (1, 5)), (0, (1, 1)),
                 (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]
    counter = 0
    dict_ = {}
    for current_conf in pulsed_programmings:
        for current in current_conf:
            key = f'{current.pulse_algorithm}_{current.variance_read*300}'
            if dict_.get(f'{key}') is None:
                dict_[f'{key}'] = {}
            if dict_.get(f'{key}').get(float(current.tolerance)) is None:
                dict_[f'{key}'][float(current.tolerance)] = []
            dict_[f'{key}'][float(current.tolerance)].append(current)

    for i in dict_.keys():
        dict_[i] = dict(sorted(dict_.get(i).items()))

    # for i in dict_.keys():
    #     print(i)
    #     for j in dict_.get(i).keys():
    #         print(j, dict_.get(i).get(j))
    y_ = []
    for key in dict_.keys():
        for current_key in dict_.get(key).keys():
            for current in dict_.get(key).get(current_key):
                accuracy = 100 * (current.res_states_practical[0][1] - current.res_states[0][1]) / current.res_states[0][1]
                y_.append(accuracy)
            x.append(dict_.get(key).get(current_key)[0].tolerance)
            y.append(np.mean(y_[:]))
            error_bar.append(np.std(y_[:]))
            y_.clear()
        label = f'{key.split("_")[0]} {key.split("_")[1]}% read'
        # ax.errorbar(x, y, error_bar, color=colors[counter], marker='o', label=label, linestyle=linestyle[counter],capsize=5)
        ax.plot(x, y, color=colors[counter], marker='o', label=label, linestyle=linestyle[counter])
        counter += 1
        x.clear()
        y.clear()
        error_bar.clear()

    for key in dict_.keys():
        pulsed_programmings_ = dict_.get(key).get(1)
        nb_simulations = len(pulsed_programmings_)

        textstr = 'Constants\n---------------\n'
        # textstr += f'Pulse algorithm : {algorithm}\n'
        textstr += f'Number of reading : {pulsed_programmings_[0].number_of_reading}\n'
        textstr += f'Variability write : {pulsed_programmings_[0].variance_write * 300} %\n'
        textstr += f'Tolerance : {pulsed_programmings_[0].tolerance} '
        textstr += '%\n' if pulsed_programmings_[0].is_relative_tolerance else 'Ohm\n'
        break
    plt.figtext(0.70, 0.35, textstr, fontsize=8)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'{nb_simulations} simulations per point')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Tolerance (%)')
    filename = f'accuracy_2_plot.jpg'
    plt.tight_layout()
    plt.savefig(f'{directory_name}\\{filename}', dpi=1200)

    plt.close('all')
    # plt.show()
