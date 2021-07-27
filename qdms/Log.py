import os
import numpy as np
import time
import h5py

from .Data_Driven import Data_Driven
from .PulsedProgramming import PulsedProgramming
from .Circuit import Circuit
from .MemristorSimulation import MemristorSimulation
from .QDSimulation import QDSimulation


def save_everything_hdf5(path, directory_name, memristor_sim=None, qd_simulation=None, pulsed_programming=None, circuit=None, memristor=None, verbose=False):
    """
    This function save all the parameters in a folder name SaveData.

    Parameters
    ----------
    memristor_sim : MemristorSimulation.MemristorSimulation
        The memristor simulation

    qd_simulation : QDSimulation
        The quantum dot simulation

    pulsed_programming : PulsedProgramming.PulsedProgramming
        The pulsed programming

    circuit : Circuit.Circuit
        Circuit

    memristor : MemristorModel.Memristor.Memristor
        memristor

    path : string
        Where the the directory_name will be.

    directory_name : string
        The directory name where the data will be save

    verbose : bool
        Output in console the timers..

    Returns
    ----------
    """
    create_save_directory(path, directory_name)
    if memristor is not None:
        if verbose:
            print('\n##########################\n'
                  'Start saving')
            start = time.time()
        save_memristor_hdf5(memristor, path + '\\' + directory_name)
        if verbose:
            print(f'Memristor: {time.time()-start}')
            start = time.time()
    if circuit is not None:
        save_circuit_hdf5(circuit,  path + '\\' + directory_name)
        if verbose:
            print(f'Circuit: {time.time()-start}')
            start = time.time()
    if pulsed_programming is not None:
        save_pulsed_programming_hdf5(pulsed_programming,  path + '\\' + directory_name)
        if verbose:
            print(f'Pulsed programming: {time.time()-start}')
            start = time.time()
    if memristor_sim is not None:
        save_memristor_simulation_hdf5(memristor_sim,  path + '\\' + directory_name)
        if verbose:
            print(f'Memristor simulation: {time.time()-start}')
            start = time.time()
    if qd_simulation is not None:
        save_qd_simulation_hdf5(qd_simulation,  path + '\\' + directory_name)
        if verbose:
            print(f'QD simulation: {time.time()-start}')


def create_save_directory(path, directory_name):
    """
    This function makes the directory to save the data.

    Parameters
    ----------
    path : string
        Where the the directory_name will be.

    directory_name : string
        The directory name where the plots will be save

    Returns
    ----------
    succes : bool
        True if the directories were created successfully.
    """
    try:
        if not os.path.isdir(f'{path}'):
            os.mkdir(f'{path}')
        os.mkdir(f'{path}\\{directory_name}')
        return True
    except OSError:
        print('Error creating directories')
        return False


def save_memristor_hdf5(memristor, path):
    if not os.path.isdir(f'{path}'):
        os.mkdir(f'{path}')
    with h5py.File(f'{path}\\memristor_data.hdf5', 'w') as f:
        f.create_dataset("memristor_model", data=str(type(memristor)))
        f.create_dataset("time_series_resolution", data=memristor.time_series_resolution)
        f.create_dataset("r_off", data=memristor.r_off)
        f.create_dataset("r_on", data=memristor.r_on)
        f.create_dataset("A_p", data=memristor.A_p)
        f.create_dataset("A_n", data=memristor.A_n)
        f.create_dataset("t_p", data=memristor.t_p)
        f.create_dataset("t_n", data=memristor.t_n)
        f.create_dataset("k_p", data=memristor.k_p)
        f.create_dataset("k_n", data=memristor.k_n)
        f.create_dataset("r_n", data=memristor.r_n)
        f.create_dataset("r_p", data=memristor.r_p)
        f.create_dataset("eta", data=memristor.eta)
        f.create_dataset("a_p", data=memristor.a_p)
        f.create_dataset("a_n", data=memristor.a_n)
        f.create_dataset("b_p", data=memristor.b_p)
        f.create_dataset("b_n", data=memristor.b_n)
        f.create_dataset("g", data=memristor.g)
        f.create_dataset("is_variability_on", data=memristor.is_variability_on)


def save_circuit_hdf5(circuit, path):
    with h5py.File(f'{path}\\circuit_data.hdf5', 'w') as f:
        f.create_dataset("number_of_memristor", data=circuit.number_of_memristor)
        f.create_dataset("gain_resistance", data=circuit.gain_resistance)
        f.create_dataset("v_in", data=circuit.v_in)
        f.create_dataset("R_L", data=circuit.R_L)
        f.create_dataset("is_new_architecture", data=circuit.is_new_architecture)


def save_pulsed_programming_hdf5(pulsed_programming, path):
    with h5py.File(f'{path}\\pulsed_programming_data.hdf5', 'w') as f:
        f.create_dataset("pulse_algorithm", data=pulsed_programming.pulse_algorithm)
        f.create_dataset("max_voltage", data=pulsed_programming.max_voltage)
        f.create_dataset("tolerance", data=pulsed_programming.tolerance)
        f.create_dataset("index_variability", data=pulsed_programming.index_variability)
        f.create_dataset("variance_write", data=pulsed_programming.variance_write)
        f.create_dataset("variability_write", data=pulsed_programming.variability_write)
        f.create_dataset("number_of_reading", data=pulsed_programming.number_of_reading)
        graph_resistance_1, graph_resistance_2, graph_resistance_3, graph_resistance_4 = zip(*pulsed_programming.graph_resistance)
        graph_resistance_3 = [i.encode('utf8') for i in graph_resistance_3]
        f.create_dataset("graph_resistance_1", data=graph_resistance_1)
        f.create_dataset("graph_resistance_2", data=graph_resistance_2)
        f.create_dataset("graph_resistance_3", data=graph_resistance_3)
        f.create_dataset("graph_resistance_4", data=graph_resistance_4)
        graph_voltages_1, graph_voltages_2, graph_voltages_3 = zip(*pulsed_programming.graph_voltages)
        graph_voltages_3 = [i.encode('utf8') for i in graph_voltages_3]
        f.create_dataset("graph_voltages_1", data=graph_voltages_1)
        f.create_dataset("graph_voltages_2", data=graph_voltages_2)
        f.create_dataset("graph_voltages_3", data=graph_voltages_3)
        f.create_dataset("max_pulse", data=pulsed_programming.max_pulse)
        f.create_dataset("is_relative_tolerance", data=pulsed_programming.is_relative_tolerance)


def save_memristor_simulation_hdf5(memristor_sim, path):
    filename = 'memristor_sim_data'
    with h5py.File(f'{path}\\{filename}.hdf5', 'w') as f:
        f.create_dataset("is_using_conductance", data=memristor_sim.is_using_conductance)
        f.create_dataset("nb_states", data=memristor_sim.nb_states)
        f.create_dataset("distribution_type", data=memristor_sim.distribution_type)
        f.create_dataset("voltages_memristor", data=memristor_sim.voltages_memristor)
        f.create_dataset("verbose", data=memristor_sim.verbose)
        f.create_dataset("list_resistance", data=memristor_sim.list_resistance)
        f.create_dataset("timers", data=memristor_sim.timers)


def save_qd_simulation_hdf5(memristor_sim, path):
    with h5py.File(f'{path}\\qd_simulation_data.hdf5', 'w') as f:
        stability_diagram = np.array(memristor_sim.stability_diagram)
        f.create_dataset("stability_diagram", data=stability_diagram.astype(np.float64))
        f.create_dataset("voltages", data=memristor_sim.voltages)
        f.create_dataset("Cg1", data=memristor_sim.Cg1)
        f.create_dataset("Cg2", data=memristor_sim.Cg2)
        f.create_dataset("CL", data=memristor_sim.CL)
        f.create_dataset("CR", data=memristor_sim.CR)
        f.create_dataset("parameter_model", data=memristor_sim.parameter_model)
        f.create_dataset("T", data=memristor_sim.T)
        f.create_dataset("Cm", data=memristor_sim.Cm)
        f.create_dataset("kB", data=memristor_sim.kB)
        f.create_dataset("N_min", data=memristor_sim.N_min)
        f.create_dataset("N_max", data=memristor_sim.N_max)
        f.create_dataset("n_dots", data=memristor_sim.n_dots)
        f.create_dataset("verbose", data=memristor_sim.verbose)


def load_everything_hdf5(path, memristor=None, circuit=None, pulsed_programming=None, memristor_sim=None,
                    qd_simulation=None, verbose=False):
    """
    This function load a full simulation from a directory path, considering the orignal name created by save_everything_hdf5().
    If memristor_sim, qd_simulation, pulsed_programming, circuit, memristor are not None, than the loaded data will be
    override by the object. If a number, than it won't load this item and the one linked to it and will return None.

    Parameters
    ----------
    memristor_sim : MemristorSimulation.MemristorSimulation
        The memristor simulation. By default None. If not None, override the loaded data with the object passed.
        If int, it won't load this object.

    qd_simulation : QDSimulation
        The quantum dot simulation. By default None. If not None, override the loaded data with the object passed.
        If int, it won't load this object.

    pulsed_programming : PulsedProgramming.PulsedProgramming
        The pulsed programming. By default None. If not None, override the loaded data with the object passed.
        If int, it won't load this object and memristor_sim.

    circuit : Circuit.Circuit
        Circuit. By default None. If not None, override the loaded data with the object passed.
        If int, it won't load this object, pulsed_programming and memristor_sim.

    memristor : MemristorModel.Memristor.Memristor
        memristor. By default None. If not None, override the loaded data with the object passed.
        If int, it won't load this object, circuit, pulsed_programming and memristor_sim.

    path : string
        The directory path from where the data is.

    verbose : bool
        Output in console the timers.

    Returns
    ----------
    memristor : MemristorModel.Memristor.Memristor
        memristor.

    circuit : Circuit.Circuit
        Circuit.

    memristor_sim : MemristorSimulation.MemristorSimulation
        The memristor simulation.

    pulsed_programming : PulsedProgramming.PulsedProgramming
        The pulsed programming.

    qd_simulation : QDSimulation
        The quantum dot simulation.
    """
    if verbose:
        print('\n##########################\n'
              'Start loading')
        start = time.time()
    if isinstance(memristor, int):
        memristor = None
    elif memristor is None:
        memristor = load_memristor_hdf5(path + '\\memristor_data.hdf5')
    if verbose:
        print(f'Memristor loaded: {time.time()-start}')
        start = time.time()

    if isinstance(circuit, int):
        circuit = None
    elif circuit is None and memristor is not None:
        circuit = load_circuit_hdf5(path + '\\circuit_data.hdf5', memristor)
    if verbose:
        print(f'Circuit loaded: {time.time()-start}')
        start = time.time()

    if isinstance(memristor_sim, int):
        memristor_sim = None
    elif memristor_sim is None and circuit is not None:
        memristor_sim = load_memristor_simulation_hdf5(path + f'\\memristor_sim_data.hdf5', circuit)
    if verbose:
        print(f'Memristor simulation loaded: {time.time()-start}')
        start = time.time()

    if isinstance(pulsed_programming, int):
        pulsed_programming = None
    elif pulsed_programming is None and memristor_sim is not None:
        pulsed_programming = load_pulsed_programming_hdf5(path + '\\pulsed_programming_data.hdf5', memristor_sim)
    if verbose:
        print(f'Pulsed programming loaded: {time.time()-start}')
        start = time.time()

    if isinstance(qd_simulation, int):
        qd_simulation = None
    elif qd_simulation is None:
        qd_simulation = load_qd_simulation_hdf5(path + '\\qd_simulation_data.hdf5')
    if verbose:
        print(f'Quantum dot simulation loaded: {time.time()-start}')

    return memristor, circuit, memristor_sim, pulsed_programming, qd_simulation


def load_memristor_hdf5(path):
    with h5py.File(f'{path}', 'r') as file:
        memristor_model = str(np.array(file.get('memristor_model')))
        time_series_resolution = np.array(file.get('time_series_resolution'))
        r_off = np.array(file.get('r_off'))
        r_on = np.array(file.get('r_on'))
        A_p = np.array(file.get('A_p'))
        A_n = np.array(file.get('A_n'))
        t_p = np.array(file.get('t_p'))
        t_n = np.array(file.get('t_n'))
        k_p = np.array(file.get('k_p'))
        k_n = np.array(file.get('k_n'))
        r_n = list(np.array(file.get('r_n')))
        r_p = list(np.array(file.get('r_p')))
        eta = np.array(file.get('eta'))
        a_p = np.array(file.get('a_p'))
        a_n = np.array(file.get('a_n'))
        b_p = np.array(file.get('b_p'))
        b_n = np.array(file.get('b_n'))
        g = np.array(file.get('g'))
        is_variability_on = np.array(file.get('is_variability_on'))

    memristor = None
    if memristor_model == str(b"<class 'qdms.Data_Driven.Data_Driven'>"):
        memristor = Data_Driven()
        memristor.time_series_resolution = time_series_resolution
        memristor.r_off = r_off
        memristor.r_on = r_on
        memristor.A_p = A_p
        memristor.A_n = A_n
        memristor.t_p = t_p
        memristor.t_n = t_n
        memristor.k_p = k_p
        memristor.k_n = k_n
        memristor.r_p = r_p
        memristor.r_n = r_n
        memristor.eta = eta
        memristor.a_p = a_p
        memristor.a_n = a_n
        memristor.b_p = b_p
        memristor.b_n = b_n
        memristor.g = g
        memristor.is_variability_on = is_variability_on

    else:
        print(f'Log.load_memristor: memristor model <{memristor_model}> unknown')
    return memristor


def load_circuit_hdf5(path, memristor):
    """
    This function load a file created by save_circuit_hdf5() and return the object.

    Parameters
    ----------
    path : string
        The path to the file to load.

    memristor : MemristorModel.Data_Driven.Data_Driven
        The memristor object composing the circuit.

    Returns
    ----------
    circuit : Circuit.Circuit
        The circuit object.
    """
    with h5py.File(f'{path}', 'r') as file:
        number_of_memristor = np.array(file.get('number_of_memristor'))
        gain_resistance = np.array(file.get('gain_resistance'))
        v_in = np.array(file.get('v_in'))
        R_L = np.array(file.get('R_L'))
        is_new_architecture = np.array(file.get('is_new_architecture'))

    circuit = Circuit(memristor_model=memristor, number_of_memristor=number_of_memristor, is_new_architecture=is_new_architecture
                      , v_in=v_in, gain_resistance=gain_resistance, R_L=R_L)

    return circuit


def load_pulsed_programming_hdf5(path, memristor_simulation):
    """
    This function load a file created by save_pulsed_programming_hdf5() and return the object.

    Parameters
    ----------
    path : string
        The path to the file to load.

    memristor_simulation : MemristorSimulation
        The memristor simulation object composing the pulsed programming.

    Returns
    ----------
    pulsed_programming : PulsedProgramming.PulsedProgramming
        The pulsed_programming object.
    """
    with h5py.File(f'{path}', 'r') as file:
        pulse_algorithm = str(np.array(file.get('pulse_algorithm'))).lstrip("b\'").rstrip("\'")
        max_voltage = np.array(file.get('max_voltage'))
        tolerance = np.array(file.get('tolerance'))
        index_variability = np.array(file.get('index_variability'))
        variance_write = np.array(file.get('variance_write'))
        variability_write = np.array(file.get('variability_write'))
        number_of_reading = np.array(file.get('number_of_reading'))
        graph_resistance_1 = np.array(file.get('graph_resistance_1'))
        graph_resistance_2 = np.array(file.get('graph_resistance_2'))
        graph_resistance_3 = np.array(file.get('graph_resistance_3'))
        graph_resistance_4 = np.array(file.get('graph_resistance_4'))
        graph_resistance = list(zip(graph_resistance_1, graph_resistance_2, [str(a).lstrip("b\'").rstrip("\'") for a in graph_resistance_3], graph_resistance_4))
        graph_resistance = [list(a) for a in graph_resistance]
        graph_voltages_1 = np.array(file.get('graph_voltages_1'))
        graph_voltages_2 = np.array(file.get('graph_voltages_2'))
        graph_voltages_3 = np.array(file.get('graph_voltages_3'))
        graph_voltages = list(zip(graph_voltages_1, graph_voltages_2, [str(a).lstrip("b\'").rstrip("\'") for a in graph_voltages_3]))
        graph_voltages = [list(a) for a in graph_voltages]
        max_pulse = np.array(file.get('max_pulse'))
        is_relative_tolerance = np.array(file.get('is_relative_tolerance'))

    pulsed_programming = PulsedProgramming(memristor_simulation)
    pulsed_programming.pulse_algorithm = pulse_algorithm
    pulsed_programming.max_voltage = max_voltage
    pulsed_programming.tolerance = tolerance
    pulsed_programming.index_variability = index_variability
    pulsed_programming.variance_write = variance_write
    pulsed_programming.variability_write = variability_write
    pulsed_programming.number_of_reading = number_of_reading
    pulsed_programming.graph_resistance = graph_resistance
    pulsed_programming.graph_voltages = graph_voltages
    pulsed_programming.max_pulse = max_pulse
    pulsed_programming.is_relative_tolerance = is_relative_tolerance

    return pulsed_programming


def load_memristor_simulation_hdf5(path, circuit):
    """
    This function load a file created by save_memristor_simulation_hdf5() and return the object.

    Parameters
    ----------
    path : string
        The path to the file to load.

    circuit : Circuit
        The circuit object.

    Returns
    ----------
    memristor_simulation : MemristorSimulation.MemristorSimulation
        The memristor_simulation object.
    """
    with h5py.File(f'{path}', 'r') as file:
        is_using_conductance = np.array(file.get('is_using_conductance'))
        verbose = np.array(file.get('verbose'))
        nb_states = np.array(file.get('nb_states'))
        distribution_type = np.array(file.get('distribution_type'))
        timer = time.time()
        voltages_memristor = np.array(file.get('voltages_memristor'))
        print(time.time() - timer)
        print()
        list_resistance = [list(a) for a in np.array(file.get('list_resistance'))]
        timers = list(np.array(file.get('timers')))

    memristor_simulation = MemristorSimulation(circuit, nb_states)
    memristor_simulation.is_using_conductance = is_using_conductance
    memristor_simulation.distribution_type = distribution_type
    memristor_simulation.voltages_memristor = voltages_memristor
    memristor_simulation.list_resistance = list_resistance
    memristor_simulation.verbose = verbose
    memristor_simulation.timers = timers

    return memristor_simulation


def load_qd_simulation_hdf5(path):
    """
    This function load a file created by save_qd_simulation_hdf5() and return the object.

    Parameters
    ----------
    path : string
        The path to the file to load.

    Returns
    ----------
    qd_simulation : QDSimulation.QDSimulation
        The qd_simulation object.
    """
    with h5py.File(f'{path}', 'r') as file:
        stability_diagram = list(np.array(file.get('stability_diagram')))
        voltages = np.array(file.get('voltages'))
        Cg1 = np.array(file.get('Cg1'))
        Cg2 = np.array(file.get('Cg2'))
        CL = np.array(file.get('CL'))
        CR = np.array(file.get('CR'))
        parameter_model = str(np.array(file.get('parameter_model'))).lstrip("b\'").rstrip("\'")
        T = np.array(file.get('T'))
        Cm = np.array(file.get('Cm'))
        kB = np.array(file.get('kB'))
        N_min = np.array(file.get('N_min'))
        N_max = np.array(file.get('N_max'))
        n_dots = np.array(file.get('n_dots'))
        verbose = np.array(file.get('verbose'))

    qd_simulation = QDSimulation(voltages)
    qd_simulation.stability_diagram = stability_diagram
    qd_simulation.Cg1 = Cg1
    qd_simulation.Cg2 = Cg2
    qd_simulation.CL = CL
    qd_simulation.CR = CR
    qd_simulation.parameter_model = parameter_model
    qd_simulation.T = T
    qd_simulation.Cm = Cm
    qd_simulation.kB = kB
    qd_simulation.N_min = N_min
    qd_simulation.N_max = N_max
    qd_simulation.n_dots = n_dots
    qd_simulation.verbose = verbose

    return qd_simulation
