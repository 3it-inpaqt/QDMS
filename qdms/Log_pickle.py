import os
import numpy as np
import time
import bz2
import pickle
import _pickle as cPickle

from .Data_Driven import Data_Driven
from .PulsedProgramming import PulsedProgramming
from .Circuit import Circuit
from .MemristorSimulation import MemristorSimulation
from .QDSimulation import QDSimulation


# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)


# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def save_everything_pickle(path, directory_name, memristor_sim=None, qd_simulation=None, pulsed_programming=None, circuit=None, memristor=None, verbose=False):
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
        save_memristor_pickle(memristor, path + '\\' + directory_name)
        if verbose:
            print(f'Memristor: {time.time()-start}')
            start = time.time()
    if circuit is not None:
        save_circuit_pickle(circuit,  path + '\\' + directory_name)
        if verbose:
            print(f'Circuit: {time.time()-start}')
            start = time.time()
    if pulsed_programming is not None:
        save_pulsed_programming_pickle(pulsed_programming,  path + '\\' + directory_name)
        if verbose:
            print(f'Pulsed programming: {time.time()-start}')
            start = time.time()
    if memristor_sim is not None:
        save_memristor_simulation_pickle(memristor_sim,  path + '\\' + directory_name)
        if verbose:
            print(f'Memristor simulation: {time.time()-start}')
            start = time.time()
    if qd_simulation is not None:
        save_qd_simulation_pickle(qd_simulation,  path + '\\' + directory_name)
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


def save_memristor_pickle(memristor, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    compressed_pickle(f'{path}\\memristor_model', type(memristor))
    compressed_pickle(f'{path}\\time_series_resolution', memristor.time_series_resolution)
    compressed_pickle(f'{path}\\r_off', memristor.r_off)
    compressed_pickle(f'{path}\\r_on', memristor.r_on)
    compressed_pickle(f'{path}\\A_p_', memristor.A_p)
    compressed_pickle(f'{path}\\A_n_', memristor.A_n)
    compressed_pickle(f'{path}\\t_p', memristor.t_p)
    compressed_pickle(f'{path}\\t_n', memristor.t_n)
    compressed_pickle(f'{path}\\k_p', memristor.k_p)
    compressed_pickle(f'{path}\\k_n', memristor.k_n)
    compressed_pickle(f'{path}\\r_p', memristor.r_p)
    compressed_pickle(f'{path}\\r_n', memristor.r_n)
    compressed_pickle(f'{path}\\eta', memristor.eta)
    compressed_pickle(f'{path}\\a_p', memristor.a_p)
    compressed_pickle(f'{path}\\a_n', memristor.a_n)
    compressed_pickle(f'{path}\\b_p', memristor.b_p)
    compressed_pickle(f'{path}\\b_n', memristor.b_n)
    compressed_pickle(f'{path}\\g', memristor.g)
    compressed_pickle(f'{path}\\is_variability_on', memristor.is_variability_on)


def save_circuit_pickle(circuit, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    compressed_pickle(f'{path}\\number_of_memristor', circuit.number_of_memristor)
    compressed_pickle(f'{path}\\gain_resistance', circuit.gain_resistance)
    compressed_pickle(f'{path}\\v_in', circuit.v_in)
    compressed_pickle(f'{path}\\R_L', circuit.R_L)
    compressed_pickle(f'{path}\\is_new_architecture', circuit.is_new_architecture)


def save_pulsed_programming_pickle(pulsed_programming, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    compressed_pickle(f'{path}\\pulse_algorithm', pulsed_programming.pulse_algorithm)
    compressed_pickle(f'{path}\\max_voltage', pulsed_programming.max_voltage)
    compressed_pickle(f'{path}\\tolerance', pulsed_programming.tolerance)
    compressed_pickle(f'{path}\\index_variability', pulsed_programming.index_variability)
    compressed_pickle(f'{path}\\variance_write', pulsed_programming.variance_write)
    compressed_pickle(f'{path}\\variability_write', pulsed_programming.variability_write)
    compressed_pickle(f'{path}\\number_of_reading', pulsed_programming.number_of_reading)
    compressed_pickle(f'{path}\\graph_resistance', pulsed_programming.graph_resistance)
    compressed_pickle(f'{path}\\graph_voltages', pulsed_programming.graph_voltages)
    compressed_pickle(f'{path}\\max_pulse', pulsed_programming.max_pulse)
    compressed_pickle(f'{path}\\is_relative_tolerance', pulsed_programming.is_relative_tolerance)


def save_memristor_simulation_pickle(memristor_sim, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    compressed_pickle(f'{path}\\is_using_conductance', memristor_sim.is_using_conductance)
    compressed_pickle(f'{path}\\nb_states', memristor_sim.nb_states)
    compressed_pickle(f'{path}\\distribution_type', memristor_sim.distribution_type)
    compressed_pickle(f'{path}\\voltages', memristor_sim.voltages)
    compressed_pickle(f'{path}\\memristor', memristor_sim.memristor)
    compressed_pickle(f'{path}\\verbose', memristor_sim.verbose)
    compressed_pickle(f'{path}\\list_resistance', memristor_sim.list_resistance)


def save_qd_simulation_pickle(memristor_sim, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    compressed_pickle(f'{path}\\stability_diagram', memristor_sim.stability_diagram)
    compressed_pickle(f'{path}\\voltages', memristor_sim.voltages)
    compressed_pickle(f'{path}\\Cg1', memristor_sim.Cg1)
    compressed_pickle(f'{path}\\Cg2', memristor_sim.Cg2)
    compressed_pickle(f'{path}\\CL', memristor_sim.CL)
    compressed_pickle(f'{path}\\CR', memristor_sim.CR)
    compressed_pickle(f'{path}\\parameter_model', memristor_sim.parameter_model)
    compressed_pickle(f'{path}\\T', memristor_sim.T)
    compressed_pickle(f'{path}\\Cm', memristor_sim.Cm)
    compressed_pickle(f'{path}\\kB', memristor_sim.kB)
    compressed_pickle(f'{path}\\N_min', memristor_sim.N_min)
    compressed_pickle(f'{path}\\N_max', memristor_sim.N_max)
    compressed_pickle(f'{path}\\n_dots', memristor_sim.n_dots)
    compressed_pickle(f'{path}\\verbose', memristor_sim.verbose)


def load_everything_pickle(path, memristor=None, circuit=None, pulsed_programming=None, memristor_sim=None,
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
        memristor = load_memristor_pickle(path + '\\memristor_data')
    if verbose:
        print(f'Memristor loaded: {time.time()-start}')
        start = time.time()

    if isinstance(circuit, int):
        circuit = None
    elif circuit is None and memristor is not None:
        circuit = load_circuit_pickle(path + '\\circuit_data', memristor)
    if verbose:
        print(f'Circuit loaded: {time.time()-start}')
        start = time.time()

    if isinstance(memristor_sim, int):
        memristor_sim = None
    elif memristor_sim is None and circuit is not None:
        memristor_sim = load_memristor_simulation_pickle(path + f'\\memristor_sim_data', circuit)
    if verbose:
        print(f'Memristor simulation loaded: {time.time()-start}')
        start = time.time()

    if isinstance(pulsed_programming, int):
        pulsed_programming = None
    elif pulsed_programming is None and memristor_sim is not None:
        pulsed_programming = load_pulsed_programming_pickle(path + '\\pulsed_programming_data', memristor_sim)
    if verbose:
        print(f'Pulsed programming loaded: {time.time()-start}')
        start = time.time()

    if isinstance(qd_simulation, int):
        qd_simulation = None
    elif qd_simulation is None:
        qd_simulation = load_qd_simulation_pickle(path + '\\qd_simulation_data')
    if verbose:
        print(f'Quantum dot simulation loaded: {time.time()-start}')

    return memristor, circuit, memristor_sim, pulsed_programming, qd_simulation


def load_memristor_pickle(path):
    memristor_model = decompress_pickle(f'{path}\\memristor_model.pbz2')
    time_series_resolution = decompress_pickle(f'{path}\\time_series_resolution.pbz2')
    r_off = decompress_pickle(f'{path}\\r_off.pbz2')
    r_on = decompress_pickle(f'{path}\\r_on.pbz2')
    A_p = decompress_pickle(f'{path}\\A_p_.pbz2')
    A_n = decompress_pickle(f'{path}\\A_n_.pbz2')
    t_p = decompress_pickle(f'{path}\\t_p.pbz2')
    t_n = decompress_pickle(f'{path}\\t_n.pbz2')
    k_p = decompress_pickle(f'{path}\\k_p.pbz2')
    k_n = decompress_pickle(f'{path}\\k_n.pbz2')
    r_n = decompress_pickle(f'{path}\\r_n.pbz2')
    r_p = decompress_pickle(f'{path}\\r_p.pbz2')
    eta = decompress_pickle(f'{path}\\eta.pbz2')
    a_p = decompress_pickle(f'{path}\\a_p.pbz2')
    a_n = decompress_pickle(f'{path}\\a_n.pbz2')
    b_p = decompress_pickle(f'{path}\\b_p.pbz2')
    b_n = decompress_pickle(f'{path}\\b_n.pbz2')
    g = decompress_pickle(f'{path}\\g.pbz2')
    is_variability_on = decompress_pickle(f'{path}\\is_variability_on.pbz2')

    if str(memristor_model) == "<class 'qdms.Data_Driven.Data_Driven'>":
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
        raise Exception(f'Log.load_memristor: memristor model <{memristor_model}> unknown')
    return memristor


def load_circuit_pickle(path, memristor):
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
    number_of_memristor = decompress_pickle(f'{path}\\number_of_memristor.pbz2')
    gain_resistance = decompress_pickle(f'{path}\\gain_resistance.pbz2')
    v_in = decompress_pickle(f'{path}\\v_in.pbz2')
    R_L = decompress_pickle(f'{path}\\R_L.pbz2')
    is_new_architecture = decompress_pickle(f'{path}\\is_new_architecture.pbz2')

    circuit = Circuit(memristor_model=memristor, number_of_memristor=number_of_memristor, is_new_architecture=is_new_architecture
                      , v_in=v_in, gain_resistance=gain_resistance, R_L=R_L)

    return circuit


def load_pulsed_programming_pickle(path, memristor_simulation):
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
    pulse_algorithm = decompress_pickle(f'{path}\\pulse_algorithm.pbz2')
    max_voltage = decompress_pickle(f'{path}\\max_voltage.pbz2')
    tolerance = decompress_pickle(f'{path}\\tolerance.pbz2')
    index_variability = decompress_pickle(f'{path}\\index_variability.pbz2')
    variance_write = decompress_pickle(f'{path}\\variance_write.pbz2')
    variability_write = decompress_pickle(f'{path}\\variability_write.pbz2')
    number_of_reading = decompress_pickle(f'{path}\\number_of_reading.pbz2')
    graph_resistance = decompress_pickle(f'{path}\\graph_resistance.pbz2')
    graph_voltages = decompress_pickle(f'{path}\\graph_voltages.pbz2')
    max_pulse = decompress_pickle(f'{path}\\max_pulse.pbz2')
    is_relative_tolerance = decompress_pickle(f'{path}\\is_relative_tolerance.pbz2')

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


def load_memristor_simulation_pickle(path, circuit):
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
    is_using_conductance = decompress_pickle(f'{path}\\is_using_conductance.pbz2')
    verbose = decompress_pickle(f'{path}\\verbose.pbz2')
    nb_states = decompress_pickle(f'{path}\\nb_states.pbz2')
    voltages = decompress_pickle(f'{path}\\voltages.pbz2')
    memristor = decompress_pickle(f'{path}\\memristor.pbz2')
    list_resistance = decompress_pickle(f'{path}\\list_resistance.pbz2')
    distribution_type = decompress_pickle(f'{path}\\distribution_type.pbz2')

    memristor_simulation = MemristorSimulation(circuit, nb_states)
    memristor_simulation.is_using_conductance = is_using_conductance
    memristor_simulation.distribution_type = distribution_type
    memristor_simulation.memristor = memristor
    memristor_simulation.voltages = voltages
    memristor_simulation.list_resistance = list_resistance
    memristor_simulation.verbose = verbose

    return memristor_simulation


def load_qd_simulation_pickle(path):
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
    stability_diagram = decompress_pickle(f'{path}\\stability_diagram.pbz2')
    voltages = decompress_pickle(f'{path}\\voltages.pbz2')
    Cg1 = decompress_pickle(f'{path}\\Cg1.pbz2')
    Cg2 = decompress_pickle(f'{path}\\Cg2.pbz2')
    CL = decompress_pickle(f'{path}\\CL.pbz2')
    CR = decompress_pickle(f'{path}\\CR.pbz2')
    parameter_model = decompress_pickle(f'{path}\\parameter_model.pbz2')
    T = decompress_pickle(f'{path}\\T.pbz2')
    Cm = decompress_pickle(f'{path}\\Cm.pbz2')
    kB = decompress_pickle(f'{path}\\kB.pbz2')
    N_min = decompress_pickle(f'{path}\\N_min.pbz2')
    N_max = decompress_pickle(f'{path}\\N_max.pbz2')
    n_dots = decompress_pickle(f'{path}\\n_dots.pbz2')
    verbose = decompress_pickle(f'{path}\\verbose.pbz2')

    qd_simulation = QDSimulation(voltages)
    qd_simulation.stability_diagram = stability_diagram
    qd_simulation.voltages = voltages
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
