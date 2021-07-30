import bz2
import pickle
import _pickle as cPickle
import os
import time


# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)


# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def save_everything_pickle(path, memristor_sim=None, qd_simulation=None, pulsed_programming=None, circuit=None, memristor=None, algorithm=None, verbose=False):
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

    verbose : bool
        Output in console the timers..

    Returns
    ----------
    """
    if not os.path.isdir(f'{path}'):
        os.mkdir(f'{path}')
    if memristor is not None:
        if verbose:
            print('\n##########################\n'
                  'Start saving')
            start = time.time()
        compressed_pickle(f'{path}\\memristor', memristor)
        if verbose:
            print(f'Memristor: {time.time()-start}')
            start = time.time()
    if circuit is not None:
        compressed_pickle(f'{path}\\circuit', circuit)
        if verbose:
            print(f'Circuit: {time.time()-start}')
            start = time.time()
    if pulsed_programming is not None:
        compressed_pickle(f'{path}\\pulsed_programming', pulsed_programming)
        if verbose:
            print(f'Pulsed programming: {time.time()-start}')
            start = time.time()
    if memristor_sim is not None:
        compressed_pickle(f'{path}\\memristor_sim', memristor_sim)
        if verbose:
            print(f'Memristor simulation: {time.time()-start}')
            start = time.time()
    if qd_simulation is not None:
        compressed_pickle(f'{path}\\qd_simulation', qd_simulation)
        if verbose:
            print(f'QD simulation: {time.time()-start}')
            start = time.time()
    if algorithm is not None:
        compressed_pickle(f'{path}\\algorithm', algorithm)
        if verbose:
            print(f'QD simulation: {time.time()-start}')


def load_everything_pickle(path, verbose=False):
    """
    This function load a full simulation from a directory path, considering the orignal name created by save_everything_pickle().

    Parameters
    ----------
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

    memristor = decompress_pickle(f"{path}\\memristor.pbz2") if os.path.exists(f"{path}\\memristor.pbz2") else None
    if verbose:
        print(f'Memristor loaded: {time.time()-start}')
        start = time.time()

    circuit = decompress_pickle(f"{path}\\circuit.pbz2") if os.path.exists(f"{path}\\circuit.pbz2") else None
    if verbose:
        print(f'Circuit loaded: {time.time()-start}')
        start = time.time()

    memristor_sim = decompress_pickle(f"{path}\\memristor_sim.pbz2") if os.path.exists(f"{path}\\memristor_sim.pbz2") else None
    if verbose:
        print(f'Memristor simulation loaded: {time.time()-start}')
        start = time.time()

    pulsed_programming = decompress_pickle(f"{path}\\pulsed_programming.pbz2") if os.path.exists(f"{path}\\pulsed_programming.pbz2") else None
    if verbose:
        print(f'Pulsed programming loaded: {time.time()-start}')
        start = time.time()

    qd_simulation = decompress_pickle(f"{path}\\qd_simulation.pbz2") if os.path.exists(f"{path}\\qd_simulation.pbz2") else None
    if verbose:
        print(f'Quantum dot simulation loaded: {time.time()-start}')

    algorithm = decompress_pickle(f"{path}\\algorithm.pbz2") if os.path.exists(f"{path}\\algorithm.pbz2") else None
    if verbose:
        print(f'Algorithm loaded: {time.time()-start}')

    return memristor, circuit, memristor_sim, pulsed_programming, qd_simulation, algorithm
