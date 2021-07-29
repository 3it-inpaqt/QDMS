import bz2
import pickle
import _pickle as cPickle


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
        compressed_pickle(f'{path}\\{directory_name}\\memristor', memristor)
        if verbose:
            print(f'Memristor: {time.time()-start}')
            start = time.time()
    if circuit is not None:
        compressed_pickle(f'{path}\\{directory_name}\\circuit', circuit)
        if verbose:
            print(f'Circuit: {time.time()-start}')
            start = time.time()
    if pulsed_programming is not None:
        compressed_pickle(f'{path}\\{directory_name}\\pulsed_programming', pulsed_programming)
        if verbose:
            print(f'Pulsed programming: {time.time()-start}')
            start = time.time()
    if memristor_sim is not None:
        compressed_pickle(f'{path}\\{directory_name}\\', memristor_sim)
        if verbose:
            print(f'Memristor simulation: {time.time()-start}')
            start = time.time()
    if qd_simulation is not None:
        compressed_pickle(f'{path}\\{directory_name}\\qd_simulation', qd_simulation)
        if verbose:
            print(f'QD simulation: {time.time()-start}')


def load_everything_pickle(path, memristor=False, circuit=False, pulsed_programming=False, memristor_sim=False
                           , qd_simulation=False, verbose=False):
    """
    This function load a full simulation from a directory path, considering the orignal name created by save_everything_pickle().

    Parameters
    ----------
    memristor, circuit, pulsed_programming, memristor_sim, qd_simulation : Bool
        If True, will load the object from the given path

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
    memristor_ = decompress_pickle(path + '\\memristor') if memristor else None
    if verbose:
        print(f'Memristor loaded: {time.time()-start}')
        start = time.time()

    circuit_ = decompress_pickle(path + '\\circuit') if circuit else None
    if verbose:
        print(f'Circuit loaded: {time.time()-start}')
        start = time.time()

    memristor_sim_ = decompress_pickle(path + '\\memristor_sim') if memristor_sim else None
    if verbose:
        print(f'Memristor simulation loaded: {time.time()-start}')
        start = time.time()

    pulsed_programming_ = decompress_pickle(path + '\\pulsed_programming') if pulsed_programming else None
    if verbose:
        print(f'Pulsed programming loaded: {time.time()-start}')
        start = time.time()

    qd_simulation_ = decompress_pickle(path + '\\qd_simulation') if qd_simulation else None
    if verbose:
        print(f'Quantum dot simulation loaded: {time.time()-start}')

    return memristor_, circuit_, memristor_sim_, pulsed_programming_, qd_simulation_


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
