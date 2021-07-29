import matplotlib.legend
import matplotlib.pyplot as plt
import os
import numpy as np
from .QDSimulation import QDSimulation
import math
import time
from .HelperFunction import is_square
import h5py


def plot_everything(memristor_sim, qd_sim, pulsed_programming, directory_name=None, plots=None, verbose=False):
    """
    This function plot the result plot, resistance plot, the pulsed programming plot and the stability diagram.
    It creates the folders and saves the plots accordingly.

    Parameters
    ----------
    memristor_sim : MemristorSimulation.MemristorSimulation
        The memristor simulation

    qd_sim : QDSimulation
        The quantum dot simulation

    pulsed_programming : PulsedProgramming.PulsedProgramming
        The pulsed programming

    directory_name : string
        The directory name where the plots will be save. If left by default, which is None, the plots will be show instead.

    plots : list of string
        This list contains the plots to output. By default, it contains all the plots.
            'result' : Plot the voltage as a function of resistance.
            'resist' : Plot the resistance distribution.
            'pulsed_programming' : Plot the resistance in function of the pulses and the amplitude of the pulses as a
                function of time.
            'stability' : Plot the stability diagram, the number of electron as a function of the gate voltage.
            'honeycomb' : Plot the current as a function of the gate voltage.

    verbose : bool
        Output in console the timer of the plots.

    file_output : bool
        If true, output the values of the graph in a file.

    Returns
    ----------
    """

    if plots is None:
        plots = ['result', 'resist', 'pulsed_programming', 'amplitude', 'gaussian', 'stability', 'honeycomb']

    if verbose:
        print('\n##########################\n'
              'Start plots')
        start = time.time()
    if 'result' in plots and memristor_sim is not None:
        create_result_plot(memristor_sim, directory_name)
    if verbose:
        print(f'Result plot: {time.time()-start}')
        start = time.time()

    if 'resist' in plots and memristor_sim is not None:
        create_resist_plot(pulsed_programming, directory_name)
    if verbose:
        print(f'Resist plot: {time.time()-start}')
        start = time.time()

    if 'pulsed_programming' in plots and pulsed_programming is not None:
        create_pulsed_programming_plot(pulsed_programming, directory_name)
    if verbose:
        print(f'Pulsed programming plot: {time.time()-start}')
        start = time.time()

    if 'amplitude' in plots and pulsed_programming is not None:
        create_amplitude_plot(pulsed_programming, directory_name)
    if verbose:
        print(f'Amplitude plot: {time.time()-start}')
        start = time.time()

    if 'gaussian' in plots and pulsed_programming is not None:
        create_gaussian_distribution(pulsed_programming, directory_name)
    if verbose:
        print(f'Gaussian plot: {time.time() - start}')
        start = time.time()

    if 'stability' in plots and qd_sim is not None:
        create_stability_diagram(qd_sim, directory_name)
    if verbose:
        print(f'Stability plot: {time.time()-start}')
        start = time.time()

    if 'honeycomb' in plots and qd_sim is not None:
        create_honeycomb_diagram(qd_sim, directory_name)
    if verbose:
        print(f'Honeycomb plot: {time.time()-start}')


def create_result_plot(memristor_simulation, directory_name=None):
    """
    This function creates plots from the simulation voltages and save them in Result

    Parameters
    ----------
    memristor_simulation : MemristorSimulation.MemristorSimulation
        The memristor simulation object.

    directory_name : string
        The directory name where the plots will be save. If left by default, which is None, the plots will be show instead.

    Returns
    ----------
    """
    if directory_name is not None:
        if not os.path.isdir(f'{directory_name}'):
            os.mkdir(f'{directory_name}')

    plt.clf()
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twiny()

    new_tick_locations = np.linspace(memristor_simulation.resistances[0], memristor_simulation.resistances[-1], num=6)
    ax1.scatter(memristor_simulation.resistances, memristor_simulation.voltages, label=f'{memristor_simulation.pulsed_programming.circuit.number_of_memristor} memristor with {memristor_simulation.pulsed_programming.nb_states} states')

    ax1.set_xlabel(r'Resistance R ($\Omega$)')
    ax1.set_ylabel('Voltage (V)')

    def tick_function(n):
        r = 1 / n
        return ["%.5f" % z for z in r]

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel(r"Conductance G (S)")

    filename = f'{memristor_simulation.pulsed_programming.circuit.number_of_memristor}_memristor_{memristor_simulation.pulsed_programming.nb_states}_states.jpg'
    plt.tight_layout()
    plt.tight_layout()
    if directory_name is None:
        plt.show()
    else:
        plt.savefig(fname=f'{directory_name}\\{filename}')

    plt.close('all')


def create_resist_plot(pulsed_programming, directory_name=None):
    """
    This function creates plots from the simulation resistances and save them in Resist

    Parameters
    ----------
    pulsed_programming : PulsedProgramming.PulsedProgramming
        The pulsed programming

    directory_name : string
        The directory name where the plots will be save. If left by default, which is None, the plots will be show instead.

    Returns
    ----------
    """
    if directory_name is not None:
        if not os.path.isdir(f'{directory_name}'):
            os.mkdir(f'{directory_name}')
    # list_resist_temp = list_resist
    # if simulation.is_using_conductance:
    #     for i in range(len(list_resist)):
    #         list_resist_temp[i] = [1/j for j in list_resist[i]]
    list_resist = pulsed_programming.res_states_practical
    plt.clf()
    f, ax = plt.subplots(1)
    ax.set_ylabel(r"Resistance R ($\Omega$)")
    ax.set_title('Resistance distribution used')
    ax.set_xlabel('Index of the resistance')
    for i in range(len(list_resist)):
        list_resist[i].sort()
        ax.plot(list_resist[i], 'o')
    filename = f'{pulsed_programming.circuit.number_of_memristor}_memristor_{pulsed_programming.nb_states}_states' \
               f'_{pulsed_programming.distribution_type}.jpg'
    plt.tight_layout()
    if directory_name is None:
        plt.show()
    else:
        plt.savefig(fname=f'{directory_name}\\{filename}')
    plt.close('all')


def create_pulsed_programming_plot(pulsed_programming, directory_name=None):
    """
    This function creates a plot from the pulsed programming and save them in Simulation\\PulsedProgramming.
    Resistance in function of the pulses.
        The resistance states targeted are shown over the plot.
        The number of pulses needed to obtain the current state is annotated.

    Parameters
    ----------
    pulsed_programming : PulsedProgramming.PulsedProgramming
        The pulsed programming

    directory_name : string
        The directory name where the plots will be save. If left by default, which is None, the plots will be show instead.

    Returns
    ----------
    """
    if directory_name is not None:
        if not os.path.isdir(f'{directory_name}'):
            os.mkdir(f'{directory_name}')

    ax = plt.axes()

    ax.set_xlabel('Pulse Number')
    ax.set_ylabel('Resistance \u03A9')
    y, x, action, annotation = zip(*pulsed_programming.graph_resistance)
    time_tick_locations = []

    list_set = []
    list_reset = []
    list_read = []

    annotate_point = []
    counter = 0

    counter_start = 0

    for current_index in x:
        current_y = y[current_index]
        if action[current_index] == 'read':
            list_read.append([current_y, counter])
            counter += 1
        elif action[current_index] == 'set':
            list_set.append([current_y, counter])
            counter += 1
        elif action[current_index] == 'reset':
            list_reset.append([current_y, counter])
            counter += 1
        if annotation[current_index]:
            counter_end = counter
            annotate_point.append([counter_end - counter_start, counter_end, current_y])
            counter_start = counter_end

    if len(list_reset) != 0:
        x_reset, y_reset = zip(*list_reset)
        plt.scatter(y_reset, x_reset, s=4, color='#2bff00', label='Reset')
    else:
        y_reset = [0]

    if len(list_set) != 0:
        x_set, y_set = zip(*list_set)
        plt.scatter(y_set, x_set, s=4, color='#0055ff', label='Set')
    else:
        y_set = [0]

    if len(list_read) != 0:
        x_read, y_read = zip(*list_read)
        plt.scatter(y_read, x_read, s=4, color='#ff0000', label='Read')
    else:
        y_read = [0]

    for annotation_ in annotate_point:
        last_pulse = max(max(y_read), max(y_reset), max(y_set))
        n_max = last_pulse + 0.2*last_pulse
        n_min = annotation_[1] + 0.01*last_pulse
        plt.hlines(annotation_[2], n_min, n_max, linestyles='dashed', colors='black')
        h = annotation_[2] + (pulsed_programming.hrs - pulsed_programming.lrs) / 100
        space = (5 - len(str(annotation_[0]))) * '  '
        text = f'{annotation_[0]}{space}{round(annotation_[2])} \u03A9'
        plt.annotate(text=text, xy=(last_pulse, h), color='r')
        time_tick_locations.append(annotation_[1])
    # plt.tight_layout()
    plt.legend(loc='upper left')

    def tick_function(n):
        """
        This function creates the ticks label for the time axis.
        Each pulses need 1 us to read, 200 ns to write and 200 ns of spacing between each of them.

        Parameters
        ----------
        n : iterable[int]
            An array of the number of pulses

        Returns
        ----------
        array : iterable[string]
            A list of the time label in ms.
        """
        r = [(1e-6 + 200e-9*3) * i * 1000 for i in n]
        return ["%.2f" % z for z in r]

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(time_tick_locations)
    ax2.set_xticklabels(tick_function(time_tick_locations))
    ax2.set_xlabel(r"Time (ms)")

    plt.title('Pulsed programming of a memristor')
    filename = f'{pulsed_programming.nb_states}_states_{pulsed_programming.distribution_type}_{pulsed_programming.pulse_algorithm}.jpg'
    plt.tight_layout()
    if directory_name is None:
        plt.show()
    else:
        plt.savefig(fname=f'{directory_name}\\{filename}')
    plt.close('all')


def create_amplitude_plot(pulsed_programming, directory_name=None):
    """
    This function creates a plot from the amplitude of the pulses in the pulsed programming simulation.

    Parameters
    ----------
    pulsed_programming : PulsedProgramming.PulsedProgramming
        The pulsed programming

    directory_name : string
        The directory name where the plots will be save. If left by default, which is None, the plots will be show instead.

    Returns
    ----------
    """
    if directory_name is not None:
        if not os.path.isdir(f'{directory_name}'):
            os.mkdir(f'{directory_name}')

    voltages_read = []
    voltages_set = []
    voltages_reset = []

    for i in pulsed_programming.graph_voltages:
        if i[2] == 'reset':
            voltages_reset.append([i[0], i[1]])
        elif i[2] == 'set':
            voltages_set.append([i[0], i[1]])
        elif i[2] == 'read':
            voltages_read.append([i[0], i[1]])

    plt.figure()
    if len(voltages_reset) != 0:
        x_reset, y_reset = zip(*voltages_reset)
        plt.scatter(y_reset, x_reset, s=4, color='#2bff00', label='Reset')

    if len(voltages_set) != 0:
        x_set, y_set = zip(*voltages_set)
        plt.scatter(y_set, x_set, s=4, color='#0055ff', label='Set')

    if len(voltages_read) != 0:
        x_read, y_read = zip(*voltages_read)
        plt.scatter(y_read, x_read, s=4, color='#ff0000', label='Read')

    plt.legend()
    plt.xlabel('Number of pulses')
    plt.ylabel('Voltage (V)')
    filename = f'{pulsed_programming.pulse_algorithm}_{pulsed_programming.max_voltage}_V_max.jpg'
    plt.tight_layout()
    if directory_name is None:
        plt.show()
    else:
        plt.savefig(fname=f'{directory_name}\\{filename}')
    plt.close('all')


def create_gaussian_distribution(pulsed_programming, directory_name=None):
    """
    Output the gaussian distribution of the variability_read and variability_write.

    Parameters
    ----------
    pulsed_programming : PulsedProgramming.PulsedProgramming
        The pulsed programming

    directory_name : string
        The directory name where the plots will be save. If left by default, which is None, the plots will be show instead.

    Returns
    ----------
    """
    if directory_name is not None:
        if not os.path.isdir(f'{directory_name}'):
            os.mkdir(f'{directory_name}')

    # Write
    if pulsed_programming.variance_write != 0:
        mu = 0
        sigma = pulsed_programming.variance_write
        count, bins, ignored = plt.hist(pulsed_programming.variability_write, 30, density=True)
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
                 np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
                 linewidth=2, color='r')
        plt.xlabel('Variability write (%)')
        plt.ylabel('Number of appearance')
        filename = f'variability_write_gaussian_{pulsed_programming.variance_write}_variance.jpg'
        plt.tight_layout()
        if directory_name is None:
            plt.show()
        else:
            plt.savefig(fname=f'{directory_name}\\{filename}')
        plt.close('all')


def create_stability_diagram(qd_simulation, directory_name=None):
    """
    This function creates the stability diagram from the qd_simulation and save them in Simulation\\StabilityDiagram.
    It's uses scatter with the height represented as color.

    Parameters
    ----------
    qd_simulation : QDSimulation.QDSimulation
        The quantum dot simulation

    directory_name : string
        The directory name where the plots will be save. If left by default, which is None, the plots will be show instead.

    Returns
    ----------
    """
    if directory_name is not None:
        if not os.path.isdir(f'{directory_name}'):
            os.mkdir(f'{directory_name}')
    plt.figure()
    x, y = np.meshgrid(qd_simulation.voltages, qd_simulation.voltages)

    def find_size(number_of_element):
        if number_of_element < 50:
            size = 5
        elif number_of_element < 100:
            size = 4
        elif number_of_element < 200:
            size = 3
        elif number_of_element < 500:
            size = 2
        elif number_of_element < 2000:
            size = 1
        else:
            size = 0.1
        return size

    plt.scatter(x, y, s=find_size(len(x)), c=qd_simulation.stability_diagram)
    plt.xlabel('X-axis', fontweight='bold')
    plt.ylabel('Y-axis', fontweight='bold')

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Number of electrons')
    plt.title(f'Number of voltages: {len(qd_simulation.voltages)}')
    plt.xlabel(r'$V_{g1}$ (V)')
    plt.ylabel(r'$V_{g2}$ (V)')
    filename = f'number_of_electron_diagram_{qd_simulation.N_max}_Nmax_{qd_simulation.Cm}_Cm_{qd_simulation.parameter_model}.jpg'
    plt.tight_layout()
    if directory_name is None:
        plt.show()
    else:
        plt.savefig(fname=f'{directory_name}\\{filename}')
    plt.close('all')


def create_honeycomb_diagram(qd_simulation, directory_name=None):
    """
    This function creates the honeycomb diagram from the qd_simulation and save them in Simulation\\StabilityDiagram.
    It's the differential of the stability diagram created in create_stability_diagram

    Parameters
    ----------
    qd_simulation : QDSimulation.QDSimulation
        The quantum dot simulation

    directory_name : string
        The directory name where the plots will be save. If left by default, which is None, the plots will be show instead.

    Returns
    ----------
    """
    if directory_name is not None:
        if not os.path.isdir(f'{directory_name}'):
            os.mkdir(f'{directory_name}')

    fig, ax = plt.subplots()
    x, y = np.meshgrid(qd_simulation.voltages, qd_simulation.voltages)

    dx, dy = np.gradient(qd_simulation.stability_diagram)
    color = np.sqrt((dx/2)**2 + (dy/2)**2)

    ax.quiver(x, y, dx, dy, color, scale=1000)
    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Y-axis', fontweight='bold')

    plt.title(f'Resolution: {round(np.mean(np.diff(qd_simulation.voltages)),8)} (V)   Cm: {qd_simulation.Cm}')
    plt.xlabel(r'$V_{g1}$ (V)')
    plt.ylabel(r'$V_{g2}$ (V)')
    filename = f'stability_diagram_{qd_simulation.N_max}_Nmax_{qd_simulation.Cm}_Cm_{qd_simulation.parameter_model}.jpg'
    plt.tight_layout()
    if directory_name is None:
        plt.show()
    else:
        plt.savefig(fname=f'{directory_name}\\{filename}')
    plt.close('all')


def create_staircase_plot(qd_simulation, directory_name=None):
    if directory_name is not None:
        if not os.path.isdir(f'{directory_name}'):
            os.mkdir(f'{directory_name}')

    fig, ax = plt.subplots()

    y = np.diff(qd_simulation.stability_diagram[0])
    x = qd_simulation.voltages[0:len(qd_simulation.voltages) - 1]

    scatter = {}
    for i in range(len(y)):
        if round(y[i], 3) != 0:
            scatter[i] = y[i]

    p = 0
    keys = []
    for c in list(scatter.keys()):
        if c != p + 1:
            keys.append(p)
            keys.append(c)
        p = c
    keys.pop(0)
    keys.append(list(scatter.keys())[-1])
    resolution = round(np.mean(np.diff(x)), 5)
    ax.set_title(f'{resolution} resolution (V)')
    ax.set_xlabel(f'Voltage (V)')
    ax.set_ylabel(f'Current')
    ax.plot(x, y)
    for key in keys:
        ax.scatter(x[key], scatter.get(key))
    if directory_name is None:
        plt.show()
    else:
        plt.savefig(fname=f'{directory_name}\\{filename}')

