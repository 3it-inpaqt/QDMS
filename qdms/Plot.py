import matplotlib.legend
import matplotlib.pyplot as plt
import os
import numpy as np
from .QDSimulation import QDSimulation
import math
import time
from .HelperFunction import is_square
import h5py


def plot_everything(memristor_sim, qd_sim, pulsed_programming, directory_name, plots=None, file_output=False, verbose=False):
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
        The directory name where the plots will be save

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
    if not os.path.isdir(f'{directory_name}'):
        os.mkdir(f'{directory_name}')
    if verbose:
        print('\n##########################\n'
              'Start plots')
        start = time.time()
    if 'result' in plots and memristor_sim is not None:
        create_result_plot(memristor_sim, directory_name, file_output=file_output)
    if verbose:
        print(f'Result plot: {time.time()-start}')
        start = time.time()

    if 'resist' in plots and memristor_sim is not None:
        create_resist_plot(pulsed_programming, directory_name, file_output=file_output)
    if verbose:
        print(f'Resist plot: {time.time()-start}')
        start = time.time()

    if 'pulsed_programming' in plots and pulsed_programming is not None:
        create_pulsed_programming_plot(pulsed_programming, directory_name, file_output=file_output)
    if verbose:
        print(f'Pulsed programming plot: {time.time()-start}')
        start = time.time()

    if 'amplitude' in plots and pulsed_programming is not None:
        create_amplitude_plot(pulsed_programming, directory_name, file_output=file_output)
    if verbose:
        print(f'Amplitude plot: {time.time()-start}')
        start = time.time()

    if 'gaussian' in plots and pulsed_programming is not None:
        create_gaussian_distribution(pulsed_programming, directory_name, file_output=file_output)
    if verbose:
        print(f'Gaussian plot: {time.time() - start}')
        start = time.time()

    if 'stability' in plots and qd_sim is not None:
        create_stability_diagram(qd_sim, directory_name, file_output=file_output)
    if verbose:
        print(f'Stability plot: {time.time()-start}')
        start = time.time()

    if 'honeycomb' in plots and qd_sim is not None:
        create_honeycomb_diagram(qd_sim, directory_name, file_output=file_output)
    if verbose:
        print(f'Honeycomb plot: {time.time()-start}')


def create_result_plot(memristor_simulation, directory_name, file_output=False):
    """
    This function creates plots from the simulation voltages and save them in Result

    Parameters
    ----------
    memristor_simulation : MemristorSimulation.MemristorSimulation
        The memristor simulation object.

    directory_name : string
        The directory name where the plots will be save

    file_output : bool
        If true, output the values of the graph in a file.
    Returns
    ----------
    """
    if not os.path.isdir(f'{directory_name}\\Result'):
        os.mkdir(f'{directory_name}\\Result')

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
    plt.savefig(fname=f'{directory_name}\\Result\\{filename}')
    plt.close('all')

    if file_output:
        with h5py.File(f'{directory_name}\\Result\\result.hdf5', 'w') as f:
            f.create_dataset("voltages", data=memristor_simulation.voltages)
            f.create_dataset("resistances", data=memristor_simulation.resistances)


def create_resist_plot(pulsed_programming, directory_name, file_output=False):
    """
    This function creates plots from the simulation resistances and save them in Resist

    Parameters
    ----------
    pulsed_programming : PulsedProgramming.PulsedProgramming
        The pulsed programming

    directory_name : string
        The directory name where the plots will be save

    file_output : bool
        If true, output the values of the graph in a file.
    Returns
    ----------
    """
    if not os.path.isdir(f'{directory_name}\\Resist'):
        os.mkdir(f'{directory_name}\\Resist')
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
    plt.savefig(fname=f'{directory_name}\\Resist\\{filename}')
    plt.close('all')

    if file_output:
        with h5py.File(f'{directory_name}\\Resist\\resist.hdf5', 'w') as f:
            f.create_dataset("res_states", data=pulsed_programming.res_states)
            f.create_dataset("res_states_practical", data=pulsed_programming.res_states_practical)
            f.create_dataset("nb_states", data=pulsed_programming.nb_states)


def create_pulsed_programming_plot(pulsed_programming, directory_name, file_output=False):
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
        The directory name where the plots will be save

    file_output : bool
        If true, output the values of the graph in a file.
    Returns
    ----------
    """
    if not os.path.isdir(f'{directory_name}\\PulsedProgramming'):
        os.mkdir(f'{directory_name}\\PulsedProgramming')

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
    plt.savefig(fname=f'{directory_name}\\PulsedProgramming\\{filename}')
    plt.close('all')

    if file_output:
        x, y, action, annotation = zip(*pulsed_programming.graph_resistance)
        with h5py.File(f'{directory_name}\\PulsedProgramming\\pulsed_graph.hdf5', 'w') as f:
            f.create_dataset("x", data=x)
            f.create_dataset("y", data=y)
            f.create_dataset("action", data=action)
            f.create_dataset("annotation", data=annotation)


def create_amplitude_plot(pulsed_programming, directory_name, file_output=False):
    """
    This function creates a plot from the amplitude of the pulses in the pulsed programming simulation.

    Parameters
    ----------
    pulsed_programming : PulsedProgramming.PulsedProgramming
        The pulsed programming

    directory_name : string
        The directory name where the plots will be save

    file_output : bool
        If true, output the values of the graph in a file.
    Returns
    ----------
    """
    if not os.path.isdir(f'{directory_name}\\PulsedProgramming'):
        os.mkdir(f'{directory_name}\\PulsedProgramming')

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
    plt.savefig(fname=f'{directory_name}\\PulsedProgramming\\{filename}')
    plt.close('all')

    if file_output:
        voltages, counter, action = zip(*pulsed_programming.graph_voltages)
        with h5py.File(f'{directory_name}\\PulsedProgramming\\amplitude.hdf5', 'w') as f:
            f.create_dataset("voltages", data=voltages)
            f.create_dataset("counter", data=counter)
            f.create_dataset("action", data=action)


def create_gaussian_distribution(pulsed_programming, directory_name, file_output=False):
    """
    Output the gaussian distribution of the variability_read and variability_write.

    Parameters
    ----------
    pulsed_programming : PulsedProgramming.PulsedProgramming
        The pulsed programming

    directory_name : string
        The directory name where the plots will be save

    file_output : bool
        If true, output the values of the graph in a file.
    Returns
    ----------
    """
    if not os.path.isdir(f'{directory_name}\\PulsedProgramming'):
        os.mkdir(f'{directory_name}\\PulsedProgramming')
    # Read
    if pulsed_programming.variance_read != 0:
        mu = 0
        sigma = pulsed_programming.variance_read
        count, bins, ignored = plt.hist(pulsed_programming.variability_read, 30, density=True)
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
                 np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
                 linewidth=2, color='r')
        plt.xlabel('Variability read (%)')
        plt.ylabel('Number of appearance')
        filename = f'variability_read_gaussian_{pulsed_programming.variance_read}_variance.jpg'
        plt.tight_layout()
        plt.savefig(fname=f'{directory_name}\\PulsedProgramming\\{filename}')
        plt.close('all')

        if file_output:
            with h5py.File(f'{directory_name}\\PulsedProgramming\\variability_read.hdf5', 'w') as f:
                f.create_dataset("variability_read", data=pulsed_programming.variability_read)

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
        plt.savefig(fname=f'{directory_name}\\PulsedProgramming\\{filename}')
        plt.close('all')

        if file_output:
            with h5py.File(f'{directory_name}\\PulsedProgramming\\variability_write.hdf5', 'w') as f:
                f.create_dataset("variability_write", data=pulsed_programming.variability_write)


def create_stability_diagram(qd_simulation, directory_name, file_output=False):
    """
    This function creates the stability diagram from the qd_simulation and save them in Simulation\\StabilityDiagram.
    It's uses scatter with the height represented as color.

    Parameters
    ----------
    qd_simulation : QDSimulation.QDSimulation
        The quantum dot simulation

    directory_name : string
        The directory name where the plots will be save

    file_output : bool
        If true, output the values of the graph in a file.
    Returns
    ----------
    """
    if not os.path.isdir(f'{directory_name}\\StabilityDiagram'):
        os.mkdir(f'{directory_name}\\StabilityDiagram')
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
    plt.savefig(f'{directory_name}\\StabilityDiagram\\{filename}')
    plt.close('all')

    if file_output:
        with h5py.File(f'{directory_name}\\StabilityDiagram\\stability.hdf5', 'w') as f:
            f.create_dataset("voltages", data=qd_simulation.voltages)
            f.create_dataset("stability_diagram", data=qd_simulation.stability_diagram)


def create_honeycomb_diagram(qd_simulation, directory_name, file_output=False):
    """
    This function creates the honeycomb diagram from the qd_simulation and save them in Simulation\\StabilityDiagram.
    It's the differential of the stability diagram created in create_stability_diagram

    Parameters
    ----------
    qd_simulation : QDSimulation.QDSimulation
        The quantum dot simulation

    directory_name : string
        The directory name where the plots will be save

    file_output : bool
        If true, output the values of the graph in a file.
    Returns
    ----------
    """
    if not os.path.isdir(f'{directory_name}\\StabilityDiagram'):
        os.mkdir(f'{directory_name}\\StabilityDiagram')
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
    plt.savefig(f'{directory_name}\\StabilityDiagram\\{filename}', dpi=1200)
    plt.close('all')

    if file_output:
        with h5py.File(f'{directory_name}\\StabilityDiagram\\honeycomb.hdf5', 'w') as f:
            dx, dy = np.gradient(qd_simulation.stability_diagram)
            color = np.sqrt((dx / 2) ** 2 + (dy / 2) ** 2)

            f.create_dataset("voltages", data=qd_simulation.voltages)
            f.create_dataset("honeycomb", data=color)


def create_staircase_plot(qd_simulation, directory_name, file_output=False):
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
    plt.savefig(f'{directory_name}\\staircase_{resolution}.png', dpi=600)


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
