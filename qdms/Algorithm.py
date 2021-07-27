import numpy as np
import math
import time
from bisect import bisect_left


def algorithm(resolution, memristor_simulation, diff_flag=False):
    """
    Parameters
    ----------.
    resolution : flaot
        Resolution of the stability diagram.

    memristor_simulation: MemristorSimulation
        The memristor simulation object

    diff_flag : bool
        if True, will output the difference between the target and the result

    Returns
    -------
    voltages : dict
        Dictionary where the key is the voltage output and the package is the resistance value of each memristor

    """
    v_min, v_max = ask_v_min_v_max(memristor_simulation.circuit)
    voltage_target = np.linspace(v_min, v_max, num=math.ceil((v_max - v_min) / resolution) + 1)
    print(f'Sweep between {v_min} and {v_max} with a step of {resolution}, which give {round(len(voltage_target))} values')
    voltages = find_correspondence(voltage_target, [l[1] for l in memristor_simulation.voltages_memristor])

    if diff_flag:
        diff = []
        for i in range(len(list(voltages.keys()))):
            # print(f'{list(voltages.keys())[i]}\t{np.sort([round(i) for i in voltages.get(list(voltages.keys())[i])])}')
            diff.append(abs(list(voltages.keys())[i] - voltage_target[i]))
        print(f'Max diff: {max(diff)} (V)\t% of diff: {max(diff) / resolution * 100} %')
        print(f'Mean diff: {np.mean(diff)} (V)\t% of diff: {np.mean(diff) / resolution * 100} %')
        print()
    return voltages


def calculate_min_max_voltage(circuit):
    """
    Parameters
    ----------.
    circuit : Circuit
        Circuit object

    Returns
    -------
    voltage_min, voltage_max : float
        The voltage minimum and maximum possible with this circuit

    """
    r_min = circuit.memristor_model.r_on
    r_max = circuit.memristor_model.r_off
    g_min = circuit.number_of_memristor * 1 / r_max
    g_max = circuit.number_of_memristor * 1 / r_min
    voltage_max = circuit.calculate_voltage(g_min)
    voltage_min = circuit.calculate_voltage(g_max)
    return voltage_min, voltage_max


def ask_v_min_v_max(circuit):
    """
    Parameters
    ----------.
    circuit : Circuit
        Circuit object

    Returns
    -------
    voltage_min, voltage_max : float
        The voltage minimum and maximum chose by the user

    """
    v_min, v_max = calculate_min_max_voltage(circuit)
    voltage_min = voltage_max = -1
    while not v_min <= voltage_min <= v_max:
        voltage_min = float(input(f'Enter v_min (must be between {v_min} and {v_max}):'))
    while not voltage_min <= voltage_max <= v_max:
        voltage_max = float(input(f'Enter v_max (must be between {voltage_min} and {v_max}):'))
    return voltage_min, voltage_max


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def find_correspondence(voltage_target, voltage_table):
    """
    Parameters
    ----------.
    voltage_target : list of float
        The list of wanted voltages

    voltage_table : list of float
        The list of possible voltages from the memristor_simulation

    Returns
    -------
    voltages : dict
        Dictionary where the key is the voltage output and the package is the resistance value of each memristor, sorted.


    """
    voltages = {}
    time_start = time.time()
    for i in range(len(voltage_target)):
        # v = min(memristor_simulation_.voltages_memristor, key=lambda x:(abs(x - voltages_t)))
        v = take_closest(voltage_table, voltage_target[i])
        voltages[v] = np.sort(voltage_table)
    print(f'Total time {time.time() - time_start}')
    return voltages
