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

    Returns
    -------
    parameters : list
        List of all the parameters needed to get a stability diagram with the input parameters.

    """
    v_min, v_max = ask_v_min_v_max(memristor_simulation.circuit)
    voltage_target = np.linspace(v_min, v_max, num=math.ceil((v_max - v_min) / resolution) + 1)
    print(f'Sweep between {v_min} and {v_max} with a step of {resolution}, which give {round(len(voltage_target))} values')
    voltages = find_correspondence(voltage_target, memristor_simulation.voltages_memristor)

    if diff_flag:
        diff = []
        for i in range(len(list(voltages.keys()))):
            # print(f'{list(voltages.keys())[i]}\t{np.sort([round(i) for i in voltages.get(list(voltages.keys())[i])])}')
            diff.append(list(voltages.keys())[i] - voltage_target[i])
        print(f'Max diff: {max(diff)} (V)\t% of diff: {max(diff) / resolution * 100} %')
        print(f'Mean diff: {np.mean(diff)} (V)\t% of diff: {np.mean(diff) / resolution * 100} %')
        print()
    return voltages


def calculate_min_max_voltage(circuit):
    r_min = circuit.memristor_model.r_on
    r_max = circuit.memristor_model.r_off
    g_min = circuit.number_of_memristor * 1 / r_max
    g_max = circuit.number_of_memristor * 1 / r_min
    voltage_max = circuit.calculate_voltage(g_min)
    voltage_min = circuit.calculate_voltage(g_max)
    return voltage_min, voltage_max


def ask_v_min_v_max(circuit):
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
    voltages = {}
    time_start = time.time()
    for voltage_t in voltage_target:
        # v = min(memristor_simulation_.voltages_memristor, key=lambda x:(abs(x - voltages_t)))
        v = take_closest(list(voltage_table.keys()), voltage_t)
        voltages[v] = np.sort(voltage_table.get(v))
    print(f'Total time {time.time() - time_start}')
    return voltages


###########
# DEPRECATED
# def find_conductance(circuit, voltages):
#     conductance = []
#     for voltage in voltages:
#         if circuit.is_new_architecture:
#             res = (voltage * circuit.R_L) / circuit.v_in
#         else:
#             res = (circuit.gain_resistance * circuit.v_in) / voltage
#         conductance.append(1 / res)
#     return conductance
#
#
# def change_conductance_simple(g_target, circuit, record_, current_res=0):
#     delta_g = g_target - circuit.current_conductance()
#
#     circuit.list_memristor[current_res].g += delta_g
#
#     diff_g = circuit.list_memristor[current_res].g - 1 / circuit.list_memristor[current_res].r_off
#     if diff_g < 0:
#         circuit.list_memristor[current_res].g = 1 / circuit.list_memristor[current_res].r_off
#         if not current_res + 1 == len(circuit.list_memristor):
#             change_conductance_simple(g_target, circuit, record_, current_res + 1)
#         else:
#             print('Impossible to get the wanted conductance')
#
#     res = 1 / circuit.list_memristor[current_res].g
#     # tolerance = 50
#     tolerance = circuit.memristor_model.variability(res) * 10
#     for state_res in record_[current_res]:
#         if state_res - tolerance < res < state_res + tolerance:
#             circuit.list_memristor[current_res].g = 1 / state_res
#
#
# def set_initial_g(g_target, circuit, current_res=0):
#     delta_g = g_target - circuit.current_conductance()
#     if circuit.list_memristor[current_res].g + delta_g < 1 / circuit.memristor_model.r_off:
#         circuit.list_memristor[current_res].g = 1 / circuit.memristor_model.r_off
#         set_initial_g(g_target, circuit, current_res + 1)
#     else:
#         circuit.list_memristor[current_res].g += delta_g
#
#
# def rec(g_target, circuit, record, current_res):
#     # print(1/circuit.list_memristor[current_res].g)
#     delta_g = g_target - circuit.current_conductance()
#     delta_res = (1 / (circuit.list_memristor[current_res].g + delta_g)) - record.get(current_res)[-1]
#     # if delta_res < 0:
#     #     print(round(record.get(current_res)[-1]), round(1 / (circuit.list_memristor[current_res].g + delta_g)), round(delta_res), current_res)
#     if abs(delta_res) < 100 and delta_res != 0:
#         adjust = 100 / delta_res
#         circuit.list_memristor[current_res].g += delta_g * adjust
#         rec(g_target, circuit, record, current_res - 1)
#         # circuit.list_memristor[current_res - 1].g -= delta_g * adjust - delta_g
#     else:
#         circuit.list_memristor[current_res].g += delta_g
#
#
# def set_g(g_target, circuit, record, current_res=0):
#     delta_g = g_target - circuit.current_conductance()
#     if circuit.list_memristor[current_res].g + delta_g < 1 / circuit.memristor_model.r_off:
#         circuit.list_memristor[current_res].g = 1 / circuit.memristor_model.r_off
#         set_g(g_target, circuit, record, current_res + 1)
#     else:
#         # rec(g_target, circuit, record, current_res)
#         circuit.list_memristor[current_res].g += delta_g
#
#
# def set_g_2(g_target, circuit, record, current_res):
#     delta_g = g_target - circuit.current_conductance()
#     final_res = 1 / (circuit.list_memristor[current_res].g + delta_g)
#     delta_res = final_res - record.get(current_res)[-1]
#
#     if delta_res > 100:
#         print(f'Big delta res\t\t\t{delta_res}\t{[round(1 / i.g) for i in circuit.list_memristor]}')
#         if final_res < circuit.memristor_model.r_off:
#             circuit.list_memristor[current_res].g += delta_g
#         else:
#             circuit.list_memristor[current_res].g = 1 / circuit.memristor_model.r_off
#             set_g_2(g_target, circuit, record, current_res + 1)
#
#     elif 10 < delta_res < 100:
#         print(f'Little delta res\t\t{delta_res}\t{[round(1 / i.g) for i in circuit.list_memristor]}')
#         adjust = 100 / delta_res
#         if circuit.list_memristor[current_res].g + delta_g * adjust < circuit.memristor_model.r_off:
#             circuit.list_memristor[current_res].g += delta_g * adjust
#             set_g_2(g_target, circuit, record, current_res - 1)
#
#     elif -10 < delta_res < 10:
#         print(f'LITTLE delta res, skip\t{delta_res}\t{[round(1 / i.g) for i in circuit.list_memristor]}')


        # else:
        #     circuit.list_memristor[current_res].g = 1 / circuit.memristor_model.r_off
        #     balance()
        #     set_g_2(g_target, circuit, record, current_res)

    # elif -100 <= delta_res < 0:
    #     print(f'Little minus delta res\t{delta_res}\t{[round(1 / i.g) for i in circuit.list_memristor]}')
    #
    # elif delta_res < -100:
    #     print(f'Big minus delta res\t\t{delta_res}\t{[round(1 / i.g) for i in circuit.list_memristor]}')
    #     if final_res > circuit.memristor_model.r_on:
    #         circuit.list_memristor[current_res].g += delta_g
    #     else:
    #         circuit.list_memristor[current_res].g = 1 / circuit.memristor_model.r_on
    #         set_g_2(g_target, circuit, record, current_res - 1)
    #

    # if delta_res < 100 and delta_res != 0 and final_res <= circuit.memristor_model.r_off:
    #     adjust = 100 / delta_res
    #     final_res = 1 / (circuit.list_memristor[current_res].g + delta_g * adjust)
    #     delta_res = final_res - record.get(current_res)[-1]
    #     if final_res > circuit.memristor_model.r_off:
    #         print(final_res)
    #     circuit.list_memristor[current_res].g += delta_g * adjust
    #     circuit.list_memristor[current_res - 1].g -= delta_g * adjust
    # else:
    #     circuit.list_memristor[current_res].g += delta_g

# def switch_v_in_v_min(circuit, voltage_min):
#     r_min = circuit.memristor_model.r_on
#     g_max = circuit.number_of_memristor * 1 / r_min
#
#     if circuit.is_new_architecture:
#         v_in = (voltage_min * circuit.R_L) / (1/g_max)
#     else:
#         v_in = voltage_min / (g_max * circuit.gain_resistance)
#     circuit.v_in = v_in
#
#
# def switch_v_in(circuit, voltage_min, voltage_max):
#     r_min = circuit.memristor_model.r_on
#     r_max = circuit.memristor_model.r_off
#     g_min = circuit.number_of_memristor * 1 / r_max
#     g_max = circuit.number_of_memristor * 1 / r_min
#
#     if circuit.is_new_architecture:
#         v_in_min = (voltage_min * circuit.R_L) / (1/g_max)
#         v_in_max = (voltage_max * circuit.R_L) / (1/g_min)
#     else:
#         v_in_min = voltage_min / (g_max * circuit.gain_resistance)
#         v_in_max = voltage_max / (g_min * circuit.gain_resistance)
#     v_in = np.mean([v_in_min, v_in_max])
#     circuit.v_in = v_in