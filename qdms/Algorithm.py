import qdms
import numpy as np


def algorithm(voltage_min, voltage_max, resolution):
    """

    Parameters
    ----------
    voltage_min : float
        Minimum voltage for the stability diagram.
    voltage_max : float
        Maximum voltage for the stability diagram.
    resolution : flaot
        Resolution of the stability diagram.

    Returns
    -------
    parameters : list
        List of all the parameters needed to get a stability diagram with the input parameters.

    """

    return


# def find_min_max_voltage(memristor, target_voltage_min, target_voltage_max):
#     v_in = 1e-3
#     nb_memristor = 9
#     flag_finish = False
#     nb_iteration = 0
#     while not flag_finish and nb_iteration < 1000:
#         circuit = qdms.Circuit(memristor, nb_memristor, v_in=v_in)
#         voltage_min, voltage_max = calculate_min_max_voltage(circuit)
#         flag_finish, v_in = evaluate_finish(voltage_min, voltage_max, target_voltage_min, target_voltage_max, v_in)
#         nb_iteration += 1
#     return
#
#
# def find_min_voltage(memristor, target_voltage_min):
#     v_in = 1e-3
#     nb_memristor = 9
#     flag_finish = False
#     nb_iteration = 0
#     p_found_min = False
#     while not flag_finish and nb_iteration < 1000:
#         circuit = qdms.Circuit(memristor, nb_memristor, v_in=v_in)
#         voltage_min, voltage_max = calculate_min_max_voltage(circuit)
#         found_min = True if voltage_min < target_voltage_min else False
#
#         if found_min and not p_found_min:
#             pass
#         if found_min:
#             v_in += v_in / 10
#
#         nb_iteration += 1
#         p_found_min = found_min
#     return

#
# def evaluate_finish(voltage_min, voltage_max, target_voltage_min, target_voltage_max, v_in):
#     found_min = True if voltage_min < target_voltage_min else False
#     found_max = True if voltage_max > target_voltage_max else False
#     flag_finish = False
#     if found_max and found_min:
#         print(f'Found value!\tmin: {round(voltage_min, 3)}\tmax: {round(voltage_max, 3)}\twith v_in: {v_in}')
#         flag_finish = True
#     elif found_min:
#         v_in += v_in/10
#     elif found_max:
#         v_in -= v_in/10
#     else:
#         print('Target value to large')
#         flag_finish = True
#     print(f'min: {round(voltage_min, 3)}\tmax: {round(voltage_max, 3)}\tv_in: {round(v_in, 5)}\tflag_finish: {flag_finish}')
#     return flag_finish, v_in


def switch_v_in(circuit, voltage_min):
    r_min = circuit.memristor_model.r_on
    g_max = circuit.number_of_memristor * 1 / r_min

    if circuit.is_new_architecture:
        v_in = (voltage_min * circuit.R_L) / (1/g_max + circuit.gain_resistance)
    else:
        v_in = voltage_min / (g_max * circuit.gain_resistance)
    circuit.v_in = v_in


def calculate_min_max_voltage(circuit):
    r_min = circuit.memristor_model.r_on
    r_max = circuit.memristor_model.r_off
    g_min = circuit.number_of_memristor * 1 / r_max
    g_max = circuit.number_of_memristor * 1 / r_min
    voltage_max = circuit.calculate_voltage(g_min)
    voltage_min = circuit.calculate_voltage(g_max)
    return voltage_min, voltage_max


def find_voltages(resolution, circuit):
    voltage_min, voltage_max = calculate_min_max_voltage(circuit)
    voltages = []
    voltage = voltage_min
    while voltage < voltage_max:
        voltages.append(voltage)
        voltage += resolution
    return voltages


def find_resistances(voltages, circuit):
    resistances = []
    for voltage in voltages:
        if circuit.is_new_architecture:
            res = (voltage * circuit.R_L) / circuit.v_in - circuit.gain_resistance
        else:
            res = (circuit.gain_resistance * circuit.v_in) / voltage
        resistances.append(res)
    return resistances


memristor = qdms.Data_Driven()
# find_min_max_voltage(memristor, 0.20, 1)
# find_min_voltage(memristor, 0.2)
circuit_ = qdms.Circuit(memristor, 9)
switch_v_in(circuit_, 0.2)
voltages_ = find_voltages(0.01, circuit_)
resistances_ = find_resistances(voltages_, circuit_)
print(np.array(resistances_))
