import qdms
import numpy as np
import copy
import math


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


def switch_v_in_v_min(circuit, voltage_min):
    r_min = circuit.memristor_model.r_on
    g_max = circuit.number_of_memristor * 1 / r_min

    if circuit.is_new_architecture:
        v_in = (voltage_min * circuit.R_L) / (1/g_max)
    else:
        v_in = voltage_min / (g_max * circuit.gain_resistance)
    circuit.v_in = v_in


def switch_v_in(circuit, voltage_min, voltage_max):
    r_min = circuit.memristor_model.r_on
    r_max = circuit.memristor_model.r_off
    g_min = circuit.number_of_memristor * 1 / r_max
    g_max = circuit.number_of_memristor * 1 / r_min

    if circuit.is_new_architecture:
        v_in_min = (voltage_min * circuit.R_L) / (1/g_max)
        v_in_max = (voltage_max * circuit.R_L) / (1/g_min)
    else:
        v_in_min = voltage_min / (g_max * circuit.gain_resistance)
        v_in_max = voltage_max / (g_min * circuit.gain_resistance)
    v_in = np.mean([v_in_min, v_in_max])
    circuit.v_in = v_in


def calculate_min_max_voltage(circuit):
    r_min = circuit.memristor_model.r_on
    r_max = circuit.memristor_model.r_off
    g_min = circuit.number_of_memristor * 1 / r_max
    g_max = circuit.number_of_memristor * 1 / r_min
    voltage_max = circuit.calculate_voltage(g_min)
    voltage_min = circuit.calculate_voltage(g_max)
    return voltage_min, voltage_max


def find_conductance(circuit, voltages):
    conductance = []
    for voltage in voltages:
        if circuit.is_new_architecture:
            res = (voltage * circuit.R_L) / circuit.v_in
        else:
            res = (circuit.gain_resistance * circuit.v_in) / voltage
        conductance.append(1 / res)
    return conductance


def change_conductance_simple(g_target, circuit, current_res=0):
    # current_v_out = circuit.current_v_out()
    # print(current_v_out)
    delta_g = g_target - circuit.current_conductance()
    circuit.list_memristor[current_res].g += delta_g
    diff_g = circuit.list_memristor[current_res].g - 1 / circuit.list_memristor[current_res].r_off
    if diff_g < 0:
        circuit.list_memristor[current_res].g = 1 / circuit.list_memristor[current_res].r_off
        if not current_res + 1 == len(circuit.list_memristor):
            change_conductance_simple(g_target, circuit, current_res + 1)
        else:
            print('Impossible to get the wanted conductance')


def find_smallest_res_change(res):
    res = np.swapaxes(np.array(res), 0, 1)
    minimum = []
    for res_list in res:
        diff = np.diff(res_list)
        print(f'{len(diff[np.nonzero(diff)])+1} etat\t{list(res_list)}')
        try:
            minimum_ = np.min(diff[np.nonzero(diff)])
        except:
            pass
        minimum.append(minimum_)
    return min(minimum)


def ask_v_max(circuit):
    v_min, v_max = calculate_min_max_voltage(circuit)
    voltage_max = 0
    while v_max < voltage_max or voltage_max < v_min:
        voltage_max = float(input(f'Voltage maximum must be between {v_min} and {v_max} (V). Enter v_max wanted: '))
    return voltage_max


memristor = qdms.Data_Driven()
circuit_ = qdms.Circuit(memristor, 9)
resolution = 0.001
v_min = 0.115


switch_v_in_v_min(circuit_, v_min)
v_max = ask_v_max(circuit_)
switch_v_in(circuit_, v_min, v_max)

num = (v_max - v_min) / resolution
voltages_ = np.linspace(v_min, v_max, num=math.ceil(num))
conductance_ = find_conductance(circuit_, voltages_)
resistance_ = [1 / i for i in conductance_]
# print(len(conductance_))

m = []
# SIMPLE
#####################
for g in conductance_:
    change_conductance_simple(g, circuit_)
    print(g - circuit_.current_conductance(),[1/l.g for l in circuit_.list_memristor])

# COMPLEX
#####################
result = []
rotation = 0
index = 1
# for res in resistance_:
#     print(res)
#     circuit_.list_memristor[0].g = 1 / (res * 9)
#     # for i in range(circuit_.number_of_memristor):
#     #     circuit_.list_memristor[i].g = 1 / (res * 9)
#     print(1/circuit_.current_conductance(), [1 / l.g for l in circuit_.list_memristor])
#     print()
#     index += 1


# FINAL
#####################
# print(len(conductance_))


# for g in conductance_:
#     delta_g = (g - circuit_.current_conductance()) / 9
#     for i in range(circuit_.number_of_memristor):
#         circuit_.list_memristor[i].g += delta_g
#         print(g - circuit_.current_conductance(), [1/l.g for l in circuit_.list_memristor])
#     print()

    # m.append([1/i.g for i in circuit_.list_memristor])

# print(find_smallest_res_change(m))


# nb_state = 8
# nb_memristor = 9
# result = list(itertools.combinations_with_replacement([i+1 for i in range(nb_state)], nb_memristor))
# print(result)
# print(len(result))


# nb_state = len(conductance_)**(1/circuit_.number_of_memristor)
