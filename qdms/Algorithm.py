import qdms
import numpy as np
import copy
import math
import itertools


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


def change_conductance_simple(g_target, circuit, record_, current_res=0):
    delta_g = g_target - circuit.current_conductance()

    circuit.list_memristor[current_res].g += delta_g

    diff_g = circuit.list_memristor[current_res].g - 1 / circuit.list_memristor[current_res].r_off
    if diff_g < 0:
        circuit.list_memristor[current_res].g = 1 / circuit.list_memristor[current_res].r_off
        if not current_res + 1 == len(circuit.list_memristor):
            change_conductance_simple(g_target, circuit, record_, current_res + 1)
        else:
            print('Impossible to get the wanted conductance')

    res = 1 / circuit.list_memristor[current_res].g
    # tolerance = 50
    tolerance = circuit.memristor_model.variability(res) * 10
    for state_res in record_[current_res]:
        if state_res - tolerance < res < state_res + tolerance:
            circuit.list_memristor[current_res].g = 1 / state_res


def set_initial_g(g_target, circuit, current_res=0):
    delta_g = g_target - circuit.current_conductance()
    if circuit.list_memristor[current_res].g + delta_g < 1 / circuit.memristor_model.r_off:
        circuit.list_memristor[current_res].g = 1 / circuit.memristor_model.r_off
        set_initial_g(g_target, circuit, current_res + 1)
    else:
        circuit.list_memristor[current_res].g += delta_g


def rec(g_target, circuit, record, current_res):
    # print(1/circuit.list_memristor[current_res].g)
    delta_g = g_target - circuit.current_conductance()
    delta_res = (1 / (circuit.list_memristor[current_res].g + delta_g)) - record.get(current_res)[-1]
    # if delta_res < 0:
    #     print(round(record.get(current_res)[-1]), round(1 / (circuit.list_memristor[current_res].g + delta_g)), round(delta_res), current_res)
    if abs(delta_res) < 100 and delta_res != 0:
        adjust = 100 / delta_res
        circuit.list_memristor[current_res].g += delta_g * adjust
        rec(g_target, circuit, record, current_res - 1)
        # circuit.list_memristor[current_res - 1].g -= delta_g * adjust - delta_g
    else:
        circuit.list_memristor[current_res].g += delta_g


def set_g(g_target, circuit, record, current_res=0):
    delta_g = g_target - circuit.current_conductance()
    if circuit.list_memristor[current_res].g + delta_g < 1 / circuit.memristor_model.r_off:
        circuit.list_memristor[current_res].g = 1 / circuit.memristor_model.r_off
        set_g(g_target, circuit, record, current_res + 1)
    else:
        rec(g_target, circuit, record, current_res)


def set_g_2(g_target, circuit, record, current_res):
    delta_g = g_target - circuit.current_conductance()
    final_res = 1 / (circuit.list_memristor[current_res].g + delta_g)
    delta_res = final_res - record.get(current_res)[-1]

    if delta_res > 100:
        print(f'Big delta res\t\t\t{delta_res}\t{[round(1 / i.g) for i in circuit.list_memristor]}')
        if final_res < circuit.memristor_model.r_off:
            circuit.list_memristor[current_res].g += delta_g
        else:
            circuit.list_memristor[current_res].g = 1 / circuit.memristor_model.r_off
            set_g_2(g_target, circuit, record, current_res + 1)

    elif 10 < delta_res < 100:
        print(f'Little delta res\t\t{delta_res}\t{[round(1 / i.g) for i in circuit.list_memristor]}')
        adjust = 100 / delta_res
        if circuit.list_memristor[current_res].g + delta_g * adjust < circuit.memristor_model.r_off:
            circuit.list_memristor[current_res].g += delta_g * adjust
            set_g_2(g_target, circuit, record, current_res - 1)

    elif -10 < delta_res < 10:
        print(f'LITTLE delta res, skip\t{delta_res}\t{[round(1 / i.g) for i in circuit.list_memristor]}')


    #     else:
    #         circuit.list_memristor[current_res].g = 1 / circuit.memristor_model.r_off
    #         balance()
    #         set_g_2(g_target, circuit, record, current_res)

    # elif -100 <= delta_res < 0:
    #     print(f'Little minus delta res\t{delta_res}\t{[round(1 / i.g) for i in circuit.list_memristor]}')
    #
    # elif delta_res < -100:
    # elif delta_res < 0:
    #     print(f'Big minus delta res\t\t{delta_res}\t{[round(1 / i.g) for i in circuit.list_memristor]}')
    #     if final_res > circuit.memristor_model.r_on:
    #         circuit.list_memristor[current_res].g += delta_g
    #     else:
    #         circuit.list_memristor[current_res].g = 1 / circuit.memristor_model.r_on
    #         set_g_2(g_target, circuit, record, current_res - 1)
    # #

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


def find_current_res(circuit):
    counter = 0
    for res in circuit.list_memristor:
        if 1 / res.g < res.r_off:
            return counter
        counter += 1
    return -1


def ask_v_min_v_max(circuit):
    v_min, v_max = calculate_min_max_voltage(circuit)
    voltage_min = voltage_max = -1
    while not v_min <= voltage_min <= v_max:
        voltage_min = float(input(f'Enter v_min (must be between {v_min} and {v_max}):'))
    while not voltage_min <= voltage_max <= v_max:
        voltage_max = float(input(f'Enter v_max (must be between {voltage_min} and {v_max}):'))
    return voltage_min, voltage_max


memristor_ = qdms.Data_Driven()
circuit_ = qdms.Circuit(memristor_, 4)
pulsed_programming_ = qdms.PulsedProgramming(circuit_, 5, distribution_type='full_spread')
pulsed_programming_.simulate()
memristor_simulation_ = qdms.MemristorSimulation(pulsed_programming_)
memristor_simulation_.simulate()


resolution = 0.001
# v_min, v_max = ask_v_min_v_max(circuit_)
v_min = 0.4
v_max = 0.41
print(f'Sweep between {v_min} and {v_max} with a step of {resolution}, which give {round((v_max - v_min) / resolution)} values')

record_ = {}
num = (v_max - v_min) / resolution
voltages_ = np.linspace(v_min, v_max, num=math.ceil(num))
conductance_ = find_conductance(circuit_, voltages_)
resistances_ = 1 / (np.array(conductance_))
set_initial_g(1 / resistances_[0], circuit_)
print(resistances_[0],1/circuit_.current_conductance(),[1/i.g for i in circuit_.list_memristor])
for i in range(circuit_.number_of_memristor):
    record_[i] = [1/circuit_.list_memristor[i].g]
for g in list(1 / resistances_):
    # set_g(g, circuit_, record_)
    current_res = find_current_res(circuit_)
    set_g_2(g, circuit_, record_, current_res)
    for key in range(len(record_.keys())):
        record_[key].append(1 / circuit_.list_memristor[key].g)
    print(round(1 / g) ,round(1/circuit_.current_conductance()),[ round(1/i.g) for i in circuit_.list_memristor])
    print()

for key in record_.keys():
    print([round(res) for res in record_.get(key)])

# circuit_.list_memristor[0].g += delta_g_[0]
# print(resistances_[0],1/circuit_.current_conductance(),[1/i.g for i in circuit_.list_memristor])

# memristor_ = qdms.Data_Driven()
# circuit_ = qdms.Circuit(memristor_, 9)
# resolution = 0.0001
#
# v_min, v_max = ask_v_min_v_max(circuit_)
# switch_v_in_v_min(circuit_, v_min)
# v_max = ask_v_max(circuit_)
# switch_v_in(circuit_, v_min, v_max)

# num = (v_max - v_min) / resolution
# voltages_ = np.linspace(v_min, v_max, num=math.ceil(num))
# conductance_ = find_conductance(circuit_, voltages_)
# print(len(conductance_))

# find_all(circuit_)

# result = []
# SIMPLE
#####################
# record = {}
# for i in range(9):
#     record[i] = []
# for g in conductance_:
#     for i in range(9):
#         record[i].append(1 / circuit_.list_memristor[i].g)
#     change_conductance_simple(g, circuit_, record)
#     result.append(circuit_.current_conductance())
# for i in record.keys():
#     print([round(i) for i in record[i]])
# diff = [voltages_[i] - circuit_.calculate_voltage(result[i]) for i in range(len(conductance_))]
# counter = 0
# for i in diff:
#     if i > resolution:
#         counter += 1
# print(counter, len(diff))
    # print(g - circuit_.current_conductance(),[1/l.g for l in circuit_.list_memristor])

# COMPLEX
#####################
# result = []
# rotation = 0
# index = 1
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
