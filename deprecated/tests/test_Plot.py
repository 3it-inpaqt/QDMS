import qdms
import h5py
import numpy as np


def test_plot_amplitude():
    with h5py.File(f'.\\tests\\TestData\\Plots\\amplitude.hdf5', 'r') as file:
        voltages_load = list(file.get('voltages'))
        counter_load = list(file.get('counter'))
        action_load = list(file.get('action'))

    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor_model=memristor, number_of_memristor=9, is_new_architecture=True, v_in=1e-3,
                           gain_resistance=0, R_L=1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 4, tolerance=2, is_relative_tolerance=True)
    pulsed_programming.simulate()

    voltages, counter, action = zip(*pulsed_programming.graph_voltages)
    for i in range(len(action_load)):
        action_load[i] = str(action_load[i]).lstrip("b'").rstrip("'")
    assert list(voltages) == voltages_load and list(counter) == list(counter_load) and list(action) == action_load


def test_plot_pulsed_graph():
    with h5py.File(f'.\\tests\\TestData\\Plots\\pulsed_graph.hdf5', 'r') as file:
        x_load = list(file.get('x'))
        y_load = list(file.get('y'))
        action_load = list(file.get('action'))
        annotation_load = list(file.get('annotation'))

    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor_model=memristor, number_of_memristor=9, is_new_architecture=True, v_in=1e-3,
                           gain_resistance=0, R_L=1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 4, tolerance=2, is_relative_tolerance=True)
    pulsed_programming.simulate()

    x, y, action, annotation = zip(*pulsed_programming.graph_resistance)
    for i in range(len(action_load)):
        action_load[i] = str(action_load[i]).lstrip("b'").rstrip("'")
    assert list(x) == x_load and list(y) == y_load and list(action) == action_load and list(annotation) == annotation_load


def test_plot_honeycomb():
    with h5py.File(f'.\\tests\\TestData\\Plots\\honeycomb.hdf5', 'r') as file:
        voltages_load = list(file.get('voltages'))
        honeycomb_load = list(file.get('honeycomb'))

    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor_model=memristor, number_of_memristor=9, is_new_architecture=True, v_in=1e-3,
                           gain_resistance=0, R_L=1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 4, tolerance=2, is_relative_tolerance=True)
    pulsed_programming.simulate()
    memristor_sim = qdms.MemristorSimulation(pulsed_programming, verbose=True)
    memristor_sim.simulate()
    vector = qdms.HelperFunction.limit_vector(memristor_sim.voltages, 0.2, 0.25)
    vector = qdms.HelperFunction.simplify_vector_resolution(vector, 0.0001)
    qd_sim = qdms.QDSimulation(vector)
    qd_sim.simulate()
    dx, dy = np.gradient(qd_sim.stability_diagram)
    honeycomb = np.sqrt((dx / 2) ** 2 + (dy / 2) ** 2)

    assert np.all(voltages_load == qd_sim.voltages) and np.all(honeycomb_load == honeycomb)


def test_plot_stability():
    with h5py.File(f'.\\tests\\TestData\\Plots\\stability.hdf5', 'r') as file:
        voltages_load = list(file.get('voltages'))
        stability_load = list(file.get('stability_diagram'))

    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor_model=memristor, number_of_memristor=9, is_new_architecture=True, v_in=1e-3,
                           gain_resistance=0, R_L=1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 4, tolerance=2, is_relative_tolerance=True)
    pulsed_programming.simulate()
    memristor_sim = qdms.MemristorSimulation(pulsed_programming, verbose=True)
    memristor_sim.simulate()
    vector = qdms.HelperFunction.limit_vector(memristor_sim.voltages, 0.2, 0.25)
    vector = qdms.HelperFunction.simplify_vector_resolution(vector, 0.0001)
    qd_sim = qdms.QDSimulation(vector)
    qd_sim.simulate()

    assert np.all(voltages_load == qd_sim.voltages) and np.all(stability_load == qd_sim.stability_diagram)


def test_plot_resist():
    with h5py.File(f'.\\tests\\TestData\\Plots\\resist.hdf5', 'r') as file:
        res_states_load = list(file.get('res_states'))
        res_states_practical_load = np.array(file.get('res_states_practical'))
        nb_states_load = np.array(file.get('nb_states'))

    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor_model=memristor, number_of_memristor=9, is_new_architecture=True, v_in=1e-3,
                           gain_resistance=0, R_L=1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 4, tolerance=2, is_relative_tolerance=True)
    pulsed_programming.simulate()

    result = False
    next = True

    for i in range(len(res_states_practical_load)):
        if np.all(res_states_practical_load[i] == pulsed_programming.res_states_practical[i]):
            continue
        next = False

    if next:
        for i in range(len(res_states_load)):
            if np.all(res_states_load[i] == pulsed_programming.res_states[i]):
                continue
            next = False

    if next:
        if nb_states_load == pulsed_programming.nb_states:
            result = True

    assert result


def test_plot_result():
    with h5py.File(f'.\\tests\\TestData\\Plots\\result.hdf5', 'r') as file:
        voltages_load = list(file.get('voltages'))
        resistances_load = list(file.get('resistances'))

    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor_model=memristor, number_of_memristor=9, is_new_architecture=True, v_in=1e-3,
                           gain_resistance=0, R_L=1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 4, tolerance=2, is_relative_tolerance=True)
    pulsed_programming.simulate()
    memristor_sim = qdms.MemristorSimulation(pulsed_programming, verbose=True)
    memristor_sim.simulate()

    assert voltages_load == memristor_sim.voltages and resistances_load == memristor_sim.resistances
