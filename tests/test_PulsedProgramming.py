import qdms
import numpy as np


def test_read_resistance_without_variability():
    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor, 1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 2)
    value = pulsed_programming.read_resistance(pulsed_programming.circuit.memristor_model)
    assert round(value) == round(pulsed_programming.circuit.memristor_model.r_on)


def test_read_resistance_with_variability():
    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor, 1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 2, variance_read=1/300)
    result = []
    for _ in range(1000):
        value = pulsed_programming.read_resistance(pulsed_programming.circuit.memristor_model)
        max = pulsed_programming.circuit.memristor_model.r_on + 0.015 * pulsed_programming.circuit.memristor_model.r_on
        min = pulsed_programming.circuit.memristor_model.r_on - 0.015 * pulsed_programming.circuit.memristor_model.r_on
        if min < value < max:
            result.append(True)
        else:
            result.append(False)
    assert np.all(result)


def test_write_resistance_without_variability():
    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor, 1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 2)
    pulsed_programming.write_resistance(pulsed_programming.circuit.memristor_model, -2, 200e-9)
    value = pulsed_programming.read_resistance(pulsed_programming.circuit.memristor_model)
    assert value > pulsed_programming.circuit.memristor_model.r_on


def test_write_resistance_with_variability():
    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor, 1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 2, variance_write=1/300)
    result_max = []
    result_min = []
    pulsed_programming.circuit.memristor_model.g = 1/2000
    for _ in range(1000):
        previous = pulsed_programming.read_resistance(pulsed_programming.circuit.memristor_model)
        pulsed_programming.write_resistance(pulsed_programming.circuit.memristor_model, 0, 200e-9)
        next = pulsed_programming.read_resistance(pulsed_programming.circuit.memristor_model)
        result_max.append((next - previous) / 2000 * 100 <= 1.2)
        result_min.append((next - previous) / 2000 * 100 >= 0.9)
    assert np.all(result_max) and np.any(result_min)


def test_distribution():
    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor, 9)
    pulsed_programming_linear = qdms.PulsedProgramming(circuit, 2, distribution_type='linear')
    pulsed_programming_half_spread = qdms.PulsedProgramming(circuit, 2, distribution_type='half_spread')
    pulsed_programming_full_spread = qdms.PulsedProgramming(circuit, 2, distribution_type='full_spread')
    result = False
    if len(pulsed_programming_linear.res_states) == 1:
        if len(pulsed_programming_half_spread.res_states) == 3:
            if len(pulsed_programming_full_spread.res_states) == 9:
                result = True
    assert result


def test_simple_convergence():
    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor, 1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 2, hrs=3000, tolerance=1, is_relative_tolerance=True, pulse_algorithm='simple')
    pulsed_programming.simulate()
    assert not len(pulsed_programming.graph_resistance) - 1 == pulsed_programming.max_pulse


def test_log_convergence():
    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor, 1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 2, hrs=3000, tolerance=1, is_relative_tolerance=True,
                                                pulse_algorithm='log')
    pulsed_programming.simulate()
    assert not len(pulsed_programming.graph_resistance) -1 == pulsed_programming.max_pulse


def test_fabien_convergence():
    memristor = qdms.Data_Driven()
    circuit = qdms.Circuit(memristor, 1)
    pulsed_programming = qdms.PulsedProgramming(circuit, 2, hrs=3000, tolerance=1, is_relative_tolerance=True,
                                                pulse_algorithm='fabien')
    pulsed_programming.simulate()
    assert not len(pulsed_programming.graph_resistance) - 1 == pulsed_programming.max_pulse
