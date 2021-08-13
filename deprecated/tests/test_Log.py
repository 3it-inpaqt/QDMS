import shutil

import qdms
import numpy as np


def setup_save_load():
    memristor = qdms.Data_Driven()

    circuit = qdms.Circuit(memristor_model=memristor, number_of_memristor=4, is_new_architecture=True, v_in=1e-3,
                           gain_resistance=0, R_L=1)

    pulsed_programming = qdms.PulsedProgramming(circuit, 4, tolerance=1, is_relative_tolerance=True)
    pulsed_programming.simulate()

    memristor_sim = qdms.MemristorSimulation(pulsed_programming, verbose=False)
    memristor_sim.simulate()

    # memristor_result = qdms.HelperFunction.limit_vector(memristor_sim.voltages, 0.175, 0.210)
    memristor_result = qdms.HelperFunction.simplify_vector_resolution(memristor_sim.voltages, 0.0001)

    quantum_sim = qdms.QDSimulation(memristor_result, verbose=False)
    quantum_sim.simulate()

    return memristor, circuit, pulsed_programming, memristor_sim, quantum_sim


def compare_everything(setup, load):
    if compare_memristor(setup[0], load[0]):
        if compare_circuit(setup[1], load[1]):
            if compare_pulsed_programming(setup[2], load[2]):
                if compare_memristor_sim(setup[3], load[3]):
                    if compare_quantum_sim(setup[4], load[4]):
                        return True
    return False


def compare_memristor(setup, load):
    if type(setup) != type(load): return False
    if setup.time_series_resolution != load.time_series_resolution: return False
    if setup.r_off != load.r_off: return False
    if setup.r_on != load.r_on: return False
    if setup.A_p != load.A_p: return False
    if setup.A_n != load.A_n: return False
    if setup.t_p != load.t_p: return False
    if setup.t_n != load.t_n: return False
    if setup.k_p != load.k_p: return False
    if setup.k_n != load.k_n: return False
    if setup.r_p != load.r_p: return False
    if setup.r_n != load.r_n: return False
    if setup.eta != load.eta: return False
    if setup.a_p != load.a_p: return False
    if setup.a_n != load.a_n: return False
    if setup.b_p != load.b_p: return False
    if setup.b_n != load.b_n: return False
    if setup.g != load.g: return False
    return True


def compare_circuit(setup, load):
    if setup.number_of_memristor != load.number_of_memristor: return False
    if setup.gain_resistance != load.gain_resistance: return False
    if setup.v_in != load.v_in: return False
    if setup.R_L != load.R_L: return False
    if setup.is_new_architecture != load.is_new_architecture: return False
    return True


def compare_pulsed_programming(setup, load):
    if setup.nb_states != load.nb_states: return False
    if setup.distribution_type != load.distribution_type: return False
    if setup.pulse_algorithm != load.pulse_algorithm: return False
    if setup.lrs != load.lrs: return False
    if setup.hrs != load.hrs: return False
    if setup.res_states != load.res_states: return False
    if setup.res_states_practical != load.res_states_practical: return False
    if setup.max_voltage != load.max_voltage: return False
    if setup.tolerance != load.tolerance: return False
    if setup.index_variability != load.index_variability: return False
    if setup.variance_read != load.variance_read: return False
    if setup.variance_write != load.variance_write: return False
    if np.all(setup.variability_read != load.variability_read): return False
    if np.all(setup.variability_write != load.variability_write): return False
    if setup.number_of_reading != load.number_of_reading: return False
    if setup.graph_resistance != load.graph_resistance: return False
    if setup.graph_voltages != load.graph_voltages: return False
    if setup.max_pulse != load.max_pulse: return False
    if setup.is_relative_tolerance != load.is_relative_tolerance: return False
    return True


def compare_memristor_sim(setup, load):
    if setup.is_using_conductance != load.is_using_conductance: return False
    if setup.voltages != load.voltages: return False
    if setup.resistances != load.resistances: return False
    if setup.verbose != load.verbose: return False
    if setup.list_resistance != load.list_resistance: return False
    if setup.timers != load.timers: return False
    if setup.resolution != load.resolution: return False
    if setup.std != load.std: return False
    return True


def compare_quantum_sim(setup, load):
    if np.all(setup.stability_diagram != load.stability_diagram): return False
    if np.all(setup.voltages != load.voltages): return False
    if setup.Cg1 != load.Cg1: return False
    if setup.Cg2 != load.Cg2: return False
    if setup.CL != load.CL: return False
    if setup.CR != load.CR: return False
    if setup.parameter_model != load.parameter_model: return False
    if setup.T != load.T: return False
    if setup.Cm != load.Cm: return False
    if setup.kB != load.kB: return False
    if setup.N_min != load.N_min: return False
    if setup.N_max != load.N_max: return False
    if setup.n_dots != load.n_dots: return False
    if setup.verbose != load.verbose: return False
    return True


def test_save_load_everything_hdf5():
    memristor, circuit, pulsed_programming, memristor_sim, quantum_sim = setup_save_load()

    qdms.Log.save_everything_hdf5(path='.//Simulation', directory_name='test_save_load_everything_hdf5', memristor=memristor,
                                  pulsed_programming=pulsed_programming, circuit=circuit, memristor_sim=memristor_sim,
                                  qd_simulation=quantum_sim, verbose=False)

    memristor_load, circuit_load, pulsed_programming_load, memristor_sim_load, quantum_sim_load = qdms.Log.load_everything_hdf5(
        path='.//Simulation//test_save_load_everything_hdf5')
    result = compare_everything([memristor, circuit, pulsed_programming, memristor_sim, quantum_sim], [memristor_load, circuit_load, pulsed_programming_load, memristor_sim_load, quantum_sim_load])

    shutil.rmtree('.//Simulation//test_save_load_everything_hdf5')

    assert result
