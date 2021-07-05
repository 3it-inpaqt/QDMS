import qdms


def test_plot_everything():
    memristor = qdms.Data_Driven()

    circuit = qdms.Circuit(memristor_model=memristor, number_of_memristor=9, is_new_architecture=True, v_in=1e-3,
                           gain_resistance=0, R_L=1)

    pulsed_programming = qdms.PulsedProgramming(circuit, 5, tolerance=2, is_relative_tolerance=True)
    pulsed_programming.simulate()

    memristor_sim = qdms.MemristorSimulation(pulsed_programming)
    memristor_sim.simulate()

    vector = qdms.HelperFunction.limit_vector(memristor_sim.voltages, 0.2, 0.25)
    vector = qdms.HelperFunction.simplify_vector_resolution(vector, 0.0001)

    qd_sim = qdms.QDSimulation(vector)
    qd_sim.simulate()

    qdms.Plot.plot_everything(memristor_sim, qd_sim, pulsed_programming, 'TestData\\Plots', file_output=True, verbose=True)
    assert 1

test_plot_everything()
