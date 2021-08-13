import qdms


def test_simulate():
    memristor = qdms.Data_Driven()

    circuit = qdms.Circuit(memristor, 4)

    pulsed_programming = qdms.PulsedProgramming(circuit, 6, tolerance=1)
    pulsed_programming.simulate()

    memristor_sim = qdms.MemristorSimulation(pulsed_programming)
    memristor_sim.simulate()

    memristor_sim_load = qdms.Log.load_memristor_simulation_hdf5('./tests/TestData/memristor_sim_data.hdf5', pulsed_programming)

    assert memristor_sim_load.voltages == memristor_sim.voltages
