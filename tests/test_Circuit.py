import qdms


def test_calculate_voltage():
    memristor = qdms.Data_Driven()
    circuit1 = qdms.Circuit(memristor, 3, is_new_architecture=True, R_L=1, gain_resistance=0, v_in=1e-3)
    result = [circuit1.calculate_voltage(1/1000) == 1]

    circuit2 = qdms.Circuit(memristor, 3, is_new_architecture=False, R_L=0, gain_resistance=1, v_in=1)
    result.append(circuit2.calculate_voltage(1/1000) == 0.001)

    for i in result:
        if i:
            continue
        assert False
    assert True

