import numpy as np
import qdms
import math


def test_set_parameter_model():
    x = np.linspace(0, 0.1, num=100)
    qd_sim = qdms.QDSimulation(x)
    result = []
    parameter_models = ['UNSW', 'QuTech', 'Princeton', 'Sandia_national_lab', 'CEA_LETI', 'UCL', 'WrongParameter']

    def qd_sim_values(qd_sim):
        return [qd_sim.Cg1, qd_sim.Cg2, qd_sim.CL, qd_sim.CR, qd_sim.parameter_model]

    for parameter_model in parameter_models:
        qd_sim.set_parameter_model(parameter_model)
        if parameter_model == 'UNSW':
            result.append([10.3e-18, 10.3e-18, 5 * 10.3e-18, 5 * 10.3e-18, 'UNSW'] == qd_sim_values(qd_sim))
        if parameter_model == 'QuTech':
            result.append([5.80e-18, 4.56e-18, 2.5 * 5.80e-18, 2.7 * 4.56e-18, 'QuTech'] == qd_sim_values(qd_sim))
        if parameter_model == 'Princeton':
            result.append([24.3e-18, 24.3e-18, 0.08 * 24.3e-18, 0.08 * 24.3e-18, 'Princeton'] == qd_sim_values(qd_sim))
        if parameter_model == 'Sandia_national_lab':
            result.append([1.87e-18, 1.87e-18, 1.7 * 1.87e-18, 1.7 * 1.87e-18, 'Sandia_national_lab'] == qd_sim_values(qd_sim))
        if parameter_model == 'CEA_LETI':
            result.append([10.3e-18, 19.7e-18, 0.1 * 10.3e-18, 0.2 * 19.7e-18, 'CEA_LETI'] == qd_sim_values(qd_sim))
        if parameter_model == 'UCL':
            result.append([9.1e-19, 9.1e-19, 2.2 * 9.1e-19, 2.2 * 9.1e-19, 'UCL'] == qd_sim_values(qd_sim))
        if parameter_model == 'WrongParameter':
            result.append([10.3e-18, 10.3e-18, 5 * 10.3e-18, 5 * 10.3e-18, 'UNSW'] == qd_sim_values(qd_sim))
    for i in result:
        if i:
            continue
        assert False
    assert True


def test_simulate():
    x = np.linspace(0, 0.1, num=100)
    qd_sim = qdms.QDSimulation(x)
    qd_sim.simulate()
    result = []
    for i in [0, 20, 40, 60, 80, 99]:
        result.append(round(qd_sim.stability_diagram[i][0]))
        result.append(round(qd_sim.stability_diagram[i][i]))
        result.append(round(qd_sim.stability_diagram[0][i]))
    assert result == [0, 0, 0, -1, 0, 1, -3, 0, 3, -4, 0, 4, -5, 0, 5, -6, 0, 6]


def test_find_number_electron():
    x = np.linspace(0, 1, num=20)
    qd_sim = qdms.QDSimulation(x)
    qd_sim.N_max = 65
    qd_sim.simulate()
    c1 = math.isnan(qd_sim.stability_diagram[0][0])
    c2 = math.isnan(qd_sim.stability_diagram[-1][0])
    c3 = math.isnan(qd_sim.stability_diagram[0][-1])
    c4 = math.isnan(qd_sim.stability_diagram[-1][-1])

    if not c1 and not c2 and not c3 and not c4:
        assert True
    else:
        assert False

