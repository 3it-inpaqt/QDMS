import time
from .coulomb_blockade import *


class QDSimulation:
    """
    This class contains all the parameters for the quantum dot simulation using coulomb_blockade.py.
    After initializing the parameters values, start the simulation with self.simulate()

    Parameters
    ----------
    stability_diagram : list of list of float
        Contains the number of electron depending the voltage in x or y.

    parameter_model : string
        Parameter model set Cg1, Cg2, CL and CR according to existing quantum dots. UNSW is the default.
        Here are the implemented model:
            'UNSW' : http://unsworks.unsw.edu.au/fapi/datastream/unsworks:42863/SOURCE02?view=true#page=172&zoom=100,160,768
                Range: x = y = (0, 0.05)
            'QuTech' : https://www.nature.com/articles/s41586-021-03469-4
                Range: x = y = (0, 0.15)
            'Princeton' : https://dataspace.princeton.edu/bitstream/88435/dsp01f4752k519/1/Zajac_princeton_0181D_12764.pdf
                Range: x = y = (0, 0.035)
            'Sandia_national_lab' : https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7838537
                Range: x = y = (0, 0.4)
            'CEA_LETI' : https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.14.024066
                Range: x = (0, 0.09) ; y = (0, 0.045)
            'UCL' : https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.010353
                Range: x = y = (0, 0.95)

    voltages : Iterable[float]
        Contains the voltages (V) from the memristor simulation.

    Cg1 : float
        Gate 1 capacitance (F).

    Cg2 : float
        Gate 2 capacitance (F).

    CL : float
        Ratio of the left capacitance (F).

    CR : float
        Ratio of the right capacitance (F).

    T : float
        Temperature (K)

    N_max : int
        Maximum number of electron

    n_dots : int
        Number of dots.
        Accepted : 1 or 2

    Cm : float
        Ratio of the cross capacitance.

    verbose : bool
        If true, output in console of time.

    """

    def __init__(self, voltages_x, voltages_y, n_dots=2, T=0.1, Cm=0.4, parameter_model='UNSW', verbose=True):
        if n_dots != 2:
            raise Exception(f'{n_dots} is not supported. Use 2 instead.')
        self.stability_diagram = [[None for _ in range(len(voltages))] for _ in range(len(voltages))]
        self.voltages_x = voltages_x
        self.voltages_y = voltages_y

        self.Cg1 = 0
        self.Cg2 = 0
        self.CL = 0
        self.CR = 0
        self.parameter_model = parameter_model
        self.set_parameter_model(parameter_model)

        self.T = T
        self.Cm = Cm
        self.kB = 1.381e-23
        self.N_min, self.N_max = self.find_number_electron()
        self.n_dots = n_dots
        self.verbose = verbose

    def print(self):
        print(np.array(self.stability_diagram))
        print(np.array(self.voltages_x))
        print(np.array(self.voltages_y))
        print(self.Cg1)
        print(self.Cg2)
        print(self.CL)
        print(self.CR)
        print(self.T)
        print(self.Cm)
        print(self.kB)
        print(self.N_min)
        print(self.N_max)
        print(self.n_dots)
        print(self.verbose)

    def set_parameter_model(self, parameter_model):
        self.parameter_model = parameter_model
        if self.parameter_model == 'UNSW':
            self.Cg1 = 10.3e-18
            self.Cg2 = self.Cg1
            self.CL = 5 * self.Cg1
            self.CR = self.CL
        elif self.parameter_model == 'QuTech':
            self.Cg1 = 5.80e-18
            self.Cg2 = 4.56e-18
            self.CL = 2.5 * self.Cg1
            self.CR = 2.7 * self.Cg2
        elif self.parameter_model == 'Princeton':
            self.Cg1 = 24.3e-18
            self.Cg2 = self.Cg1
            self.CL = 0.08 * self.Cg1
            self.CR = self.CL
        elif self.parameter_model == 'Sandia_national_lab':
            self.Cg1 = 1.87e-18
            self.Cg2 = self.Cg1
            self.CL = 1.7*self.Cg1
            self.CR = self.CL
        elif self.parameter_model == 'CEA_LETI':
            self.Cg1 = 10.3e-18
            self.Cg2 = 19.7e-18
            self.CL = 0.1 * self.Cg1
            self.CR = 0.2 * self.Cg2
        elif self.parameter_model == 'UCL':
            self.Cg1 = 9.1e-19
            self.Cg2 = self.Cg1
            self.CL = 2.2 * self.Cg1
            self.CR = self.CL
        else:
            raise Exception(f'Parameter model {self.parameter_model} not supported.')

    def simulate(self):
        """
        Function to simulate the number of electron depending on the voltages.
        The output is stored in self.stability_diagram

        Parameters
        ----------

        Returns
        ----------

        """
        if self.verbose:
            print()
            print("##########################")
            print(f"Start QD simulation with {len(self.voltages_x)} voltages and {self.N_max} electrons")

        if self.n_dots == 2:
            x, y = np.meshgrid(self.voltages_x, self.voltages_y)
            self.stability_diagram = N_moy_DQD(x, y, Cg1=self.Cg1, Cg2=self.Cg2, Cm=self.Cm * (self.Cg1+self.Cg2)/2,
                                                CL=self.CL, CR=self.CR, N_min=self.N_min, N_max=self.N_max,
                                                kBT=2 * self.kB * self.T, e=1.602e-19, verbose=self.verbose)

    def find_number_electron(self):
        if self.parameter_model == 'UNSW':
            return int(min(self.voltages) * 65), int(max(self.voltages) * 65)

        elif self.parameter_model == 'QuTech':
            return int(min(self.voltages) * 30), int(max(self.voltages) * 35)

        elif self.parameter_model == 'Princeton':
            return int(min(self.voltages) * 155), int(max(self.voltages) * 155)

        elif self.parameter_model == 'Sandia_national_lab':
            return int(min(self.voltages) * 10), int(max(self.voltages) * 15)

        elif self.parameter_model == 'CEA_LETI':
            return int(min(self.voltages) * 60), int(max(self.voltages) * 125)

        elif self.parameter_model == 'UCL':
            return int(min(self.voltages) * 1), int(max(self.voltages) * 10)
