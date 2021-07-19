import math
import time
import numpy as np
from .PulsedProgramming import PulsedProgramming


class MemristorSimulation:
    """
    This class contains all the parameters for the memristor simulation.
    After initializing the parameters values, start the simulation with self.simulate()

    Parameters
    ----------
    pulsed_programming : PulsedProgramming.PulsedProgramming
       The pulsed programming of the circuit that will be simulated.

    is_using_conductance : bool
        If true, the distribution of the resistance used will be found with conductance instead of resistance. Linear
        distribution will therefore be non-linear.

    voltages : list of float
        Contains all the voltages output by the simulation (V)

    resistances : list of float
        Contains the resistances link to the voltage.

    verbose : bool
        If true, the simulation will output parameter description and timer on how much time left to the simulation.

    timers : list of float
        Contains the different timers of the simulation. (s)

    start_loop_rec, start_inner_loop_rec, self.end_loop_rec, end_inner_loop_rec : float
        Different timers used in the recursive loop (s)

    list_resistance : list of list of float
        Contains the different resistance used in the simulation. (Ohm)
        ex: [[10, 20, 30], [10, 25, 30], [10, 15, 30]]
    """
    def __init__(self, pulsed_programming, is_using_conductance=False, verbose=False):
        if not isinstance(pulsed_programming, PulsedProgramming):
            print(f'Error: pulsed programming object is not from PulsedProgramming class.')
            exit(1)
        self.pulsed_programming = pulsed_programming
        self.is_using_conductance = is_using_conductance
        self.voltages = []
        self.voltages_memristor = {}
        self.resistances = []
        self.verbose = verbose
        self.resolution = 0
        self.std = 0

        # Inner parameters
        self.timers = []  # [List_resist, loop_rec, sorting_cutting, creating_plots, total]
        self.start_loop_rec = 0
        self.start_inner_loop_rec = 0
        self.end_loop_rec = 0
        self.end_inner_loop_rec = 0
        self.list_resistance = []

    def __str__(self):
        str_out = "Here is the current parameter list for the memristor simulation"
        str_out += "\n-------------------------------------\n"
        str_out += 'Is using conductance:\t\t' + str(self.is_using_conductance) + '\n'
        str_out += "-------------------------------------\n"
        return str_out

    def presentation(self):
        """
        This function print the main parameters of the memristor simulation.

        Parameters
        ----------

        Returns
        ----------

        """
        print("-------------------------")
        print('Conductance:\t\t' + str(self.is_using_conductance))
        print('New architecture:\t' + str(self.pulsed_programming.circuit.is_new_architecture))
        print('Number of memristor: ' + str(self.pulsed_programming.circuit.number_of_memristor))
        print('Memristor model:\t' + str(type(self.pulsed_programming.circuit.memristor_model)))
        # print('Number of states:\t' + str(self.circuit.memristor_model.nb_states))
        print("-------------------------")

    def simulate(self):
        """
        Function to simulate all the possible voltages of the circuit.

        Parameters
        ----------

        Returns
        ----------
        self.voltages : list of float
            Contains all the possible voltages
        """

        if self.verbose:
            self.presentation()
        self.list_resistance = self.pulsed_programming.res_states_practical
        current_states = []
        for _ in range(self.pulsed_programming.circuit.number_of_memristor):
            current_states.append(-1)

        if self.verbose:
            timer_start = time.time()
            self.start_loop_rec = time.time()
        self.loop_rec(self.list_resistance, self.pulsed_programming.circuit.number_of_memristor, current_states)
        if self.verbose:
            timer_end = time.time()
            self.timers.append(timer_end-timer_start)
            # print(f'Simulation timer loop_rec: {self.timers[0]} s')
        if self.verbose:
            timer_start = time.time()
        temp = []
        for index in range(len(self.voltages)):
            temp.append([self.voltages[index], self.resistances[index]])
        temp = list(np.unique(temp, axis=0))
        self.voltages, self.resistances = zip(*temp)
        self.voltages = list(self.voltages)
        self.resistances = list(self.resistances)

        # self.voltages = list(np.unique(self.voltages))
        # self.resistances = list(np.unique(self.resistances))
        if self.verbose:
            timer_end = time.time()
            self.timers.append(timer_end-timer_start)

        self.resolution = np.mean(np.diff(self.voltages))
        self.std = np.std(np.diff(self.voltages))
        return self.voltages

    def add_voltage(self, list_resistance, current_states):
        """
        This function calculate the resistance or conductance depending on the simulation parameters
        and calculate one output voltage of the circuit.

        Parameters
        ----------
        list_resistance : iterable[float][float]
            The list of all resistances (Ohm)

        current_states : list of int
            Contains the states of all loops from the recursive loop.

        Returns
        ----------

        """
        current_res = []
        conductance = 0
        j = 0
        for i in current_states:
            if self.is_using_conductance:
                conductance += list_resistance[j][i]
            else:
                conductance += 1 / list_resistance[j][i]
            current_res.append(list_resistance[j][i])
            if self.pulsed_programming.distribution_type == 'linear':
                j = 0
            elif self.pulsed_programming.distribution_type == 'full_spread':
                j += 1
        for i in range(self.pulsed_programming.circuit.number_of_memristor):
            self.pulsed_programming.circuit.list_memristor[i].g = 1 / current_res[i]
        voltage = self.pulsed_programming.circuit.calculate_voltage(conductance)
        self.voltages_memristor[voltage] = [1/i.g for i in self.pulsed_programming.circuit.list_memristor]
        self.voltages.append(voltage)
        self.resistances.append(1 / conductance)

    def loop_rec(self, list_resistance, counter, current_states):
        """
        This function is a recursive loop which calls a number of loops equal to the amount of memristor in the circuit.

        Parameters
        ----------
        list_resistance : list of float
            The list of all resistances (Ohm)

        counter : int
            Keep count of the number of iteration

        current_states : iterable[int]
            Contains the states of all loops from the recursive loop.

        Returns
        ----------
        self.voltages : list of float
            Contains all the possible voltages (V)
        """
        if counter == self.pulsed_programming.circuit.number_of_memristor-1 and self.verbose:
            self.end_loop_rec = time.time()
            self.end_inner_loop_rec = time.time()
            print(f'Total time elapsed: {round(self.end_loop_rec - self.start_loop_rec, 2)}s; '
                  f'Loop time: {round(self.end_inner_loop_rec - self.start_inner_loop_rec, 2)}s')
            loop_remaining = self.pulsed_programming.nb_states - current_states[counter]
            print(f'Loops remaining: {loop_remaining}; Expected remaining time: '
                  f'{round(loop_remaining * (self.end_inner_loop_rec - self.start_inner_loop_rec), 2)}s')
            self.start_inner_loop_rec = time.time()
        if counter >= 1:
            for x in range(self.pulsed_programming.nb_states):
                current_states[counter-1] = x
                self.loop_rec(list_resistance, counter - 1, current_states)
        else:
            self.add_voltage(list_resistance, current_states)

        return self.voltages
