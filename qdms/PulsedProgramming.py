import numpy as np
import math


def spread_resistor_list(lrs, hrs, nb_states, number):
    """
    This function help the create_res_states() function

    Parameters
    ----------
    number : int
        The number of distribution to create

    lrs : float
        Low Resistance State (LRS) (Ohm)

    hrs : float
         High Resistance State (HRS) (Ohm)

    nb_states : int
        The number of states used to program the memristor.

    Returns
    ----------
    list_resistor : iterable[float]
        list of resistor (Ohm)

    """
    list_resistor = []
    for j in range(number):
        # This equation can be optimized. It create a number of different resistance values.
        temp = [int(lrs + i * ((hrs -
                lrs) / (nb_states - 1)) +
                ((-1) ** j) * ((j + 1) // 2) * ((hrs - lrs)
                / ((nb_states - 1) * 4 * 2))) for i in range(nb_states)]
        temp[0] = lrs
        temp[-1] = hrs
        list_resistor.append(temp)
    return list_resistor


class PulsedProgramming:
    """
    This class contains all the parameters for the Pulsed programming on a memristor model.
    After initializing the parameters values, start the simulation with self.simulate()

    Parameters
    ----------
    circuit : Circuit.Circuit
        The circuit object.

    nb_states : int
        The number of states wanted in the pulse programming.

    max_voltage : float
        The max voltage (V) of a pulse. If 0, no limit is apply.

    pulse_algorithm : string
        The pulse algorithm use. Those are the available choices (Sources in the methods). Default is 'fabien'.
        'fabien' : Use fabien_convergence()
        'log' : Use a log_convergence()

    res_states : iterable[iterable[float]]
        Contains the targets resistance (Ohm) of the pulsed programming.

    res_states_practical : list of list of float
        Contains the actual resistance (Ohm) obtained by the pulsed programming.

    tolerance : float
        The tolerance_value input is an int that represent the absolute tolerance (Ohm) from the res_states the
        pulsed programming will find. Smaller is more precise, but too small can never converge.

    is_relative_tolerance : bool
        If true, the tolerance_value would be in percentage instead of (Ohm). ex: 10 : if true, 10% : if false, 10 Ohm

    variability_write : iterable[float]
        A gaussian distribution with (mu=0, sigma=variance_write)

    index_variability : int
        Index of the current variability. If over 1000, reset to 0.

    variance_write : float
        Variance of the gaussian distribution on the memristor write. See variability.

    graph_resistance : List[Union[float, int]]
        Contains all resistance of the simulation. It's used in the creation of plots.

    graph_voltages : List[Union[float, int]]
        Contains all voltages of the simulation. It's used in the creation of plots.

    distribution_type : string
        The distribution type can add controlled variability to the voltages output by targeting individual resistance
        values for the memristor's states. A plot of the distribution is output in Resist folder. The possibilities are:
            'linear' : The states are chosen in a linear manner.
            'full_spread' : All the memristor have a different distribution.

    lrs : float
        Low Resistance State (LRS) (Ohm) used for the programming. It should be higher or equal than
        circuit.memristor_model.r_on. Default is r_off.

    hrs : float
         High Resistance State (HRS) (Ohm) used for the programming. It should be lower or equal than
         circuit.memristor_model.r_off. Default is r_off.

    number_of_reading : int
        The number of correct value read before passing to the next state.

    max_pulse : int
        The max number of pulses.
    """

    def __init__(self, circuit, nb_states, pulse_algorithm='fabien', distribution_type='linear', max_voltage=0,
                 tolerance=0, is_relative_tolerance=False, variance_write=0, lrs=None, hrs=None, number_of_reading=1, max_pulse=20000):
        self.circuit = circuit
        self.nb_states = nb_states
        self.distribution_type = distribution_type
        self.pulse_algorithm = pulse_algorithm
        self.lrs = lrs if lrs is not None else circuit.memristor_model.r_on
        self.hrs = hrs if hrs is not None else circuit.memristor_model.r_off
        self.res_states = self.create_res_states()
        self.res_states_practical = []
        self.tolerance = tolerance
        self.max_voltage = max_voltage
        self.is_relative_tolerance = is_relative_tolerance
        self.index_variability = 0
        self.variance_write = variance_write
        self.variability_write = np.random.normal(0, variance_write, 1000)
        self.number_of_reading = number_of_reading
        self.max_pulse = max_pulse

        self.graph_resistance = []
        self.graph_voltages = []

    def __str__(self):
        str_out = "Here is the current parameter list for pulsed programming"
        str_out += "\n-------------------------------------\n"
        str_out += 'Memristor:\t\t\t\t' + str(type(self.circuit.memristor_model)) + '\n'
        str_out += 'Nb_states:\t\t\t\t' + str(self.nb_states) + '\n'
        str_out += 'Tolerance (Ohm):\t\t' + str(self.tolerance) + '\n'
        str_out += 'Variance write:\t' + str(self.variance_write) + '\n'
        str_out += 'Max voltage (V):\t\t' + str(self.max_voltage) + '\n'
        str_out += "-------------------------------------\n"
        return str_out

    def write_resistance(self, memristor, voltage, t_pulse):
        """
        This function change the resistance of the memristor by applying a voltage fo t_pulse.

        Parameters
        ----------
        memristor : Memristor
            The memristor wrote.

        voltage : float
            The voltage (V) applied.

        t_pulse : float
            The time of the writing pulse. (s)

        Returns
        ----------
        """
        t = int(t_pulse / memristor.time_series_resolution)
        signal = [voltage] * t
        memristor.simulate(signal)

        self.index_variability = self.index_variability + 1 if self.index_variability < len(self.variability_write) - 1 else 0
        memristor.g = 1 / (1 / memristor.g + (1 / memristor.g) * self.variability_write[self.index_variability])

    def create_res_states(self):
        """
        This function creates the theoretical resistance distribution according to the distribution_type.

        Parameters
        ----------

        Returns
        ----------
        res_states : list of list of float
            list of resistor (Ohm)

        """
        res_states = []
        if self.distribution_type == 'linear':
            res_states = [[int(self.lrs + i * ((self.hrs - self.lrs) / (self.nb_states - 1))) for i in range(self.nb_states)]]
        elif self.distribution_type == 'half_spread':
            res_states = spread_resistor_list(self.lrs, self.hrs, self.nb_states,
                                              int(math.sqrt(self.circuit.number_of_memristor)))
        elif self.distribution_type == 'full_spread':
            res_states = spread_resistor_list(self.lrs, self.hrs, self.nb_states, self.circuit.number_of_memristor)
        return res_states

    def find_number_iteration(self):
        """
        This function find the number of iteration needed to create the resistance list depending on the distribution type

        Parameters
        ----------

        Returns
        ----------
        number_iteration : int
            number of iteration
        """
        number_iteration = 1
        if self.distribution_type == 'half_spread':
            number_iteration = int(math.sqrt(self.circuit.number_of_memristor))
        elif self.distribution_type == 'full_spread':
            number_iteration = self.circuit.number_of_memristor
        return number_iteration

    def simulate(self):
        """
        This function find all practical resistance (Ohm) depending on the states using fabien_convergence.
        The output is stored in self.res_states_practical

        Parameters
        ----------

        Returns
        ----------
        res_states_practical : list of list of float
            Returns the practical value found.
        """
        if self.pulse_algorithm != 'fabien' and self.pulse_algorithm != 'log':
            print(f'Pulse algorithm not supported: {self.pulse_algorithm}')
            exit(1)
        if self.distribution_type != 'full_spread' and self.distribution_type != 'linear' and self.distribution_type != 'half_spread':
            print(f"Error: distribution type <{self.pulsed_programming.distribution_type}> invalid")
            exit(1)
        if self.distribution_type == 'half_spread' and not is_square(self.circuit.number_of_memristor):
            print(f'Error: distribution type <half_spread> is not compatible with <{self.circuit.number_of_memristor}> memristors')
            exit(1)

        number_of_iteration = self.find_number_iteration()
        self.res_states_practical = [[None for _ in range(self.nb_states)] for _ in range(number_of_iteration)]

        # temp = np.linspace(self.nb_states-1, 0, num=self.nb_states)
        # temp = [int(i) for i in temp]

        for j in range(number_of_iteration):
            for i in range(self.nb_states):
                target_res = self.res_states[j][i]
                if self.pulse_algorithm == 'fabien':
                    self.fabien_convergence(target_res, self.max_pulse)
                elif self.pulse_algorithm == 'log':
                    self.log_convergence(target_res, self.max_pulse)
                self.res_states_practical[j][i] = (self.graph_resistance[-1][0])
            self.circuit.memristor_model.g = 1 / self.circuit.memristor_model.r_on
        return self.res_states_practical

    def log_convergence(self, target_res, max_pulse):
        """
        This function run the pulsed programming with a variable voltage to find the resistance (Ohm)
        for the i_state.
        From : https://arxiv.org/abs/2103.09931

        Parameters
        ----------
        i_state : int
            The target state to find.

        res_states : iterable[float]
            List of target resistance.

        max_pulse : int
            The max number of pulses.

        Returns
        ----------
        """
        positive_voltage = voltage_set = 0.5
        negative_voltage = voltage_reset = -0.5
        # additional parameters
        min_shift = 0.005
        max_shift = 0.2
        a = 0.1

        if self.is_relative_tolerance:
            res_max = target_res + self.tolerance * target_res / 100
            res_min = target_res - self.tolerance * target_res / 100
        else:
            res_max = target_res + self.tolerance
            res_min = target_res - self.tolerance

        counter = len(self.graph_resistance)
        action = 'read'
        flag_finish = False
        counter_read = 0

        # is_setting = False

        r_shift = 1
        current_res = self.circuit.memristor_model.read()
        while not flag_finish:
            if res_min < current_res < res_max:
                action = 'read'
                counter_read += 1
                self.graph_voltages.append([0.2, counter, action])

            elif current_res > res_max:
                action = 'set'
                if r_shift < min_shift * (self.circuit.memristor_model.r_off - self.circuit.memristor_model.r_on):
                    positive_voltage += a * np.log10(abs(target_res - current_res) / r_shift)
                elif r_shift > max_shift * (self.circuit.memristor_model.r_off - self.circuit.memristor_model.r_on):
                    positive_voltage = voltage_set
                if self.max_voltage != 0:
                    positive_voltage = self.max_voltage if positive_voltage >= self.max_voltage else positive_voltage
                self.write_resistance(self.circuit.memristor_model, positive_voltage, 200e-9)
                self.graph_voltages.append([positive_voltage, counter, action])

            elif current_res < res_min:
                action = 'reset'
                if r_shift < min_shift * (self.circuit.memristor_model.r_off - self.circuit.memristor_model.r_on):
                    negative_voltage -= a * np.log10(abs((target_res - current_res) / r_shift))
                elif r_shift > max_shift * (self.circuit.memristor_model.r_off - self.circuit.memristor_model.r_on):
                    negative_voltage = voltage_reset
                if self.max_voltage != 0:
                    negative_voltage = -self.max_voltage if negative_voltage <= -self.max_voltage else negative_voltage
                self.write_resistance(self.circuit.memristor_model, negative_voltage, 200e-9)
                self.graph_voltages.append([negative_voltage, counter, action])

            if counter_read == self.number_of_reading:
                flag_finish = not flag_finish
            if counter >= max_pulse:
                flag_finish = not flag_finish
                print('Got max pulse')

            self.graph_resistance.append([current_res, counter, action, flag_finish])
            counter += 1

            previous_res = current_res
            current_res = self.circuit.memristor_model.read()
            r_shift = abs(current_res - previous_res) if abs(current_res - previous_res) != 0 else 1

    def fabien_convergence(self, target_res, max_pulse):
        """
        This function run the pulsed programming with a variable voltage to find the resistance (Ohm)
        for the i_state.
        From : https://iopscience.iop.org/article/10.1088/0957-4484/23/7/075201

        Parameters
        ----------
        target_res : float
            The target resistance (Ohm)

        max_pulse : int
            The max number of pulses.

        Returns
        ----------
        """
        step = 0.005
        positive_voltage = voltage_set = 0.5
        negative_voltage = voltage_reset = -0.5
        if self.is_relative_tolerance:
            res_max = target_res + self.tolerance * target_res / 100
            res_min = target_res - self.tolerance * target_res / 100
        else:
            res_max = target_res + self.tolerance
            res_min = target_res - self.tolerance

        counter = len(self.graph_resistance)
        action = 'read'
        flag_finish = False
        counter_read = 0

        while not flag_finish:
            current_res = self.circuit.memristor_model.read()

            if res_min <= current_res <= res_max:
                action = 'read'
                counter_read += 1
                self.graph_voltages.append([0.2, counter, action])
            elif current_res < res_min:
                action = 'reset'
                if self.max_voltage != 0:
                    negative_voltage = -self.max_voltage if negative_voltage <= -self.max_voltage else negative_voltage
                self.write_resistance(self.circuit.memristor_model, negative_voltage, 200e-9)
                self.graph_voltages.append([negative_voltage, counter, action])
                negative_voltage -= step
                positive_voltage = voltage_set
            elif current_res > res_max:
                action = 'set'
                if self.max_voltage != 0:
                    positive_voltage = self.max_voltage if positive_voltage >= self.max_voltage  else positive_voltage
                self.write_resistance(self.circuit.memristor_model, positive_voltage, 200e-9)
                self.graph_voltages.append([positive_voltage, counter, action])
                positive_voltage += step
                negative_voltage = voltage_reset

            if counter_read == self.number_of_reading:
                flag_finish = not flag_finish
            if counter >= max_pulse:
                flag_finish = not flag_finish
                print('Got max pulse')
            self.graph_resistance.append([current_res, counter, action, flag_finish])
            counter += 1
