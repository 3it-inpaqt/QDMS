import numpy as np
import math
import time


class PulsedProgramming:
    """
    This class contains all the parameters for the Pulsed programming on a memristor model.
    After initializing the parameters values, start the simulation with self.simulate()

    Parameters
    ----------
    max_voltage : float
        The max voltage (V) of a pulse. If 0, no limit is apply.

    pulse_algorithm : string
        The pulse algorithm use. Those are the available choices (Sources in the methods). Default is 'fabien'.
        'fabien' : Use fabien_convergence()
        'log' : Use a log_convergence()

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

    number_of_reading : int
        The number of correct value read before passing to the next state.

    max_pulse : int
        The max number of pulses.
    """

    def __init__(self, memristor_simulation, pulse_algorithm='fabien', max_voltage=0, tolerance=0, is_relative_tolerance=False,
                 variance_write=0, number_of_reading=1, max_pulse=20000, verbose=False, plot_memristor=0):
        self.memristor_simulation = memristor_simulation
        self.pulse_algorithm = pulse_algorithm
        self.tolerance = tolerance
        self.max_voltage = max_voltage
        self.is_relative_tolerance = is_relative_tolerance
        self.variance_write = variance_write
        self.number_of_reading = number_of_reading
        self.max_pulse = max_pulse
        self.verbose = verbose
        self.voltage_output = {}
        self.plot_memristor = plot_memristor

        self.index_variability = 0
        self.variability_write = np.random.normal(0, variance_write, 1000)

        self.graph_resistance = []
        self.graph_voltages = []

    def print(self):
        print(self.pulse_algorithm)
        print(self.tolerance)
        print(self.max_voltage)
        print(self.voltage_output)
        print(self.is_relative_tolerance)
        print(self.variance_write)
        print(self.number_of_reading)
        print(self.max_pulse)
        print(self.verbose)
        print(np.array(self.graph_resistance))
        print(np.array(self.graph_voltages))

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

    def find_number_iteration(self):
        """
        This function find the number of iteration needed to create the resistance list depending on the distribution type

        Returns
        ----------
        number_iteration : int
            number of iteration
        """
        number_iteration = 1
        if self.distribution_type == 'full_spread':
            number_iteration = self.circuit.number_of_memristor
        return number_iteration

    def simulate(self, voltages_target, precision=None):
        """
        This function will set the memristors to the resistance wanted in each voltages_target package.

        Parameters
        ----------
        voltages_target : dict
            dict with keys as voltage and package as list of resistance

        precision : list
            [[macro_tune, is_relative_variability], [fine_tune, is_relative_variability]] for the balance() method.
        """
        if self.pulse_algorithm != 'fabien' and self.pulse_algorithm != 'log':
            raise(Exception(f'Pulse algorithm not supported: {self.pulse_algorithm}'))
        # voltages_target_list = list(voltages_target.keys())
        # resolution = voltages_target_list[1] - voltages_target_list[0]
        index = 1
        conf_done = 0
        start_time = time.time()
        diff_voltage = {}
        for v in list(voltages_target.keys()):
            if index == 1:
                start_time_ = time.time()
            self.simulate_list_memristor(voltages_target[v], precision)
            self.voltage_output[self.memristor_simulation.circuit.current_v_out()] = [i.read() for i in self.memristor_simulation.circuit.list_memristor]
            diff_voltage[abs(v - self.memristor_simulation.circuit.current_v_out())] = [round(1 / np.sum([1/res for res in voltages_target[v]]), 4), round(1 / self.memristor_simulation.circuit.current_conductance(), 4)]
            if index == 50 and self.verbose:
                conf_done += index
                print(f'Conf done: {conf_done}\tTook: {round(time.time() - start_time_, 2)} s\tTime left: {round((time.time() - start_time_) * (len(voltages_target.keys()) - conf_done) / 50, 2)} s')
                index = 0
            index += 1

        if self.verbose:
            print(f'Total time: {time.time() - start_time}')
            print()

            for key in diff_voltage.keys():
                print(f'{round(key*1000, 4)} mV\t{diff_voltage.get(key)[0]}\t{diff_voltage.get(key)[1]} (Ohm)')

            print(f'Mean diff: {np.mean(list(diff_voltage.keys()))}')
            print(f'Min diff: {np.min(list(diff_voltage.keys()))}\tMax diff: {np.max(list(diff_voltage.keys()))}')

        return self.voltage_output

    def simulate_list_memristor(self, list_resistance, precision):
        """
        This function will set the memristors to the resistance wanted list_resistance.

        Parameters
        ----------
        list_resistance : list
            list of the wanted resistance for the memristor.

        precision : list
            [[macro_tune, is_relative_variability], [fine_tune, is_relative_variability]] for the balance() method.
        """
        for i in range(self.memristor_simulation.circuit.number_of_memristor):
            plot = True if i == self.plot_memristor else False
            if self.pulse_algorithm == 'fabien':
                self.fabien_convergence(self.memristor_simulation.circuit.list_memristor[i], list_resistance[i], plot=plot)
            elif self.pulse_algorithm == 'log':
                self.log_convergence(self.memristor_simulation.circuit.list_memristor[i], list_resistance[i], plot=plot)
        self.balance(list_resistance, precision)

    def balance(self, list_resistance, precision):
        """
        This function will set the memristors to the resistance wanted list_resistance.

        Parameters
        ----------
        list_resistance : list
            list of the wanted resistance for the memristor.

        precision : list
            [[macro_tune, is_relative_variability], [fine_tune, is_relative_variability]] for the balance() method. If 0,
            won't do it.
        """
        final_g = np.sum([1 / i for i in list_resistance])
        delta_g = final_g - self.memristor_simulation.circuit.current_conductance()
        for i in range(self.memristor_simulation.circuit.number_of_memristor):
            plot = True if -(i+1) == self.plot_memristor else False
            final_res = 1 / (self.memristor_simulation.circuit.list_memristor[-(i+1)].g + delta_g)
            if self.memristor_simulation.circuit.memristor_model.r_on <= final_res <= self.memristor_simulation.circuit.memristor_model.r_off:
                p_tolerance, p_relative = self.tolerance, self.is_relative_tolerance
                # print(f'{final_res}\t{1 / self.memristor_simulation.circuit.list_memristor[-(i+1)].g}\t{final_g - self.memristor_simulation.circuit.current_conductance()}')

                if precision[0][0] != 0 or precision is not None:
                    self.tolerance, self.is_relative_tolerance = precision[0][0], precision[0][1]
                    self.fabien_convergence(self.memristor_simulation.circuit.list_memristor[-(i+1)], final_res, plot)
                    # print(f'{final_res}\t{1 / self.memristor_simulation.circuit.list_memristor[-(i+1)].g}\t{final_g - self.memristor_simulation.circuit.current_conductance()}')

                if precision[1][0] != 0 or precision is not None:
                    self.tolerance, self.is_relative_tolerance = precision[1][0], precision[1][1]
                    self.small_convergence(self.memristor_simulation.circuit.list_memristor[-(i+1)], final_res, plot)
                    # print(f'{final_res}\t{1 / self.memristor_simulation.circuit.list_memristor[-(i+1)].g}\t{final_g - self.memristor_simulation.circuit.current_conductance()}')

                self.tolerance, self.is_relative_tolerance = p_tolerance, p_relative
                break

    def small_convergence(self, memristor, target_res, plot=False):
        """
        This function run the pulsed programming with a variable voltage to set the target_res for the memristor with a
        really small increment.

        Parameters
        ----------
        memristor : Memristor
            The memristor object

        target_res : float
            The target resistance
        """
        step = 0.001
        positive_voltage = voltage_set = 0.1
        negative_voltage = voltage_reset = -0.1
        if self.is_relative_tolerance:
            res_max = target_res + self.tolerance * target_res / 100
            res_min = target_res - self.tolerance * target_res / 100
        else:
            res_max = target_res + self.tolerance
            res_min = target_res - self.tolerance

        start_len_res = len(self.graph_resistance)
        start_len_v = len(self.graph_voltages)
        counter = 0
        action = 'read'
        flag_finish = False
        counter_read = 0

        while not flag_finish:
            current_res = memristor.read()

            if res_min <= current_res <= res_max:
                counter_read += 1
                if plot:
                    action = 'read'
                    self.graph_voltages.append([0.2, counter + start_len_v, action])
            elif current_res < res_min:
                if self.max_voltage != 0:
                    negative_voltage = -self.max_voltage if negative_voltage <= -self.max_voltage else negative_voltage
                self.write_resistance(memristor, negative_voltage, 200e-9)
                if plot:
                    action = 'reset'
                    self.graph_voltages.append([negative_voltage, counter + start_len_v, action])
                negative_voltage -= step
                positive_voltage = voltage_set
            elif current_res > res_max:
                if self.max_voltage != 0:
                    positive_voltage = self.max_voltage if positive_voltage >= self.max_voltage else positive_voltage
                self.write_resistance(memristor, positive_voltage, 200e-9)
                if plot:
                    action = 'set'
                    self.graph_voltages.append([positive_voltage, counter + start_len_v, action])
                positive_voltage += step
                negative_voltage = voltage_reset

            if counter_read == self.number_of_reading:
                flag_finish = not flag_finish
            if counter >= self.max_pulse:
                flag_finish = not flag_finish
                print(f'Got max pulse {self.max_pulse}')
            if plot:
                self.graph_resistance.append([current_res, counter + start_len_res, action, flag_finish])
            counter += 1

    def log_convergence(self, memristor, target_res, plot=False):
        """
        This function run the pulsed programming with a variable voltage to set the target_res for the memristor.
        From : https://arxiv.org/abs/2103.09931

        Parameters
        ----------
        memristor : Memristor
            The memristor object

        target_res : float
            The target resistance
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

        start_len_res = len(self.graph_resistance)
        start_len_v = len(self.graph_voltages)
        counter = 0
        action = 'read'
        flag_finish = False
        counter_read = 0

        r_shift = 1
        current_res = memristor.read()
        while not flag_finish:
            if res_min < current_res < res_max:
                counter_read += 1
                if plot:
                    action = 'read'
                    self.graph_voltages.append([0.2, counter + start_len_v, action])

            elif current_res > res_max:
                if r_shift < min_shift * (memristor.r_off - memristor.r_on):
                    positive_voltage += a * np.log10(abs(target_res - current_res) / r_shift)
                elif r_shift > max_shift * (memristor.r_off - memristor.r_on):
                    positive_voltage = voltage_set
                if self.max_voltage != 0:
                    positive_voltage = self.max_voltage if positive_voltage >= self.max_voltage else positive_voltage
                self.write_resistance(memristor, positive_voltage, 200e-9)
                if plot:
                    action = 'set'
                    self.graph_voltages.append([positive_voltage, counter + start_len_v, action])

            elif current_res < res_min:
                if r_shift < min_shift * (memristor.r_off - memristor.r_on):
                    negative_voltage -= a * np.log10(abs((target_res - current_res) / r_shift))
                elif r_shift > max_shift * (memristor.r_off - memristor.r_on):
                    negative_voltage = voltage_reset
                if self.max_voltage != 0:
                    negative_voltage = -self.max_voltage if negative_voltage <= -self.max_voltage else negative_voltage
                self.write_resistance(memristor, negative_voltage, 200e-9)
                if plot:
                    action = 'reset'
                    self.graph_voltages.append([negative_voltage, counter + start_len_v, action])

            if counter_read == self.number_of_reading:
                flag_finish = not flag_finish
            if counter >= self.max_pulse:
                flag_finish = not flag_finish
                print('Got max pulse')

            if plot:
                self.graph_resistance.append([current_res, counter + start_len_res, action, flag_finish])
            counter += 1

            previous_res = current_res
            current_res = memristor.read()
            r_shift = abs(current_res - previous_res) if abs(current_res - previous_res) != 0 else 1

    def fabien_convergence(self, memristor, target_res, plot=False):
        """
        This function run the pulsed programming with a variable voltage to set the target_res for the memristor.
        From : https://iopscience.iop.org/article/10.1088/0957-4484/23/7/075201

        Parameters
        ----------
        memristor : Memristor
            The memristor object

        target_res : float
            The target resistance
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

        start_len_res = len(self.graph_resistance)
        start_len_v = len(self.graph_voltages)
        counter = 0
        action = 'read'
        flag_finish = False
        counter_read = 0

        while not flag_finish:
            current_res = memristor.read()

            if res_min <= current_res <= res_max:
                counter_read += 1
                if plot:
                    action = 'read'
                    self.graph_voltages.append([0.2, counter + start_len_v, action])
            elif current_res < res_min:
                if self.max_voltage != 0:
                    negative_voltage = -self.max_voltage if negative_voltage <= -self.max_voltage else negative_voltage
                self.write_resistance(memristor, negative_voltage, 200e-9)
                if plot:
                    action = 'reset'
                    self.graph_voltages.append([negative_voltage, counter + start_len_v, action])
                negative_voltage -= step
                positive_voltage = voltage_set
            elif current_res > res_max:
                if self.max_voltage != 0:
                    positive_voltage = self.max_voltage if positive_voltage >= self.max_voltage  else positive_voltage
                self.write_resistance(memristor, positive_voltage, 200e-9)
                if plot:
                    action = 'set'
                    self.graph_voltages.append([positive_voltage, counter + start_len_v, action])
                positive_voltage += step
                negative_voltage = voltage_reset

            if counter_read == self.number_of_reading:
                flag_finish = not flag_finish
            if counter >= self.max_pulse:
                flag_finish = not flag_finish
                print('Got max pulse')
            if plot:
                self.graph_resistance.append([current_res, counter + start_len_res, action, flag_finish])
                # print(f'{self.graph_resistance[-1]}\t{self.graph_voltages[-1]}')
            counter += 1
