import math
import numpy as np
from .Memristor import Memristor as Memristor
import random


class Data_Driven(Memristor):
    """A Data-Driven Verilog-A ReRAM Model (https://eprints.soton.ac.uk/411693/).

    Parameters
    ----------
    time_series_resolution : float
        Time series resolution (s).
    r_off : float
        Off (maximum) resistance of the device (ohms).
    r_on : float
        On (minimum) resistance of the device (ohms).
    A_p : float
        A_p model parameter.
    A_n : float
        A_n model parameter.
    t_p : float
        t_p model parameter.
    t_n : float
        t_n model parameter.
    k_p : float
        k_p model parameter.
    k_n : float
        k_n model parameter.
    r_p : list of float
        r_p voltage-dependent resistive boundary function coefficients.
    r_n : list of float
        r_n voltage-dependent resistive boundary function coefficients.
    eta : int
        Switching direction to stimulus polarity.
    a_p : float
        a_p model parameter.
    a_n : float
        a_n model parameter.
    b_p : float
        b_p model parameter.
    b_n : float
        b_n model parameter.
    is_variability_on : bool
        If true, a variability will be apply on reading.
    """

    def __init__(self, parameter_model='O921C', is_variability_on=False):
        self.A_p = self.A_n = self.t_p = self.t_n = self.k_p = self.k_n = self.eta = self.a_p = self.a_n = self.b_p = self.b_n = self.variability_a = self.variability_b = 0
        self.r_p = self.r_n = None
        self.parameter_model = parameter_model.upper()
        r_off, r_on = self.set_parameter()
        super(Data_Driven, self).__init__(
            r_off, r_on, 1e-9, 0, 0
        )
        self.g = 1 / self.r_on
        self.is_variability_on = is_variability_on

    def set_parameter(self):
        if self.parameter_model == 'N1257R':
            r_off = 3590
            r_on = 1000
            self.A_p = 600.100775
            self.A_n = -34.5988399
            self.t_p = -0.0212028
            self.t_n = -0.05343997
            self.k_p = 5.11e-4
            self.k_n = 1.17e-3
            self.r_p = [2699.2336, -672.930205]
            self.r_n = [-1222.10682, -2656.27349]
            self.eta = 1
            self.a_p = 0.32046175
            self.a_n = 0.32046175
            self.b_p = 2.71689828
            self.b_n = 2.71689828
            self.variability_a = 1.672e-5
            self.variability_b = 6.645e-2

        elif self.parameter_model == 'O921C':
            r_off = 12759.285080891
            r_on = 1895.117222714
            self.A_p = 134.391352437019
            self.A_n = -60.836774343766194
            self.t_p = -0.05
            self.t_n = -0.010820376861176992
            self.k_p = 5.11e-4
            self.k_n = 1.17e-3
            self.r_p = [24579.55408725577, -13931.612038277426]
            self.r_n = [26698.390117950614, 13390.241870805712]
            self.eta = 1
            self.a_p = 0.7395554112462469
            self.a_n = 0.7395554112462469
            self.b_p = 1.2384430017628405
            self.b_n = 1.2384430017628405
            self.variability_a = 1.672e-5
            self.variability_b = 6.645e-2

        elif self.parameter_model == 'O921C_4K':
            r_off = 16000
            r_on = 1800
            self.A_p = 257.7128190985076
            self.A_n = -11.403039436044947
            self.t_p = -1.8072909907614225
            self.t_n = -1.553306691442444
            self.k_p = 5.11e-4
            self.k_n = 1.17e-3
            self.r_p = [9225.10091519594, -6342.703123090929]
            self.r_n = [-4585.047127293936, -15957.311142302002]
            self.eta = 1
            self.a_p =  0.33255197429308136
            self.a_n = 0.33255197429308136
            self.b_p = 1.8747884736433298
            self.b_n = 1.8747884736433298
            self.variability_a = 1.672e-5
            self.variability_b = 6.645e-2

        else:
            raise(Exception(f'Parameter model for Data_Driven {self.parameter_model} is unknown.'))

        return r_off, r_on

    def variability(self, res):
        return self.variability_a * res + self.variability_b

    def read(self):
        res = 1 / self.g
        variability = 0
        if self.is_variability_on:
            variability = self.variability(res)
        return np.random.normal(res, variability * res / 100)

    def simulate(self, voltage_signal, return_current=False, version2018=False):
        len_voltage_signal = 1
        try:
            len_voltage_signal = len(voltage_signal)
        except:
            voltage_signal = [voltage_signal]

        if return_current:
            current = np.zeros(len_voltage_signal)

        # np.seterr(all="raise")
        for t in range(0, len_voltage_signal):
            current_ = self.current(voltage_signal[t])
            if version2018:
                self.g = 1 / self.resistance2018(voltage_signal[t])
            else:
                self.g = 1 / self.resistance(voltage_signal[t])
            if return_current:
                current[t] = current_
        if return_current:
            return current
        
    def r_pn2017(self,voltage,a0,a1):
            """Function to return rp(v) or rn(v)
            From the 2017 paper model calculations
            
            Parameters
            ----------
            voltage : float
                The current applied voltage (V).
            a0, a1: float
                The value of a0 or a1

            Returns
            -------
            float
            The rp or rn resistance (Ω).
            """
            return (a0 + a1*voltage)
        
    def s_pn2017(self,voltage,A,t):
        """Function to return sp(v) or sn(v)
        From the 2017 paper model calculations
            
        Parameters
        ----------
        voltage : float
        The current applied voltage (V).
        A, t: float
        The value of model params A (A_p or A_n) or t(t_p or t_n)

        Returns
        -------
        float
        The sp or sn variability.
        """
        return A * (math.exp(abs(voltage)/t) - 1)
        
    def dRdt(self, voltage):
        """Function to return dR/dT
        From the 2017 paper model calculations
            
        Parameters
        ----------
        voltage : float
        The current applied voltage (V).
        a0, a1: float
        The value of a0 or a1

        Returns
        -------
        float
        The derivative with respect to time of the resistance
        """
        
        R = 1/self.g
        if(voltage > 0):
            r_p = self.r_pn2017(voltage,self.r_p[0],self.r_p[1])
            s_p = self.s_pn2017(voltage,self.A_p,self.t_p)
            return s_p * (r_p - R)**2 
        if(voltage <= 0):
            r_n = self.r_pn2017(voltage,self.r_n[0],self.r_n[1])
            s_n = self.s_pn2017(voltage,self.A_n,self.t_n)
            return s_n * (R - r_n)**2 
            
        return 

    def current(self, voltage):
        """Method to determine the current of the model given an applied voltage.

        Parameters
        ----------
        voltage : float
            The current applied voltage (V).

        Returns
        -------
        float
            The observed current (A).
        """
        if voltage > 0:
            return self.a_p * self.g * math.sinh(self.b_p * voltage)
        else:
            return self.a_n * self.g * math.sinh(self.b_n * voltage)
        
    def resistance(self, voltage):
        """Method to determine the resistance of the model given an applied voltage.
        Using the 2017/(2021) model

        Parameters
        ----------
        voltage : float
            The current applied voltage (V).

        Returns
        -------
        float
            The observed resistance (Ω).
        """
        R0 = 1/self.g
        if voltage > 0:
            r_p = self.r_pn2017(voltage, self.r_p[0],self.r_p[1])
            s_p = self.s_pn2017(voltage,self.A_p,self.t_p)
            resistance_ = (R0 + (s_p * r_p * (r_p - R0))*self.time_series_resolution)/(1 + s_p*(r_p - R0)*self.time_series_resolution)
            if resistance_ < r_p:  # Should be < r_p
                return R0
            else:
                return max(min(resistance_, self.r_off), self.r_on)  # Artificially confine the resistance between r_on and r_off

        elif voltage <= 0:
            r_n = self.r_pn2017(voltage, self.r_n[0],self.r_n[1])
            s_n = self.s_pn2017(voltage,self.A_n,self.t_n)
            resistance_ = (R0 + (s_n * r_n * (r_n-R0))*self.time_series_resolution)/(1 + s_n*(r_n-R0)*self.time_series_resolution)
            if resistance_ > r_n:
                return R0
            else:
                return max(min(resistance_, self.r_off), self.r_on)  # Artificially confine the resistance between r_on and r_off

        else:
            return 1 / self.g


    #TODO: code pulse fitting to verify if it fits with experimental data (see Memristor.py)
    # time (or number of pulses) vs resistance. See joao 
        
        # Comment crossbar est simulee, drop de tensions, etc.
        # Le mapping des poids sur les crossbars. 
        
    def resistance2018(self, voltage):
        """Method to determine the resistance of the model given an applied voltage.

        Parameters
        ----------
        voltage : float
            The current applied voltage (V).

        Returns
        -------
        float
            The observed resistance (Ω).
        """

        def r_pn(voltage, r_pn):
            sum = 0
            for m_pn in range(0, len(r_pn)):
                sum += r_pn[m_pn] * (voltage ** (m_pn))

            return sum
        

        if voltage > 0:
            r_pn_eval = r_pn(voltage, self.r_p)
            resistance_ = (
                np.log(
                    np.exp(self.eta * self.k_p * r_pn_eval)
                    + np.exp(
                        -self.eta
                        * self.k_p
                        * (self.A_p * (math.exp(self.t_p * abs(voltage)) - 1))
                        * self.time_series_resolution
                    )
                    * (
                        np.exp(self.eta * self.k_p * (1 / self.g))
                        - np.exp(self.eta * self.k_p * r_pn_eval)
                    )
                )
                / self.k_p
            )
            if resistance_ > self.eta * r_pn_eval:
                return 1 / self.g
            else:
                return max(
                    min(resistance_, self.r_off), self.r_on
                )  # Artificially confine the resistance between r_on and r_off
        elif voltage < 0:
            r_pn_eval = r_pn(voltage, self.r_n)
            resistance_ = (
                -np.log(
                    np.exp(
                        -self.eta * self.k_n * (1 / self.g)
                        + self.eta
                        * self.k_n
                        * (self.A_n * (-1 + np.exp(self.t_n * abs(voltage))))
                        * self.time_series_resolution
                    )
                    - np.exp(-self.eta * self.k_n * r_pn_eval)
                    * (
                        -1
                        + np.exp(
                            self.eta
                            * self.k_n
                            * (self.A_n * (-1 + np.exp(self.t_n * abs(voltage))))
                            * self.time_series_resolution
                        )
                    )
                )
                / self.k_n
            )
            if resistance_ < self.eta * r_pn_eval:
                return 1 / self.g
            else:
                return max(
                    min(resistance_, self.r_off), self.r_on
                )  # Artificially confine the resistance between r_on and r_off
        else:
            return 1 / self.g

    def set_conductance(self, conductance):
        conductance = clip(conductance, 1 / self.r_off, 1 / self.r_on)
        self.g = conductance

    def plot_hysteresis_loop(
        self,
        duration=1e-3,
        voltage_signal_amplitude=1.5,
        voltage_signal_frequency=10e3,
        return_result=False,
    ):
        return super().plot_hysteresis_loop(
            self,
            duration=duration,
            voltage_signal_amplitude=voltage_signal_amplitude,
            voltage_signal_frequency=voltage_signal_frequency,
            return_result=return_result,
        )

    def plot_bipolar_switching_behaviour(
        self,
        voltage_signal_amplitude=1.5,
        voltage_signal_frequency=10e3,
        log_scale=True,
        return_result=False,
    ):
        return super().plot_bipolar_switching_behaviour(
            self,
            voltage_signal_amplitude=voltage_signal_amplitude,
            voltage_signal_frequency=voltage_signal_frequency,
            log_scale=log_scale,
            return_result=return_result,
        )
