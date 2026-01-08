# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 13:25:49 2025

@author: samuel.delgado
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import pandas as pd
from pathlib import Path

class ElectricalController:
    def __init__(self,initial_voltage=0.0, initial_time = 0.0, series_resistance = 0.0, **kwargs):
        self.voltage = initial_voltage
        self.time = initial_time
        self.current = 0.0
        self.resistance = float('inf')
        self.series_resistance = series_resistance
        self.current_parameters = {}
        self.current_model = kwargs.get('current_model', 'ohmic')  # 'ohmic', 'schottky', 'poole_frenkel', etc.

        self.experimental_data = None
        self.measurements = {
            'time': [],
            'voltage': [],
            'current': [],
            'resistance': []
        }
    
    def apply_ramp_voltage_cycle(self,max_voltage, min_voltage,rate,time,num_cycles=1):
        """
        Apply complete resistive switching cycle with 4 phases:
        1. Increasing from 0 to max_voltage
        2. Decreasing from max_voltage to 0
        3. Decreasing from 0 to min_voltage  
        4. Increasing from min_voltage to 0
        
        Parameters:
        -----------
        max_voltage : float
            Maximum positive voltage
        min_voltage : float
            Minimum negative voltage (negative value)
        rate : float
            Voltage ramp rate (V/s) - same magnitude for all phases
        num_cycles : int
            Number of complete cycles to perform
        """
        # Calculate time for each phase
        t1 = abs(max_voltage) / rate # 0 -> max
        t2 = t1                      # max -> 0 
        t3 = abs(min_voltage) / rate # 0 -> min
        t4 = t3                      # min -> 0
        
        cycle_time = t1 + t2 + t3 + t4
        total_time = cycle_time * num_cycles
        current_time_in_cycle = total_time % cycle_time
        
        if not hasattr(self, '_total_cycles_completed'):
            self._total_cycles_completed = 0
            
        if self._total_cycles_completed >= num_cycles:
            self.voltage = 0.0
            return 0.0
        
        # Calculate current position in current cycle
        current_time_in_cycle = time % cycle_time
        current_cycle = int(time // cycle_time)
        
        # Check if we've moved to next cycle
        if current_cycle > self._total_cycles_completed:
            self._total_cycles_completed = current_cycle
            if current_cycle >= num_cycles:
                self.voltage = 0.0
                return 0.0
            
        # Calculate voltage based on current phase
        if 0 <= current_time_in_cycle <= t1:
            # Phase 1: 0 to max_voltage (positive ramp)
            self.voltage = rate * current_time_in_cycle
        elif t1 <= current_time_in_cycle < t1+t2:
            # Phase 2: max_voltage to 0 (negative ramp)
            self.voltage = max_voltage - rate * (current_time_in_cycle - t1)
        elif t1 + t2 <= current_time_in_cycle < t1 + t2 + t3:
            # Phase 3: 0 to min_voltage (negative ramp)
            self.voltage = - rate * (current_time_in_cycle - t1 - t2)
        else:
            # Phase 4: min_voltage to 0 (positive ramp)
            self.voltage = min_voltage + rate * (current_time_in_cycle - t1 - t2 - t3)
        
        self.time = time
        return self.voltage
        
    def _calculate_effective_device_voltage(self,current):
        
        votage_drop = current * self.series_resistance
        
        effective_voltage = self.voltage - votage_drop
        
        return effective_voltage
        
    def initialize_current_parameters(self,**model_params):
        """
        Initialize parameters for the selected current model
        
        Parameters can include:
        - barrier_height: Schottky barrier height (eV)
        - temperature: device temperature (K)  
        - area: electrode area (m²)
        - epsilon_r: relative permittivity
        - filament_conductivity: for ohmic models
        - etc.
        """
        self.current_parameters.update(model_params)
        
        # Validate parameters based on current model
        if self.current_model == 'schottky':
            required = ['barrier_height', 'temperature', 'area', 'epsilon_r']
            self._validate_parameters(required)
            
            # Richardson constant (A*)
            m_star_ratio = model_params.get('effective_mass_ratio', 1.0) # m*/m_e
            self.current_parameters['Richardson_constant'] = self._calculate_richardson_constant(m_star_ratio)
            
        elif self.current_model == 'ohmic':
            required = ['filament_resistance']
            self._validate_parameters(required)
            
    def _calculate_richardson_constant(self,m_star_ratio = 1.0):
        """
        Calculate Richardson constant A* = (4πem*k²)/(h³)
        
        Parameters:
        -----------
        m_star_ratio : float
            Effective mass ratio (m*/m_e), default=1.0 for free electron mass
        """
        m_star = m_star_ratio * const.m_e
        
        # Richardson constant formula
        # A* = (4 * π * e * m* * k2) / (h3)
        A_star = (4 * const.pi * const.e * m_star * const.k**2) / (const.h**3)
        
        # Convert to A/m2k2 (SI unit)
        return A_star
        
    def _validate_parameters(self,required_params):
        """Validate that required parameters are present"""
        missing = [param for param in required_params if param not in self.current_parameters]
        if missing:
            raise ValueError(f'Missing required parameters for {self.current_model}: {missing}')
    
    def calculate_current(self, voltage = None, effective_gap = None):
        """
        Calculate current based on the selected model and pre-initialized parameters
        """
        if self.current_model == 'ohmic':
            return self._calculate_ohmic_current(voltage)
        elif self.current_model == 'schottky':
            return self._calculate_schottky_current(voltage,effective_gap)
        else:
            raise ValueError(f'Unkown current model: {self.current_model}')
    
    def _calculate_ohmic_current(self,voltage):
        """Ohmic current through filament"""
        params = self.current_parameters
        filament_resistance = params['filament_resistance']
        return voltage/filament_resistance
    
    def _calculate_schottky_current(self,voltage,effective_gap):
        """Schottky emission current"""
        params = self.current_parameters
        A_star = params['Richardson_constant']
        phi_B = params['barrier_height']
        T = params['temperature']
        area = params['area']
        epsilon_r = params['epsilon_r']
        
        electric_field = voltage / effective_gap
        # Schottky barrier lowering: sqrt(qE/4π*epsilon0*εr)
        epsilon = const.epsilon_0 * epsilon_r
        delta_phi_eV = np.sqrt((const.e * abs(electric_field)) / (4 * const.pi * epsilon))
        
        # Effective barrier height
        effective_phi_eV = max(0.01, phi_B - delta_phi_eV)
        
        # Current density: J = A*T2*exp(-effective_phi_J / kT)
        kT = const.physical_constants['Boltzmann constant in eV/K'][0] * T
        J = A_star * T**2 * np.exp(-effective_phi_eV / kT)
        
        self.current = J * area * np.sign(electric_field)
        
        return self.current
    
    def load_experimental_data(self, file_path,voltage = 'AV', current = 'AI'):
        # Read excel
        df = pd.read_csv(
            file_path, sep=None,           # Auto-detect separator
            engine='python',    # More flexible
            skipinitialspace=True,  # Handle spaces after separators
            dtype=str 
        )
        
        def clean_and_convert(series):
            # Remove common problematic characters
            cleaned = series.astype(str).str.strip()
            # Handle comma as decimal separator
            cleaned = cleaned.str.replace(',', '.')
            # Remove any non-numeric characters (except ., -, +, e, E)
            cleaned = cleaned.str.replace(r'[^\d.\-+eE]', '', regex=True)
            # Convert to numeric
            numeric = pd.to_numeric(cleaned, errors='coerce')
            return numeric

        voltage = clean_and_convert(df[voltage])
        current = clean_and_convert(df[current])

        
        self.experimental_data = {
            'voltage': voltage,
            'current': current,
            'file_path': file_path
        }
        
    def get_experimental_current(self,voltage,method='nearest'):
        """
        Get experimental current for a given voltage using interpolation
        
        Parameters:
        -----------
        voltage : float or array-like
            Voltage value(s) to query
        method : str, optional
            Interpolation method ('nearest', 'linear', 'cubic')
            
        Returns:
        --------
        Current value(s) corresponding to the input voltage(s)
        """
        if self.experimental_data is None:
            raise ValueError("No experimental data loaded. Call load_experimental_data() first")
        
        from scipy.interpolate import interp1d
        
        # Create interpolation function
        interp_func = interp1d(
            self.experimental_data['voltage'],
            self.experimental_data['current'],
            kind=method,
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        return interp_func(voltage)
    
    def compare_with_simulation(self,simulated_voltages, simulated_currents):
        """
        Compare simulation results with experimental data
        
        Parameters:
        -----------
        simulated_voltages : array-like
            Simulated voltage values
        simulated_currents : array-like  
            Simulated current values
        """
        # Interpolate experimental current at simulation voltage points
        experimental_currents = self.get_experimental_current(simulated_voltages)
        
        # Handle zero/negative currents (add small offset)
        eps = 1e-20
        sim_currents_safe = np.maximum(simulated_currents, eps)
        exp_currents_safe = np.maximum(experimental_currents, eps)
    
        # Convert to log10
        log_sim = np.log10(sim_currents_safe)
        log_exp = np.log10(exp_currents_safe)
        
        # Calculate metrics
        # Log-space RMSE (LRMSE)
        lrmse = np.sqrt(np.mean((log_sim - log_exp)**2))
        # Log-space MAE (LMAE)
        lmae = np.mean(np.abs(log_sim - log_exp))
        # Mean Absolute Percentage Error (MAPE)
        mape_log = np.mean(np.abs((log_sim - log_exp) / log_exp)) * 100
        
        
        return {
            'lrmse':lrmse,
            'lmae':lmae,
            'mape_log':mape_log
        }
        
    def update_measurements(self):
        """Update current and derived measurements"""
        # Store for analysis
        self.measurements['time'].append(self.time)
        self.measurements['voltage'].append(self.voltage)
        self.measurements['current'].append(self.current)
        
    def plot_V_t(self):
        
        plt.plot(self.measurements['time'], self.measurements['voltage'])
        plt.show()
        
    def plot_V_I(self):

        # Plot experimental and simulated values
        plt.semilogy(self.measurements['voltage'], self.measurements['current'], label='Simulation', linewidth=2)
        plt.semilogy(self.experimental_data['voltage'], self.experimental_data['current'], label='Experiment', marker='o', markersize=3)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.legend()
        plt.title('Experimental vs simulated I-V Curve')
        plt.grid(True, alpha=0.3)
        plt.show()

initial_voltage=0.0 
initial_time = 0.0   
series_resistance = 3e3   
 
Controller = ElectricalController(initial_voltage,initial_time,series_resistance, current_model = 'schottky')

max_voltage = 2.6
min_voltage = -1
rate = 0.1
num_cycles=1

t1 = abs(max_voltage) / rate # 0 -> max
t2 = t1                      # max -> 0 
t3 = abs(min_voltage) / rate # 0 -> min
t4 = t3                      # min -> 0

cycle_time = t1 + t2 + t3 + t4

total_time = cycle_time * num_cycles

time_list = np.linspace(0,t1, int(t1/rate))

Controller.initialize_current_parameters(
    barrier_height=0.53,      # eV
    temperature=300,         # K  
    area= np.pi * (50*1e-6)**2,             # m²
    epsilon_r=25            # HfO2
)

effective_gap = 50*1e-10

filament_rate = int((max_voltage - 1.5) / rate)
array_gap = np.linspace(effective_gap,0,filament_rate)
i = 0
file_path = Path(r'\\FS1\Docs2\samuel.delgado\My Documents\Publications\Memristor ECM\CeO2\I-V curves\Crystalline IV.csv')
Controller.load_experimental_data(file_path)

for time in time_list:
    
    voltage = Controller.apply_ramp_voltage_cycle(max_voltage,min_voltage,rate,time,num_cycles)
    
    if voltage >= 1.8:
        #effective_gap = array_gap[i]
        i+=1
    Controller.calculate_current(voltage, effective_gap)
    Controller.update_measurements()
    
Controller.plot_V_I()

comparison = Controller.compare_with_simulation(Controller.measurements['voltage'], Controller.measurements['current'])
print(f"RMSE: {comparison['lrmse']:.2e}")
print(f"MAE: {comparison['lmae']:.2e}")
print(f"MAPE: {comparison['mape_log']:.2e}")