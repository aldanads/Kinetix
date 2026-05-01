# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 13:25:49 2025

@author: samuel.delgado
"""
from typing import Dict, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import pandas as pd
from pathlib import Path

from kinetix.configs.electrical_config import VoltageMode, CurrentModel, VoltageConfig, CurrentConfig, ElectricalConfig

ANGSTROM_TO_METER = 1e-10

class ElectricalController:
    def __init__(self,initial_voltage=0.0, initial_time = 0.0, series_resistance = 0.0, **kwargs):
        self.voltage = initial_voltage
        self.time = initial_time
        self.current = 0.0
        self.resistance = float('inf')
        self.series_resistance = series_resistance
        self.current_parameters = {}
        self.current_model = kwargs.get('current_model', 'ohmic')  # 'ohmic', 'schottky', 'poole_frenkel', etc. 
        self.crystal_size = kwargs.get('crystal_size')
        self.thickness_m = self.crystal_size[2] * ANGSTROM_TO_METER
        self.voltage_mode = None
        self._voltage_cycle_initialized = False

        self.experimental_data = None
        self.measurements = {
            'time': [],
            'voltage': [],
            'current': [],
            'resistance': []
        }
        

        
    @classmethod
    def from_config(cls, config):
        """
        Create ElectricalController from ElectricalConfig dataclass.
        
        Usage:
            config = ElectricalConfig(...)
            controller = ElectricalController.from_config(config)
        """
        controller = cls(
          initial_voltage=config.initial_voltage,
          initial_time=config.initial_time,
          series_resistance=config.series_resistance,
          crystal_size=config.crystal_size
        )
    
        # Initialize voltage protocol based on mode
        if config.voltage.mode == VoltageMode.RAMP_CYCLE:
          controller.initialize_ramp_voltage_cycle(
            max_voltage=config.voltage.max_voltage,
            min_voltage=config.voltage.min_voltage,
            rate=config.voltage.ramp_rate,
            voltage_update_time=config.voltage.voltage_update_time,
            num_cycles=config.voltage.num_cycles
          )
        elif config.voltage.mode == VoltageMode.ZERO_HOLD:
          controller.initialize_zero_voltage_hold(
            total_time=config.voltage.total_time,
            voltage_update_time=config.voltage.voltage_update_time  
          )
        elif config.voltage.mode == VoltageMode.CONSTANT:
          controller.initialize_constant_voltage(
            voltage=config.voltage.constant_voltage,
            total_time=config.voltage.total_time,
            voltage_update_time=config.voltage.voltage_update_time
          )
        
        # Initialize current parameters
        if config.current is not None:
          controller.initialize_current_parameters(
            model=config.current.model.name.lower(),
            barrier_height=config.current.barrier_height,
            temperature=config.current.temperature,
            area=config.current.area,
            epsilon_r=config.current.epsilon_r
          )
          controller.current_enabled = True
        else:
          controller.current_enabled = False
          controller.current_model = None
        
        return controller
    
    def apply_voltage(self, time: float) -> float:
        """
        Get the applied external voltage at the given simulation time.
        
        This is the ONLY method the kMC simulator should call.
        It dispatches to the appropriate profile handler based on voltage_mode.
        
        Parameters:
        -----------
        time : float
            Current simulation time (seconds)
            
        Returns:
        --------
        float
            Applied voltage (Volts)
        """
        if not self._voltage_cycle_initialized:
            raise RuntimeError(
              "Voltage protocol not initialized. Call one of:\n"
              "  - initialize_ramp_voltage_cycle()\n"
              "  - initialize_zero_voltage_hold()\n"
              "  - initialize_constant_voltage()"
            )
            
        self.time = time
        
        # Select correct handler based on active mode
        if self.voltage_mode == VoltageMode.RAMP_CYCLE:
          return self._apply_ramp_voltage_cycle(time)
        elif self.voltage_mode == VoltageMode.ZERO_HOLD:
          return self._apply_zero_voltage_hold(time)
        elif self.voltage_mode == VoltageMode.CONSTANT:
          return self._apply_constant_voltage(time)
        else:
          return 0.0  # Fallback for NONE or unknown modes
        
    # =========================================================================
    # INITIALIZATION METHODS: Configure a voltage profile
    # =========================================================================
    def initialize_ramp_voltage_cycle(self,max_voltage, min_voltage,rate,voltage_update_time,num_cycles=1):
        """
        Initialize the voltage ramp cycle parameters (call once at simulation start)
        
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
        self._ramp_max_voltage = max_voltage
        self._ramp_min_voltage = min_voltage
        self._ramp_rate = rate
        self._ramp_num_cycles = num_cycles
        
        # Pre-calculate time for each phase (static)
        self._ramp_t1 = abs(max_voltage) / rate # 0 -> max
        self._ramp_t2 = self._ramp_t1           # max -> 0
        self._ramp_t3 = abs(min_voltage) / rate # 0 -> min
        self._ramp_t4 = self._ramp_t3           # min -> 0
        
        self._ramp_cycle_time = self._ramp_t1 + self._ramp_t2 + self._ramp_t3 + self._ramp_t4
        self._ramp_total_time = self._ramp_cycle_time * num_cycles
        self.total_simulation_time = self.time + self._ramp_total_time
        
        # Initialize tracking variables
        self._total_cycles_completed = 0
        self.voltage_update_time = voltage_update_time
        
        self.voltage_mode = VoltageMode.RAMP_CYCLE
        self._voltage_cycle_initialized = True
       
        
    def initialize_zero_voltage_hold(self, total_time: float, voltage_update_time: float):
        """
        Initialize a zero-voltage (relaxation) simulation mode.
        
        Parameters:
        -----------
        total_time : float
            Duration to maintain 0V (seconds)
        voltage_update_time : float
            Measurement interval (for consistency with ramp mode)
        """
        self.voltage_update_time = voltage_update_time
        self.total_simulation_time = self.time + total_time
        self.voltage_mode = VoltageMode.ZERO_HOLD
        self._voltage_cycle_initialized = True
        
    def initialize_constant_voltage(self, voltage: float, total_time:float, voltage_update_time: float):
        self._constant_voltage_value = voltage
        self.total_simulation_time = self.time + total_time
        self.voltage_update_time = voltage_update_time
        self.voltage_mode = VoltageMode.CONSTANT
        self._voltage_cycle_initialized = True
        
    # =========================================================================
    # INTERNAL HANDLERS: Do the actual voltage calculation for each mode
    # =========================================================================    
    def _apply_ramp_voltage_cycle(self,time):
        """
        Calculate voltage at given time (call every simulation step)
        """ 
        if self._total_cycles_completed >= self._ramp_num_cycles:
            self.voltage = 0.0
            return 0.0
        
        # Calculate current position in current cycle
        current_time_in_cycle = time % self._ramp_cycle_time
        current_cycle = int(time // self._ramp_cycle_time)
        
        # Check if we've moved to next cycle
        if current_cycle > self._total_cycles_completed:
            self._total_cycles_completed = current_cycle
            if current_cycle >= self._ramp_num_cycles:
                self.voltage = 0.0
                return 0.0
            
        # Calculate voltage based on current phase
        if 0 <= current_time_in_cycle <= self._ramp_t1:
            # Phase 1: 0 to max_voltage (positive ramp)
            self.voltage = self._ramp_rate * current_time_in_cycle
        elif self._ramp_t1 <= current_time_in_cycle < self._ramp_t1 + self._ramp_t2:
            # Phase 2: max_voltage to 0 (negative ramp)
            self.voltage = self._ramp_max_voltage - self._ramp_rate * (current_time_in_cycle - self._ramp_t1)
        elif self._ramp_t1 + self._ramp_t2 <= current_time_in_cycle < self._ramp_t1 + self._ramp_t2 + self._ramp_t3:
            # Phase 3: 0 to min_voltage (negative ramp)
            self.voltage = - self._ramp_rate * (current_time_in_cycle - self._ramp_t1 - self._ramp_t2)
        else:
            # Phase 4: min_voltage to 0 (positive ramp)
            self.voltage = self._ramp_min_voltage + self._ramp_rate * (current_time_in_cycle - self._ramp_t1 - self._ramp_t2 - self._ramp_t3)
        
        return self.voltage
        
    def _apply_zero_voltage_hold(self, time: float) -> float:
        """Internal handler: Return 0V while within simulation window"""
        self.voltage = 0.0
        return 0.0
        
    def _apply_constant_voltage(self,time: float) -> float:
        return self._constant_voltage_value
        
        
    def _calculate_effective_device_voltage(self,current):
        
        voltage_drop = current * self.series_resistance
        
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
    
    def calculate_current(self,clusters = None):
        """
        Determine conduction mechanism, compute current, and update measurements.
        Returns: (effective_voltage, current)
        """
        
        if not self.current_enabled:
          return 0.0, 0.0
          
        if not clusters:
          gap_m = self.thickness_m
          model, filament_resistance = "schottky", None      
        else:
          gap_m, model, filament_resistance = self._resolve_conduction_state(clusters)
          
        self.current_model = model
        
        if model == "ohmic":
          self._calculate_ohmic_current(self.voltage,filament_resistance)
          V_eff = self._calculate_effective_device_voltage(self.current)
        else:
          V_eff = self._solve_Schottky_with_series_resistance_Newton(self.voltage,gap_m)
          self._calculate_schottky_current(V_eff,gap_m) 
        
        self.update_measurements()
        return V_eff,self.current
        
        
    def _resolve_conduction_state(self, clusters: Dict[str, object]) -> Tuple[float, str, Optional[float]]:
      """Analyze cluster geometry to determine effective gap and conduction model."""
      bottom_clusters = []
      top_clusters = []
      has_bridge = False
      min_bridge_resistance = float('inf')
      
      # 1. Classify clusters
      for cid, cluster in clusters.items():
        touches_bottom = cluster.attached_layer['bottom_layer']
        touches_top = cluster.attached_layer['top_layer']
                    
        if touches_bottom and touches_top:
          has_bridge = True
          min_bridge_resistance = min(min_bridge_resistance, cluster.total_resistance)

        if touches_bottom:
          bottom_clusters.append(cluster)
        if touches_top:
          top_clusters.append(cluster)

      # 2. Determine conduction regime
      if has_bridge:
        return 0.0, "ohmic", min_bridge_resistance
        
      if not bottom_clusters and not top_clusters:
        return self.thickness_m, "schottky", None
        
      # 3. Compute minimum insulating gap
      min_gap_angstrom = float('inf')
        
      if bottom_clusters and top_clusters:
        # Gap between closest bottom and top clusters
        min_gap_angstroms = self._min_distance_between_sets(bottom_clusters, top_clusters)
      elif bottom_clusters:
        # Gap from highest bottom cluster to top electrode (z = thickness)
        max_z_bottom = max(atom[2] for cl in bottom_clusters for atom in cl.atoms_positions)
        min_gap_angstroms = (self.thickness_m / ANGSTROM_TO_METER) - max_z_bottom
      else: # top_clusters only
        # Gap from lowest top cluster to bottom electrode (z=0)
        min_z_top = min(atom[2] for cl in top_clusters for atom in cl.atoms_positions)
        min_gap_angstroms = min_z_top
        
        
      #if np.allclose(min_gap_angstroms, 0.0): 
      #  min_gap_angstroms = 2.0
      return min_gap_angstroms * ANGSTROM_TO_METER, "schottky", None
    
    @staticmethod
    def _min_distance_between_sets(bottom_clusters: List, top_clusters: List) -> float:
      """Find minimum 3D Euclidean distance between any two atoms across cluster sets."""
      min_dist = float('inf')
              
      # Flatten positions for vectorized computation
      b_pos = np.vstack([cl.atoms_positions for cl in bottom_clusters])
      t_pos = np.vstack([cl.atoms_positions for cl in top_clusters])
      
      # Pairwise distance using broadcasting
      diff = b_pos[:, np.newaxis, :] - t_pos[np.newaxis, :, :]
      dists = np.sqrt(np.sum(diff**2, axis=2))
      min_dist = np.min(dists)
      
      return min_dist
      
    def _calculate_effective_device_voltage(self,current):
        
      votage_drop = current * self.series_resistance  
      effective_voltage = self.voltage - votage_drop
        
      return effective_voltage
      
      
    def _calculate_ohmic_current(self,voltage,filament_resistance):
        """Ohmic current through filament"""
        total_resistance = filament_resistance + self.series_resistance
        self.current = voltage/total_resistance
        return self.current
        
    def _calculate_schottky_current(self,voltage,effective_gap):
        """Schottky emission current"""
        
        if abs(voltage) < 1e-12:
          return 0.0
          
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
        
    def _solve_Schottky_with_series_resistance_Newton(self,voltage,effective_gap,max_iterations=10,tolerance=1e-12):
      """
      Solve V_applied = V_junction + I*R using Newton-Raphson
      We solve: f(I) = V_applied - I*R - V_junction(I) = 0
      """  
      # Initial guess: ignore series resistance
      I_guess = self._calculate_schottky_current(voltage,effective_gap)   
      
      for iteration in range(max_iterations):
        # Calculate effective voltage for current guess
        V_eff = voltage - I_guess * self.series_resistance
        
        # Calculate current from effective voltage
        I_calc = self._calculate_schottky_current(V_eff,effective_gap)
        
        # Function value: f(I) = I_calc - I_guess
        f_I = I_calc - I_guess
        
        # Check convergence
        if abs(f_I) < tolerance:
          return V_eff
          
        # Calculate numerical derivative df/dI
        h = max(abs(I_guess) * 1e-8,1e-15) # Adaptative step size
        V_eff_plus = self.voltage - (I_guess + h) * self.series_resistance
        I_calc_plus = self._calculate_schottky_current(V_eff_plus,effective_gap)
        f_I_plus = I_calc_plus - (I_guess + h)
        
        df_dI = (f_I_plus - f_I) / h
        
        # Avoid dividing by zero
        if abs(df_dI) < 1e-20:
          break
          
        # Newton-Raphson update: I_new = I - f(I)/f'(I)
        I_new = I_guess - f_I/df_dI
        
        # Add damping for stability
        if np.isnan(I_new) or np.isinf(I_new):
          I_new = 0.5 * I_guess + 0.5 * I_calc
        
        I_guess = I_new
      
      return voltage - I_guess * self.series_resistance
        
      
    def load_experimental_data(self, voltage = 'AV', current = 'AI'):
        
        current_directory = Path(__file__).parent
        file_path = current_directory / 'Crystalline IV.csv'
        
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

        headers_V = [f"{voltage}_pos", f"{voltage}_neg"]
        headers_I = [f"{current}_pos", f"{current}_neg"]
    
        voltage_pos = clean_and_convert(df[headers_V[0]])
        current_pos = clean_and_convert(df[headers_I[0]])
        voltage_neg = clean_and_convert(df[headers_V[1]])
        current_neg = clean_and_convert(df[headers_I[1]])
        
        voltage = pd.concat([
          voltage_pos,
          voltage_neg,
          ], ignore_index=True)
          
        current = pd.concat([
          current_pos,
          current_neg,
        ], ignore_index=True)

        
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
        # Add resistance if calculated:
        if self.current != 0 and abs(self.voltage) > 1e-12:
          self.measurements['resistance'].append(abs(self.voltage / self.current))
        else:
          self.measurements['resistance'].append(float('inf'))
        
    def plot_V_t(self):
        
        plt.plot(self.measurements['time'], self.measurements['voltage'])
        plt.show()
        
    def plot_V_I(self, save_path=None):
    
        self.load_experimental_data()

        sim_voltage = np.array(self.measurements['voltage'])
        sim_current = np.array(self.measurements['current'])
    
        exp_voltage = self.experimental_data['voltage']
        exp_current = self.experimental_data['current']

        # Plot experimental and simulated values
        plt.semilogy(sim_voltage, abs(sim_current), label='Simulation', linewidth=2)
        plt.semilogy(exp_voltage, abs(exp_current), label='Experiment', marker='o', markersize=3)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.legend()
        plt.title('Experimental vs simulated I-V Curve')
        plt.grid(True, alpha=0.3)
        
        if save_path is not None:
          save_path = Path(save_path)
          plot_path = save_path / 'iv_curves.png'
          plt.savefig(plot_path, dpi=300,bbox_inches = 'tight')
        
        plt.show()
        
    def save_IV_csv(self,save_path):
        """Save I-V data to CSV file"""
        filename = 'iv_curves.csv'
        save_path = save_path / filename
        df = pd.DataFrame({
        'voltage_simulation': self.measurements['voltage'],
        'current_simulation': self.measurements['current']
        #'voltage_experiment': self.experimental_data['voltage'],
        #'current_experiment': self.experimental_data['current']
        })
    
        df.to_csv(save_path, index=False)