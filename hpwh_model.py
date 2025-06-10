"""
=====================================================================
OCHRE Simulation Configuration for Heat Pump Water Heater Studies
=====================================================================

This module provides the base configuration for simulating a residential dwelling with
a Heat Pump Water Heater (HPWH) using the Object-oriented, Controllable, High-resolution 
Residential Energy (OCHRE) modeling framework developed by the National Renewable Energy 
Laboratory (NREL). It establishes the fundamental simulation parameters used throughout 
the price-signal-based control strategy evaluation.

The configuration includes:
1. Temporal parameters - Setting simulation timing (start time, resolution, duration)
   with a 1-minute resolution for high-fidelity HPWH operation modeling
   
2. Building specification - Using a representative 3-bedroom single-family home model
   based on HPXML standards for Portland, Oregon
   
3. Schedule data - Using standardized residential water draw profiles (schedules_1.csv)
   based on typical occupancy patterns

4. Weather data - Incorporating TMY3 weather data for Portland, Oregon, to provide
   realistic ambient conditions

The dwelling_args dictionary serves as the primary configuration interface for this thesis 
research. Various simulations (baseline, controlled, perfect knowledge, day-ahead) all use 
this common configuration to ensure consistent comparison.

This configuration is designed for use with the OCHRE framework, which models 
building thermal dynamics, appliance energy consumption, and occupant behavior 
at high temporal resolution to evaluate energy performance and demand flexibility.

Usage:
    This file can be run directly to execute a single baseline simulation:
    python ochre_config.py
    
    More commonly, this configuration is imported by other modules:
    from ochre_config import dwelling_args

The OCHRE framework was developed by NREL (Blonsky et al., 2021). This configuration 
file adapts the OCHRE example scripts for the specific purpose of HPWH control studies.

References:
    Blonsky, M., Maguire, J., McKenna, K., Cutler, D., Balamurugan, S.P. and Jin, X., 2021. 
    OCHRE: The object-oriented, controllable, high-resolution residential energy model 
    for dynamic integration studies. Applied Energy, 290, p.116732.
    
    OCHRE GitHub Repository: https://github.com/NREL/OCHRE
    The complete framework and source code are available at this repository.
    
    OCHRE Documentation: https://ochre-nrel.readthedocs.io/en/latest/index.html
    Comprehensive documentation including tutorials, API reference, and examples.

=====================================================================
"""


import os
import datetime as dt
import pandas as pd

from ochre import Dwelling, Analysis, CreateFigures
from ochre.utils import default_input_path

# Main configuration dictionary for OCHRE simulations
dwelling_args = {
    # Timing parameters
    'start_time': dt.datetime(2023, 2, 22, 0, 0),  # year, month, day, hour, minute
                                                   # Note: the model will not work with leap years
    'time_res': dt.timedelta(minutes=1),           # time resolution of the simulation
    'duration': dt.timedelta(days=1),              # duration of the simulation

    # Input parameters - Sample building and equipment characteristics file
    'hpxml_file': os.path.join(default_input_path, 'Input Files', '3bedroom.xml'),

    # Input parameters - Water draw schedule file
    'schedule_input_file': os.path.join(default_input_path, 'Input Files', 'schedules_1.csv'),

    # Input parameters - weather data
    'weather_file': os.path.join(default_input_path, 'Weather', 'USA_OR_Portland.Intl.AP.726980_TMY3.epw'),

    # Output parameters
    'verbosity': 3,                         # verbosity of time series files (0-9)

    # Equipment parameters
    'Equipment': {
        # Water heating equipment configuration
        'Water Heating': {
            'Heat Pump Water Heater': {
            },            
        },
    },
}

if __name__ == '__main__':
    # Initialization
    dwelling = Dwelling(**dwelling_args)

    # Simulation
    df, metrics, hourly = dwelling.simulate()

    # Plot results options (commented out by default)
    data = {'': df}
