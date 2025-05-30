"""
=====================================================================
Price-Signal-Based Single-Day Control Strategy 
for Heat Pump Water Heaters
=====================================================================

This module implements a rule-based control algorithm for Heat Pump Water Heaters (HPWHs)
based on dynamic electricity price signals from the CalFlexHub pricing tool. The algorithm 
identifies high-price periods in both morning (00:00-11:59) and evening (12:00-23:59) time 
blocks and generates optimal control schedules using CTA-2045-B-compatible commands.

The control strategy features:
1. CalFlexHub price integration - Uses structured, time-varying price profiles specifically 
   designed for demand response research in residential settings, with seasonal 
   price patterns reflecting actual grid conditions.

2. Dual-period analysis - Morning and evening price patterns are analyzed separately
   to capture multiple daily peaks, enhancing overall performance compared to 
   single-peak detection.
   
3. Dynamic thresholds - The strategy employs proportional thresholds to determine 
   control periods:
   - 70% of peak price for load-shedding periods
   - Season-specific thresholds for load-adding periods (40-45%)
   
4. Seasonal adaptation - Control parameters are adjusted based on seasonal conditions:
   - Winter: 35% threshold, 120-180 minute load-up periods
   - Spring/Fall: 40% threshold, 90-150 minute load-up periods
   - Summer: 45% threshold, 60-120 minute load-up periods
   
5. CTA-2045-B command implementation - The control strategy translates price-based 
   decisions into standardized commands:
   - Load-up: Standard preheating (setpoint 130°F/54.4°C)
   - Advanced Load-up: Enhanced preheating (setpoint 145°F/62.8°C)
   - Shed: Load reduction (setpoint 120°F/48.9°C)

This implementation for a perfect knowledge scenario uses the same day's price data to 
generate control schedules, demonstrating theoretically optimal performance. The algorithm 
processes CalFlexHub price signals, creates detailed scheduling information, and runs 
OCHRE-based HPWH simulations to evaluate energy cost savings and peak-period load reduction.

Functions:
    read_and_process_data: Processes raw CalFlexHub price data by season
    get_forecast_and_actual_data: Creates data pairs using the same day's data
    split_day_periods: Divides daily data into morning/evening periods
    get_seasonal_threshold: Retrieves season-specific price thresholds
    get_seasonal_parameters: Gets season-specific timing parameters
    identify_period_peaks_and_shed: Detects price peaks and shed periods
    identify_morning_loadup: Generates morning load-up periods
    identify_evening_advanced_loadup: Generates evening advanced load-up periods
    save_schedule_to_csv: Formats and saves schedule data
    get_water_heater_controls: Implements CTA-2045 controls
    run_simulation: Executes OCHRE simulation with schedule
    main: Coordinates overall execution flow

Usage:
    python sd_price_based_control.py
    
Parameters can be modified within functions to adjust thresholds,
seasonal adaptations, and simulation parameters.

Author: Othman A. Murad
=====================================================================
"""


import os
import datetime as dt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from ochre import Dwelling
from bin.run_dwelling import dwelling_args

def read_and_process_data(file_path):
    """
    Read and process the CSV file
    """
    df = pd.read_csv(file_path)
    df['start_time'] = pd.to_datetime(df['start_time']).dt.tz_localize(None)
    df = df.sort_values('start_time')
    return df

def split_day_periods(df):
    """
    Split the dataset into morning (00:00-11:59) and evening (12:00-23:59) periods
    """
    df = df.copy()
    df['hour'] = df['start_time'].dt.hour
    morning_df = df[df['hour'] < 12].copy()
    evening_df = df[df['hour'] >= 12].copy()
    return morning_df, evening_df

def get_seasonal_threshold(season):
    """
    Get season-specific price thresholds for advanced load-up
    These thresholds determine when advanced load-up should begin based on price patterns
    """
    thresholds = {
        'Winter': 0.45,  # Higher threshold due to higher overall prices and heating needs
        'Summer': 0.35,  # Lower threshold due to better efficiency and warmer inlet
        'Spring': 0.40,  # Medium threshold for moderate conditions
        'Fall': 0.40     # Medium threshold for moderate conditions
    }
    return thresholds.get(season, 0.40)

def get_seasonal_parameters(season):
    """
    Get season-specific parameters including min/max durations
    Durations vary based on seasonal efficiency and inlet water temperatures
    """
    params = {
        'Winter': {
            'min_duration': 120,  # Minimum 2 hours due to cold inlet
            'max_duration': 180,  # Maximum 3 hours for extreme conditions
            'default_duration': 180  # Default if price-based approach fails
        },
        'Summer': {
            'min_duration': 60,   # Minimum 1 hour due to warm inlet
            'max_duration': 120,  # Maximum 2 hours due to high efficiency
            'default_duration': 120
        },
        'Spring': {
            'min_duration': 90,   # Minimum 1.5 hours for moderate conditions
            'max_duration': 150,  # Maximum 2.5 hours
            'default_duration': 150
        },
        'Fall': {
            'min_duration': 90,   # Minimum 1.5 hours for moderate conditions
            'max_duration': 150,  # Maximum 2.5 hours
            'default_duration': 150
        }
    }
    return params.get(season, params['Fall'])

def identify_period_peaks_and_shed(df):
    """
    Identify peak price and determine shed period based on 70% threshold
    """
    if df.empty:
        return None, []
    
    # Set minimum price threshold
    MIN_PEAK_PRICE = 0.15
    
    # Find the maximum price point that exceeds minimum threshold
    peak_candidates = df[df['price'] >= MIN_PEAK_PRICE]
    if peak_candidates.empty:
        return None, []
    
    peak_idx = peak_candidates['price'].idxmax()
    peak_time = df.loc[peak_idx, 'start_time']
    peak_price = df.loc[peak_idx, 'price']
    
    # Calculate 70% threshold for shed period
    price_threshold = peak_price * 0.7
    
    # Define period boundaries
    is_morning = peak_time.hour < 12
    period_start = peak_time.replace(hour=0 if is_morning else 12, minute=0)
    period_end = peak_time.replace(hour=11 if is_morning else 23, minute=59)
    period_df = df[(df['start_time'] >= period_start) & (df['start_time'] <= period_end)]
    
    # Fixed shed start (1 hour before peak)
    shed_start = peak_time - pd.Timedelta(minutes=60)
    
    # Find recovery point after peak
    post_peak = period_df[period_df['start_time'] > peak_time]
    recovery_times = post_peak[post_peak['price'] < price_threshold]
    
    # Set shed end time
    if recovery_times.empty:
        shed_end = period_end
    else:
        shed_end = recovery_times['start_time'].iloc[0]
    
    # Additional validation
    if shed_start < period_start:
        shed_start = period_start
    if shed_end > period_end:
        shed_end = period_end
        
    return (peak_time, peak_price), [(shed_start, shed_end)]

def identify_morning_loadup(df, shed_periods):
    """
    Identify morning load-up period ending at shed start
    Fixed 2-hour duration for morning load-up
    """
    if not shed_periods:
        return []
    
    loadup_periods = []
    shed_start, _ = shed_periods[0]
    
    # Load-up ends exactly when shed starts
    end_time = shed_start
    # Load-up starts 120 minutes before shed
    start_time = end_time - pd.Timedelta(minutes=120)
    
    loadup_periods.append((start_time, end_time, shed_start))
    return loadup_periods

def identify_evening_advanced_loadup(df, shed_periods, season, peak_info):
    """
    Identify evening advanced load-up period using price-based approach
    Includes seasonal thresholds and duration constraints
    """
    if not shed_periods or not peak_info:
        return []
    
    peak_time, peak_price = peak_info
    shed_start, _ = shed_periods[0]
    
    # Get seasonal parameters
    seasonal_params = get_seasonal_parameters(season)
    min_duration = pd.Timedelta(minutes=seasonal_params['min_duration'])
    max_duration = pd.Timedelta(minutes=seasonal_params['max_duration'])
    
    # Calculate price threshold based on season
    price_threshold = peak_price * get_seasonal_threshold(season)
    
    # Find when price first exceeds the threshold before shed
    pre_shed_df = df[df['start_time'] < shed_start].copy()
    threshold_periods = pre_shed_df[pre_shed_df['price'] >= price_threshold]
    
    if threshold_periods.empty:
        # Fallback to default duration if no threshold crossing found
        start_time = shed_start - pd.Timedelta(minutes=seasonal_params['default_duration'])
    else:
        start_time = threshold_periods['start_time'].iloc[0]
    
    # Apply duration constraints
    duration = shed_start - start_time
    if duration > max_duration:
        start_time = shed_start - max_duration
    elif duration < min_duration:
        start_time = shed_start - min_duration
    
    return [(start_time, shed_start, shed_start)]

def save_schedule_to_csv(df, morning_loadup, morning_shed, evening_adv_loadup, evening_shed, season):
    """
    Save the control schedule to CSV
    """
    data = []
    row = {}
    
    # Process morning periods
    if morning_loadup and morning_shed:
        lu_start, lu_end, _ = morning_loadup[0]
        s_start, s_end = morning_shed[0]
        row.update({
            'M_LU_time': lu_start.strftime('%H:%M'),
            'M_LU_duration': (lu_end - lu_start).total_seconds() / 3600,
            'M_S_time': s_start.strftime('%H:%M'),
            'M_S_duration': (s_end - s_start).total_seconds() / 3600
        })
    else:
        row.update({
            'M_LU_time': '0:00',
            'M_LU_duration': 0.0,
            'M_S_time': '0:00',
            'M_S_duration': 0.0
        })
    
    # Process evening periods
    if evening_adv_loadup and evening_shed:
        alu_start, alu_end, _ = evening_adv_loadup[0]
        s_start, s_end = evening_shed[0]
        row.update({
            'E_ALU_time': alu_start.strftime('%H:%M'),
            'E_ALU_duration': (alu_end - alu_start).total_seconds() / 3600,
            'E_S_time': s_start.strftime('%H:%M'),
            'E_S_duration': (s_end - s_start).total_seconds() / 3600
        })
    else:
        row.update({
            'E_ALU_time': 'N/A',
            'E_ALU_duration': 0.0,
            'E_S_time': 'N/A',
            'E_S_duration': 0.0
        })
    
    data.append(row)
    return data[0]

def get_water_heater_controls(hour_of_day, current_setpoint, schedule_data, **unused_inputs):
    """
    Get water heater control parameters based on schedule
    Implements CTA-2045 commands through temperature setpoints
    """
    control = {
        'Water Heating': {
            'Setpoint': current_setpoint,
            'Deadband': 2.8,
            'Load Fraction': 1
        }
    }
    
    # Get schedule times
    m_lu_time = pd.to_datetime(schedule_data['M_LU_time'], format='%H:%M').hour
    m_s_time = pd.to_datetime(schedule_data['M_S_time'], format='%H:%M').hour
    e_alu_time = pd.to_datetime(schedule_data['E_ALU_time'], format='%H:%M').hour
    e_s_time = pd.to_datetime(schedule_data['E_S_time'], format='%H:%M').hour
    
    # Morning load up period (CTA-2045 Load Up)
    if m_lu_time <= hour_of_day < (m_lu_time + schedule_data['M_LU_duration']):
        control['Water Heating'].update({
            'Setpoint': 54.4,       # 130°F
            'Deadband': 2.8,        # Deadband 5°F
            'Load Fraction': 1
        })
    
    # Morning shed period (CTA-2045 Shed)
    elif m_s_time <= hour_of_day < (m_s_time + schedule_data['M_S_duration']):
        control['Water Heating'].update({
            'Setpoint': 48.9,       # 120°F
            'Deadband': 2.8,        # Deadband 5°F
            'Load Fraction': 1
        })
    
    # Evening advanced load up period (CTA-2045 Advanced Load Up)
    elif e_alu_time <= hour_of_day < (e_alu_time + schedule_data['E_ALU_duration']):
        control['Water Heating'].update({
            'Setpoint': 62.8,       # 145°F
            'Deadband': 2.8,        # Deadband 5°F
            'Load Fraction': 1
        })
    
    # Evening shed period (CTA-2045 Shed)
    elif e_s_time <= hour_of_day < (e_s_time + schedule_data['E_S_duration']):
        control['Water Heating'].update({
            'Setpoint': 48.9,       # 120°F
            'Deadband': 2.8,        # Deadband 5°F
            'Load Fraction': 1
        })
    
    return control

def run_simulation(schedule_data):
    """
    Run OCHRE simulation with the generated schedule
    """
    # Create results directory if it doesn't exist
    RESULTS_DIR = "/Users/othmanmurad/Documents/OCHRE/results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize dwelling
    dwelling = Dwelling(name="Water Heater Control Test", **dwelling_args)
    water_heater = dwelling.get_equipment_by_end_use('Water Heating')
    
    # Simulation loop
    for t in dwelling.sim_times:
        assert dwelling.current_time == t
        current_setpoint = water_heater.schedule.loc[t, 'Water Heating Setpoint (C)']
        
        control_signal = get_water_heater_controls(
            hour_of_day=t.hour,
            current_setpoint=current_setpoint,
            schedule_data=schedule_data
        )
        house_status = dwelling.update(control_signal=control_signal)
    
    # Get controlled results
    df_controlled, metrics, hourly = dwelling.finalize()
    
    # Run baseline simulation
    dwelling_baseline = Dwelling(name="Water Heater Baseline", **dwelling_args)
    df_baseline, _, _ = dwelling_baseline.simulate()
    
    # Save results
    df_controlled.to_csv(os.path.join(RESULTS_DIR, 'SD-controlled_results.csv'))
    df_baseline.to_csv(os.path.join(RESULTS_DIR, 'SD-baseline_results.csv'))
    
    print("Simulation results saved to:")
    print(f"- {os.path.join(RESULTS_DIR, 'SD-controlled_results.csv')}")
    print(f"- {os.path.join(RESULTS_DIR, 'SD-baseline_results.csv')}")
    
    return df_controlled, df_baseline

def main(season):
    """
    Main function to run the price-responsive control schedule generation
    """
    # Set up paths and parameters
    input_dir = "/Users/othmanmurad/Documents/OCHRE/ochre/defaults/Input Files"
    file_path = os.path.join(input_dir, f"{season}_24Hrs-results-15min.csv")
    RESULTS_DIR = "/Users/othmanmurad/Documents/OCHRE/results"
    
    # Step 1: Generate optimal schedule
    print(f"Generating optimal control schedule for {season}...")
    df = read_and_process_data(file_path)
    morning_df, evening_df = split_day_periods(df)
    
    # Identify peaks and shed periods
    morning_peak, morning_shed = identify_period_peaks_and_shed(morning_df)
    evening_peak, evening_shed = identify_period_peaks_and_shed(evening_df)
    
    # Identify control periods
    morning_loadup = identify_morning_loadup(morning_df, morning_shed)
    evening_adv_loadup = identify_evening_advanced_loadup(evening_df, evening_shed, season, evening_peak)
    
    # Generate schedule
    schedule_data = save_schedule_to_csv(df, morning_loadup, morning_shed,
                                       evening_adv_loadup, evening_shed, season)
    
    # Save schedule to CSV
    schedule_df = pd.DataFrame([schedule_data])
    schedule_file = os.path.join(RESULTS_DIR, f'Schedule_SD_{season}.csv')
    schedule_df.to_csv(schedule_file, index=False)
    print(f"Control schedule saved to {schedule_file}")
    
    # Step 2: Run OCHRE simulation
    print("Running OCHRE simulation with optimal schedule...")
    df_controlled, df_baseline = run_simulation(schedule_data)
    
    print("Simulation complete. Results saved in the results directory:")
    print(f"- Control schedule: Schedule_SD_{season}.csv")
    print("- Controlled results: SD-controlled_results.csv")
    print("- Baseline results: SD-baseline_results.csv")

# Example usage for different seasons
if __name__ == "__main__":
    season = "Winter"           # Can be "Winter", "Spring", "Summer", or "Fall"
    main(season)
