"""
=====================================================================
Price-Signal-Based 14-Day Control Strategy 
for Heat Pump Water Heaters
=====================================================================

This module implements a dual-peak detection algorithm that responds to dynamic electricity 
price signals to optimize Heat Pump Water Heater (HPWH) operation. The system identifies 
high-price periods in both morning (00:00-11:59) and evening (12:00-23:59) time blocks, 
generating CTA-2045-B-compatible commands to shift HPWH operation away from peak pricing periods.

The control strategy features:
1. CalFlexHub integration - Processes 14-day seasonal price profiles from the CalFlexHub 
   pricing tool, which provides structured, time-varying electricity rates designed 
   specifically for residential demand response research
2. Dual-period peak detection - separately analyzing morning and evening price patterns
3. Dynamic price thresholds - using proportional thresholds (40/70% of peak price) to 
   determine load-adding and load-shedding periods
4. Seasonal adaptation - adjusting thresholds based on seasonal conditions 
   (winter: 35%, spring/fall: 40%, summer: 45%)
5. CTA-2045-B command integration - translating price decisions into standardized commands:
   - Load-up: Standard preheating (setpoint 130°F)
   - Advanced Load-up: Enhanced preheating (setpoint 145°F)
   - Shed: Load reduction (setpoint 120°F)

The module implements a perfect knowledge control scenario where the same day's price 
profile is used to generate control schedules. It identifies appropriate control periods, 
creates detailed scheduling information for OCHRE-based HPWH simulations, and generates 
comprehensive performance comparisons between baseline and controlled operation.

Performance validation through simulation has demonstrated:
- Electricity cost savings of 11-33% across seasons
- Peak-period load reductions of 86-91%
- Modest 6-8% increases in total energy consumption due to thermal preheating

Functions:
    read_and_process_data: Processes raw price data by season from CalFlexHub
    get_forecast_and_actual_data: Creates data pairs using the same day's data
    split_day_periods: Divides daily data into morning/evening periods
    get_seasonal_threshold: Retrieves season-specific price thresholds
    get_seasonal_parameters: Gets season-specific timing parameters
    identify_period_peaks_and_shed: Detects price peaks and shed periods
    identify_morning_loadup: Generates morning load-up periods
    identify_evening_advanced_loadup: Generates evening advanced load-up periods
    create_schedule_row: Formats schedule data for output
    generate_control_schedules: Creates complete control schedules
    get_water_heater_controls: Implements CTA-2045 controls
    run_simulation: Executes OCHRE simulation with schedule
    main: Coordinates overall execution flow

Usage:
    python3 md_price_based_control.py

Parameters can be modified within the script to adjust thresholds, 
seasonal adaptations, and simulation parameters.

Author: Othman A. Murad
=====================================================================
"""


import os
import datetime as dt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime, timedelta

from ochre import Dwelling
from run_dwelling import dwelling_args

def read_and_process_data(file_path, season):
    """Read and process the 14-day price data with season-specific column names"""
    print(f"Reading file: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Initial data shape: {df.shape}")
    
    # Map season-specific column names to standard names
    column_mapping = {
        'Winter': {
            'date_col': 'winter_period_data_origination_date',
            'price_col': 'winter_period_price'
        },
        'Summer': {
            'date_col': 'summer_period_data_origination_date',
            'price_col': 'summer_period_price'
        },
        'Spring': {
            'date_col': 'spring_period_data_origination_date',
            'price_col': 'spring_period_price'
        },
        'Fall': {
            'date_col': 'fall_period_data_origination_date',
            'price_col': 'fall_period_price'
        }
    }
    
    # Get column names for current season
    cols = column_mapping.get(season)
    if not cols:
        raise ValueError(f"Invalid season: {season}")
    
    print(f"Using columns: {cols}")
    
    # Rename columns to standard names
    df['start_time'] = pd.to_datetime(df[cols['date_col']], format='%m/%d/%y %H:%M')
    df['price'] = pd.to_numeric(df[cols['price_col']], errors='coerce')
    
    # Sort and group by date
    df = df.sort_values('start_time')
    grouped = df.groupby(df['start_time'].dt.date)
    
    print(f"Number of days in grouped data: {len(grouped)}")
    
    return grouped

def split_day_periods(df):
    """Split into morning and evening periods"""
    df = df.copy()
    df['hour'] = df['start_time'].dt.hour
    morning_df = df[df['hour'] < 12].copy()
    evening_df = df[df['hour'] >= 12].copy()
    return morning_df, evening_df

def get_seasonal_threshold(season):
    """Get season-specific price thresholds for advanced load-up"""
    thresholds = {
        'Winter': 0.45,  # Higher threshold due to higher overall prices
        'Summer': 0.35,  # Lower threshold due to better efficiency
        'Spring': 0.40,  # Medium threshold
        'Fall': 0.40     # Medium threshold
    }
    return thresholds.get(season, 0.40)

def get_seasonal_parameters(season):
    """Get season-specific parameters including duration constraints"""
    params = {
        'Winter': {
            'min_duration': 120,  # Minimum 2 hours due to cold inlet
            'max_duration': 180,  # Maximum 3 hours
            'default_duration': 180
        },
        'Summer': {
            'min_duration': 60,   # Minimum 1 hour due to warm inlet
            'max_duration': 120,  # Maximum 2 hours
            'default_duration': 120
        },
        'Spring': {
            'min_duration': 90,   # Minimum 1.5 hours
            'max_duration': 150,  # Maximum 2.5 hours
            'default_duration': 150
        },
        'Fall': {
            'min_duration': 90,   # Minimum 1.5 hours
            'max_duration': 150,  # Maximum 2.5 hours
            'default_duration': 150
        }
    }
    return params.get(season, params['Fall'])

def identify_period_peaks_and_shed(df):
    """Identify peak price and determine shed period"""
    if df.empty:
        return None, []
    
    # Set minimum price threshold
    MIN_PEAK_PRICE = 0.10
    
    # Split the day into morning (5-9) and evening (12-23) periods
    morning_mask = (df['start_time'].dt.hour >= 5) & (df['start_time'].dt.hour <= 9)
    evening_mask = df['start_time'].dt.hour >= 12
    
    is_morning = df['start_time'].iloc[0].hour < 12
    
    if is_morning:
        # Morning period handling (unchanged)
        morning_df = df[morning_mask]
        peak_candidates = morning_df[morning_df['price'] >= MIN_PEAK_PRICE]
        if peak_candidates.empty:
            return None, []
        
        peak_idx = peak_candidates['price'].idxmax()
        peak_time = df.loc[peak_idx, 'start_time']
        peak_price = df.loc[peak_idx, 'price']
        price_threshold = peak_price * 0.7
        
        morning_start = peak_time.replace(hour=5, minute=0)
        morning_end = peak_time.replace(hour=9, minute=0)
        morning_period = df[(df['start_time'] >= morning_start) & 
                          (df['start_time'] <= morning_end)]
        
        above_threshold = morning_period[morning_period['price'] >= price_threshold]
        if above_threshold.empty:
            return None, []
            
        shed_start = above_threshold['start_time'].iloc[0]
        shed_end = morning_end
        
    else:
        # Evening period handling
        evening_df = df[evening_mask].copy()
        
        # Focus on evening peak period (16:00 - 22:00)
        peak_period_mask = (evening_df['start_time'].dt.hour >= 16) & (evening_df['start_time'].dt.hour <= 22)
        peak_period = evening_df[peak_period_mask]
        
        if peak_period.empty:
            return None, []
        
        # Find the maximum price in the evening peak period
        max_price_row = peak_period.loc[peak_period['price'].idxmax()]
        peak_time = max_price_row['start_time']
        peak_price = max_price_row['price']
        
        # Check if peak meets minimum threshold
        if peak_price < MIN_PEAK_PRICE:
            return None, []
        
        # Calculate 70% threshold
        price_threshold = peak_price * 0.7
        
        # Find when price first exceeds 70% threshold before peak
        pre_peak = evening_df[evening_df['start_time'] < peak_time]
        above_threshold = pre_peak[pre_peak['price'] >= price_threshold]
        
        if above_threshold.empty:
            # If no threshold crossing found, use start of significant rise
            price_changes = pre_peak['price'].diff()
            significant_rises = price_changes[price_changes > 0.02]
            
            if not significant_rises.empty:
                first_rise_idx = significant_rises.index[0]
                if first_rise_idx in pre_peak.index:
                    shed_start = pre_peak.loc[first_rise_idx, 'start_time']
                else:
                    return None, []
            else:
                return None, []
        else:
            shed_start = above_threshold.iloc[0]['start_time']
        
        # Find when price drops below 70% threshold after peak
        post_peak = evening_df[evening_df['start_time'] > peak_time]
        recovery_times = post_peak[post_peak['price'] < price_threshold]
        
        if recovery_times.empty:
            shed_end = peak_time.replace(hour=23, minute=59)
        else:
            shed_end = recovery_times.iloc[0]['start_time']
            
    return (peak_time, peak_price), [(shed_start, shed_end)]

def identify_morning_loadup(df, shed_periods):
    """Identify morning load-up period ending at shed start"""
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
    """Identify evening advanced load-up period using price-based approach"""
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

def create_schedule_row(morning_loadup, morning_shed, evening_adv_loadup, evening_shed):
    """Create a single row of schedule data"""
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
            'M_LU_time': '00:00',
            'M_LU_duration': 0.0,
            'M_S_time': '00:00',
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
            'E_ALU_time': '00:00',
            'E_ALU_duration': 0.0,
            'E_S_time': '00:00',
            'E_S_duration': 0.0
        })
    
    return row

def generate_control_schedules(grouped_data, season, output_dir):
    """Generate control schedules for all days"""
    all_schedules = []
    
    for date, df_day in grouped_data:
        morning_df, evening_df = split_day_periods(df_day)
        
        # Get peaks and shed periods
        morning_peak, morning_shed = identify_period_peaks_and_shed(morning_df)
        evening_peak, evening_shed = identify_period_peaks_and_shed(evening_df)
        
        # Generate control periods
        morning_loadup = identify_morning_loadup(morning_df, morning_shed)
        evening_adv_loadup = identify_evening_advanced_loadup(evening_df, evening_shed,
                                                            season, evening_peak)
        
        # Create and store schedule data
        schedule_row = create_schedule_row(morning_loadup, morning_shed,
                                         evening_adv_loadup, evening_shed)
        all_schedules.append(schedule_row)
    
    return all_schedules

def get_water_heater_controls(sim_day, current_time, current_setpoint, control_schedule):
    """Control water heater based on schedule implementing CTA-2045 commands"""
    control = {
        'Water Heating': {
            'Setpoint': current_setpoint,
            'Deadband': 2.8,        # Deadband 5°F
            'Load Fraction': 1      # "0" for forced OFF - DLC
        }
    }
    
    try:
        # Get schedule for current simulation day (1-based indexing)
        day_schedule = control_schedule.iloc[sim_day - 1]
        current_hour = current_time.time()
        
        # Calculate period end times
        base_date = current_time.date()
        
        m_lu_end = (datetime.combine(base_date, pd.to_datetime(day_schedule['M_LU_time']).time()) + 
                    timedelta(hours=day_schedule['M_LU_duration'])).time()
        m_s_end = (datetime.combine(base_date, pd.to_datetime(day_schedule['M_S_time']).time()) + 
                   timedelta(hours=day_schedule['M_S_duration'])).time()
        e_alu_end = (datetime.combine(base_date, pd.to_datetime(day_schedule['E_ALU_time']).time()) + 
                    timedelta(hours=day_schedule['E_ALU_duration'])).time()
        e_s_end = (datetime.combine(base_date, pd.to_datetime(day_schedule['E_S_time']).time()) + 
                   timedelta(hours=day_schedule['E_S_duration'])).time()
        
        # Morning load up period (CTA-2045 Load Up)
        if pd.to_datetime(day_schedule['M_LU_time']).time() <= current_hour < m_lu_end:
            control['Water Heating'].update({
                'Setpoint': 54.4,       # 130°F
                'Deadband': 2.8,        # Deadband 5°F
                'Load Fraction': 1      # "0" for forced OFF - DLC
            })
        
        # Morning shed period (CTA-2045 Shed)
        elif pd.to_datetime(day_schedule['M_S_time']).time() <= current_hour < m_s_end:
            control['Water Heating'].update({
                'Setpoint': 48.9,       # 120°F
                'Deadband': 2.8,        # Deadband 5°F
                'Load Fraction': 1      # "0" for forced OFF - DLC
            })
        
        # Evening advanced load up period (CTA-2045 Advanced Load Up)
        elif pd.to_datetime(day_schedule['E_ALU_time']).time() <= current_hour < e_alu_end:
            control['Water Heating'].update({
                'Setpoint': 62.8,       # 145°F
                'Deadband': 2.8,        # Deadband 5°F
                'Load Fraction': 1      # "0" for forced OFF - DLC
            })
        
        # Evening shed period (CTA-2045 Shed)
        elif pd.to_datetime(day_schedule['E_S_time']).time() <= current_hour < e_s_end:
            control['Water Heating'].update({
                'Setpoint': 48.9,       # 120°F
                'Deadband': 2.8,        # Deadband 5°F
                'Load Fraction': 1      # "0" for forced OFF - DLC
            })
            
    except Exception as e:
        print(f"Warning: Error applying control schedule for day {sim_day}, time {current_time}: {str(e)}")
        
    return control

def run_simulation(control_schedule, season):
    """Run the OCHRE simulation with generated schedule"""
    try:
        print("Starting controlled simulation...")
        
        # Initialize dwelling
        dwelling = Dwelling(name="Water Heater Control Test", **dwelling_args)
        water_heater = dwelling.get_equipment_by_end_use('Water Heating')
        
        # Get simulation start time
        sim_start = dwelling.sim_times[0]
        
        # Simulation loop
        for t in dwelling.sim_times:
            assert dwelling.current_time == t
            
            # Calculate current simulation day (1-based)
            sim_day = (t.date() - sim_start.date()).days + 1
            
            # Get current setpoint
            current_setpoint = water_heater.schedule.loc[t, 'Water Heating Setpoint (C)']
            
            # Get and apply control signal
            control_signal = get_water_heater_controls(
                sim_day=sim_day,
                current_time=t,
                current_setpoint=current_setpoint,
                control_schedule=control_schedule
            )
            house_status = dwelling.update(control_signal=control_signal)
        
        # Get controlled results
        df_controlled, metrics, hourly = dwelling.finalize()
        
        # Run baseline simulation
        print("Running baseline simulation...")
        dwelling_baseline = Dwelling(name="Water Heater Baseline", **dwelling_args)
        df_baseline, _, _ = dwelling_baseline.simulate()
        
        return df_controlled, df_baseline
        
    except Exception as e:
        raise Exception(f"Error in simulation: {str(e)}")

def main(season):
    """Main execution function"""
    # Set up paths and parameters
    input_dir = "/Users/othmanmurad/Documents/OCHRE/ochre/defaults/Input Files"
    file_path = os.path.join(input_dir, f"{season}_14Days-results.csv")
    
    # Set up results directories
    RESULTS_DIR = "/Users/othmanmurad/Documents/OCHRE/results"
    OUTPUT_DIR = os.path.join(RESULTS_DIR, f"{season}_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        print(f"Processing {season} season data...")
        
        # Step 1: Generate control schedules
        print("Generating control schedules...")
        grouped_data = read_and_process_data(file_path, season)
        all_schedules = generate_control_schedules(grouped_data, season, OUTPUT_DIR)
        
        # Save combined schedule
        schedule_df = pd.DataFrame(all_schedules)
        schedule_file = os.path.join(RESULTS_DIR, f'Schedule_MD_{season}.csv')
        schedule_df.to_csv(schedule_file, index=False)
        print(f"Control schedules saved to {schedule_file}")
        
        # Step 2: Run simulation
        print("Running OCHRE simulation...")
        df_controlled, df_baseline = run_simulation(schedule_df, season)
        
        # Save simulation results
        df_controlled.to_csv(os.path.join(RESULTS_DIR, 'MD-controlled_results.csv'))
        df_baseline.to_csv(os.path.join(RESULTS_DIR, 'MD-baseline_results.csv'))
        
        print("Processing complete!")
        print(f"Results saved in: {RESULTS_DIR}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage:
if __name__ == "__main__":
    season = "Winter"           # Can be "Winter", "Spring", "Summer", or "Fall"
    main(season)
