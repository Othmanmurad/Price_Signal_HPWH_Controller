
"""
=====================================================================
Large-Scale Multi-Simulation Analysis Framework 
for HPWH Control Strategy
=====================================================================

This module implements a comprehensive statistical analysis system for evaluating the 
performance of price-signal-based control strategies for Heat Pump Water Heaters (HPWHs) 
across multiple simulations. It processes the results from Case Study II which used 
eight different water draw profiles across 48 simulations per season (192 annual total) 
to evaluate control strategy robustness under diverse usage patterns.

The analysis framework features:
1. Automated multi-simulation processing - Handles all simulation directories, 
   extracts performance metrics, and aggregates results into consolidated datasets
   
2. Comprehensive statistical analysis - Calculates detailed statistics including:
   - Mean and standard deviation values for all metrics
   - Confidence intervals and statistical significance testing
   - Quantile analysis (25th, 50th, 75th percentiles)
   - Min/max ranges and probability distributions
   
3. Multi-dimensional performance evaluation - Quantifies performance across key dimensions:
   - Energy efficiency (consumption changes, percentage differences)
   - Economic impact (cost savings in absolute and percentage terms)
   - Grid support capability (peak-period load reduction)
   - Temperature maintenance (average and minimum water temperatures)
   
4. Advanced visualization system - Generates:
   - Distribution plots with confidence intervals and standard deviations
   - Bar charts comparing performance across simulations
   - Statistical summary images with key performance indicators
   - Seasonal performance comparison visualizations
   
5. Detailed reporting infrastructure - Creates:
   - Comprehensive statistical summary files
   - Consolidated performance CSV files
   - Per-simulation metrics with day-by-day breakdowns

This analysis framework underpins the statistical validation presented in Case Study II, 
demonstrating the control strategy's robustness across diverse residential usage patterns.
The code processes both baseline and controlled simulation results, calculates control-period
specific metrics (shed periods, load-up periods), and provides rigorous statistical
analysis of the performance differences.

Functions:
    compare_simulation_results: Main function to process all simulations
    load_control_schedule: Loads control period schedules from CSV files
    process_simulation: Processes individual simulation results
    calculate_cost: Calculates electricity costs using time-varying pricing
    calculate_peak_energy: Calculates energy consumption during peak periods
    calculate_loadup_energy: Calculates energy consumption during load-up periods
    calculate_period_energy_for_day: Calculates energy for specific time periods
    calculate_statistics: Generates comprehensive statistical summaries
    create_summary_plots: Creates visualization plots for key metrics
    create_daily_cost_summary: Creates daily cost comparison summaries
    create_distribution_plots: Creates bell curve plots with statistics
    create_summary_image: Creates statistical summary visualization
    compare_results: Main entry point for the analysis framework

Usage:
    python md_large_scale_results_analysis.py
    
    Where [season] is an optional parameter specifying which season to analyze
    (Winter, Spring, Summer, or Fall). Defaults to Winter if not specified.

Author: Othman A. Murad
=====================================================================
"""


import os
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from scipy import stats 
from datetime import datetime, time, timedelta

def compare_simulation_results(large_scale_dir, season="Winter"):
    """
    Compare baseline and controlled results for all simulations
    
    Args:
        large_scale_dir (str): Path to the LargeScale directory containing simulation folders
        season (str): Season for analysis ("Winter", "Spring", "Summer", "Fall")
    
    Returns:
        pd.DataFrame: Summary dataframe with comparison results for all simulations
    """
    print(f"\nComparing {season} baseline and controlled simulations...")
    
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PRICE_DATA = os.path.join(BASE_DIR, f'inputs/{season}_14Days-results.csv')
    
    # Check if price data exists
    if not os.path.exists(PRICE_DATA):
        print(f"Error: Price data file not found at {PRICE_DATA}")
        return None
    
    # Read price data
    price_df = pd.read_csv(PRICE_DATA)
    date_col = f'{season.lower()}_period_data_origination_date'
    price_col = f'{season.lower()}_period_price'
    
    # Check if the expected columns exist
    if date_col not in price_df.columns or price_col not in price_df.columns:
        possible_date_cols = [col for col in price_df.columns if 'date' in col.lower()]
        possible_price_cols = [col for col in price_df.columns if 'price' in col.lower()]
        
        print(f"Warning: Expected columns '{date_col}' and '{price_col}' not found.")
        print(f"Available date columns: {possible_date_cols}")
        print(f"Available price columns: {possible_price_cols}")
        
        if possible_date_cols and possible_price_cols:
            date_col = possible_date_cols[0]
            price_col = possible_price_cols[0]
            print(f"Using '{date_col}' and '{price_col}' instead.")
        else:
            print("Error: Cannot find appropriate columns for date and price.")
            return None
    
    price_df = price_df.rename(columns={
        date_col: 'timestamp',
        price_col: 'price'
    })
    
    # Process timestamps
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
    price_df['day'] = price_df.index // 24
    price_df['hour'] = price_df['timestamp'].dt.hour
    
    print(f"Reading {season} price data, shape: {price_df.shape}")
    
    # Find all simulation directories
    simulation_dirs = glob.glob(os.path.join(large_scale_dir, "simulation_*"))
    
    if not simulation_dirs:
        print(f"Warning: No simulation directories found in {large_scale_dir}")
        print("Available directories:")
        for item in os.listdir(large_scale_dir):
            full_path = os.path.join(large_scale_dir, item)
            if os.path.isdir(full_path):
                print(f"  - {item}")
        return None
        
    simulation_dirs.sort(key=lambda x: int(re.search(r'simulation_(\d+)', x).group(1)))
    print(f"Found {len(simulation_dirs)} simulation directories.")
    
    # Create output directory for summary
    summary_dir = os.path.join(large_scale_dir, "statistics")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Initialize summary dataframe
    summary_data = []
    
    # Store simulation details for statistics file
    simulation_details = []
    
    # Process each simulation
    for sim_dir in simulation_dirs:
        sim_num = int(re.search(r'simulation_(\d+)', sim_dir).group(1))
        
        # Results files
        baseline_file = os.path.join(sim_dir, "baseline_results.csv")
        controlled_file = os.path.join(sim_dir, "controlled_results.csv")
        schedule_file = os.path.join(sim_dir, "control_schedule.csv")
        
        # Check if all required files exist
        files_exist = True
        if not os.path.exists(baseline_file):
            print(f"Warning: Baseline file not found at {baseline_file}")
            files_exist = False
        if not os.path.exists(controlled_file):
            print(f"Warning: Controlled file not found at {controlled_file}")
            files_exist = False
        if not os.path.exists(schedule_file):
            print(f"Warning: Schedule file not found at {schedule_file}")
            files_exist = False
        
        if files_exist:
            try:
                # Load control schedule
                control_periods = load_control_schedule(schedule_file)
                
                # Process simulation
                sim_data = process_simulation(baseline_file, controlled_file, sim_num, control_periods, price_df)
                summary_data.append(sim_data)
                
                # Store simulation details
                detail = f"Processed simulation {sim_num:2d}: {sim_data['Energy Savings (kWh)']:.2f} kWh, "
                detail += f"${sim_data['Cost Savings ($)']:.2f}, "
                detail += f"Peak reduction: {sim_data['Peak Energy Reduction (kWh)']:.2f} kWh "
                detail += f"({sim_data['Peak Energy Reduction (%)']:.1f}%)"
                
                simulation_details.append(detail)
                print(detail)
                
            except Exception as e:
                print(f"Error processing simulation {sim_num}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Create summary dataframe
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by simulation number
        summary_df.sort_values('Simulation', inplace=True)
        
        # Save summary to CSV
        summary_file = os.path.join(summary_dir, f"{season}_comparison_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        # Calculate and save statistics
        calculate_statistics(summary_df, summary_dir, season, simulation_details)
        
        # Create summary plots
        create_summary_plots(summary_df, summary_dir, season)
        
        print(f"\nComparison analysis completed for {season} season.")
        print(f"Summary saved to: {summary_file}")
        
        return summary_df
    
    else:
        print("No simulations processed.")
        return None

def load_control_schedule(schedule_file):
    """
    Load and process control schedule from schedule file
    
    Args:
        schedule_file (str): Path to control schedule CSV file
    
    Returns:
        dict: Dictionary of control periods by day
    """
    try:
        schedule_df = pd.read_csv(schedule_file)
        
        control_periods = {}
        # Assuming the schedule file contains one row per day
        for day in range(len(schedule_df)):
            if day < len(schedule_df):
                day_schedule = schedule_df.iloc[day]
                
                # Create a dictionary for each day's control periods
                control_periods[day] = {
                    'M_LU': (day_schedule['M_LU_time'], day_schedule['M_LU_duration']),
                    'M_S': (day_schedule['M_S_time'], day_schedule['M_S_duration']),
                    'E_ALU': (day_schedule['E_ALU_time'], day_schedule['E_ALU_duration']),
                    'E_S': (day_schedule['E_S_time'], day_schedule['E_S_duration'])
                }
            else:
                # Use first day's schedule as a default if we have more days than schedule entries
                control_periods[day] = control_periods[0]
        
        return control_periods
        
    except Exception as e:
        print(f"Error loading control schedule: {str(e)}")
        # Return empty schedule as fallback
        return {0: {'M_LU': ('00:00', 0), 'M_S': ('00:00', 0), 
                   'E_ALU': ('00:00', 0), 'E_S': ('00:00', 0)}}

def process_simulation(baseline_file, controlled_file, sim_num, control_periods, price_df):
    """
    Process and compare baseline and controlled results for a single simulation
    
    Args:
        baseline_file (str): Path to the baseline results CSV file
        controlled_file (str): Path to the controlled results CSV file
        sim_num (int): Simulation number
        control_periods (dict): Dictionary of control periods by day
        price_df (pd.DataFrame): Price data
    
    Returns:
        dict: Dictionary containing comparison results
    """
    # Read results files
    baseline_df = pd.read_csv(baseline_file, index_col=0, parse_dates=True)
    controlled_df = pd.read_csv(controlled_file, index_col=0, parse_dates=True)
    
    # Get the start and end dates
    start_date = baseline_df.index[0].strftime('%Y-%m-%d')
    end_date = baseline_df.index[-1].strftime('%Y-%m-%d')
    
    # Get the duration in days
    duration_days = (baseline_df.index[-1] - baseline_df.index[0]).days + 1
    
    # Add day information to dataframes
    for df in [baseline_df, controlled_df]:
        df['day'] = [(ts - baseline_df.index[0]).days for ts in df.index]
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
    
    # Calculate interval energy (1-minute intervals)
    baseline_df['interval_energy'] = baseline_df['Water Heating Electric Power (kW)'] / 60  # kWh
    controlled_df['interval_energy'] = controlled_df['Water Heating Electric Power (kW)'] / 60  # kWh
    
    # Calculate total energy consumption
    baseline_energy = baseline_df['interval_energy'].sum()
    controlled_energy = controlled_df['interval_energy'].sum()
    energy_savings = baseline_energy - controlled_energy
    energy_savings_pct = (energy_savings / baseline_energy) * 100 if baseline_energy > 0 else 0
    
    # Calculate cost
    baseline_cost = calculate_cost(baseline_df, price_df)
    controlled_cost = calculate_cost(controlled_df, price_df)
    cost_savings = baseline_cost - controlled_cost
    cost_savings_pct = (cost_savings / baseline_cost) * 100 if baseline_cost > 0 else 0
    
    # Calculate peak energy consumption (during shed periods)
    baseline_peak, controlled_peak = calculate_peak_energy(baseline_df, controlled_df, control_periods)
    peak_reduction = baseline_peak - controlled_peak
    peak_reduction_pct = (peak_reduction / baseline_peak) * 100 if baseline_peak > 0 else 0
    
    # Calculate load-up energy consumption
    baseline_loadup, controlled_loadup = calculate_loadup_energy(baseline_df, controlled_df, control_periods)
    loadup_increase = controlled_loadup - baseline_loadup
    loadup_increase_pct = (loadup_increase / baseline_loadup) * 100 if baseline_loadup > 0 else 0
    
    # Calculate temperature metrics
    baseline_avg_temp = baseline_df['Hot Water Outlet Temperature (C)'].mean()
    controlled_avg_temp = controlled_df['Hot Water Outlet Temperature (C)'].mean()
    baseline_min_temp = baseline_df['Hot Water Outlet Temperature (C)'].min()
    controlled_min_temp = controlled_df['Hot Water Outlet Temperature (C)'].min()
    
    # Calculate water draw
    water_draw_cols = [col for col in baseline_df.columns if 'Draw' in col and 'Volume' in col]
    if water_draw_cols:
        water_draw_col = water_draw_cols[0]
        baseline_water = baseline_df[water_draw_col].sum()
        controlled_water = controlled_df[water_draw_col].sum()
    else:
        # If no explicit water draw volume column, try hot water delivered
        if 'Hot Water Delivered (L/min)' in baseline_df.columns:
            baseline_water = baseline_df['Hot Water Delivered (L/min)'].sum()
            controlled_water = controlled_df['Hot Water Delivered (L/min)'].sum()
        else:
            baseline_water = 0
            controlled_water = 0
            print(f"Warning: Could not find water draw column for simulation {sim_num}")
    
    # Return comparison data
    return {
        'Simulation': sim_num,
        'Start Date': start_date,
        'End Date': end_date,
        'Duration (days)': duration_days,
        'Baseline Energy (kWh)': round(baseline_energy, 2),
        'Controlled Energy (kWh)': round(controlled_energy, 2),
        'Energy Savings (kWh)': round(energy_savings, 2),
        'Energy Savings (%)': round(energy_savings_pct, 2),
        'Baseline Cost ($)': round(baseline_cost, 2),
        'Controlled Cost ($)': round(controlled_cost, 2),
        'Cost Savings ($)': round(cost_savings, 2),
        'Cost Savings (%)': round(cost_savings_pct, 2),
        'Baseline Peak Energy (kWh)': round(baseline_peak, 2),
        'Controlled Peak Energy (kWh)': round(controlled_peak, 2),
        'Peak Energy Reduction (kWh)': round(peak_reduction, 2),
        'Peak Energy Reduction (%)': round(peak_reduction_pct, 2),
        'Baseline Load-up Energy (kWh)': round(baseline_loadup, 2),
        'Controlled Load-up Energy (kWh)': round(controlled_loadup, 2),
        'Load-up Energy Increase (kWh)': round(loadup_increase, 2),
        'Load-up Energy Increase (%)': round(loadup_increase_pct, 2),
        'Baseline Avg Temp (C)': round(baseline_avg_temp, 2),
        'Controlled Avg Temp (C)': round(controlled_avg_temp, 2),
        'Baseline Min Temp (C)': round(baseline_min_temp, 2),
        'Controlled Min Temp (C)': round(controlled_min_temp, 2),
        'Baseline Water Draw (L)': round(baseline_water, 2),
        'Controlled Water Draw (L)': round(controlled_water, 2)
    }

def calculate_cost(df, price_df):
    """
    Calculate cost based on time-of-use pricing
    
    Args:
        df (pd.DataFrame): Dataframe with energy data
        price_df (pd.DataFrame): Price data
    
    Returns:
        float: Total cost
    """
    try:
        # Create a mapping of day and hour to price
        price_map = {}
        for _, row in price_df.iterrows():
            day = int(row['day'])
            hour = int(row['hour'])
            price = float(row['price'])
            price_map[(day, hour)] = price
        
        # Calculate cost using vectorized operations
        total_cost = 0.0
        avg_price = price_df['price'].mean()
        
        for day in df['day'].unique():
            day_df = df[df['day'] == day]
            for hour in range(24):
                hour_df = day_df[day_df['hour'] == hour]
                if not hour_df.empty:
                    # Get price for this day and hour (default to mean price if not found)
                    key = (int(day) % len(set(price_df['day'])), hour)
                    price = price_map.get(key, avg_price)
                    
                    # Calculate cost for this hour
                    hour_energy = hour_df['interval_energy'].sum()
                    hour_cost = hour_energy * price
                    total_cost += hour_cost
        
        return total_cost
    
    except Exception as e:
        print(f"Error calculating cost: {str(e)}")
        # Fallback to using average price if there's an error
        avg_price = price_df['price'].mean()
        return df['interval_energy'].sum() * avg_price

def calculate_peak_energy(baseline_df, controlled_df, control_periods):
    """
    Calculate energy consumption during peak periods (shed periods)
    
    Args:
        baseline_df (pd.DataFrame): Baseline dataframe with energy data
        controlled_df (pd.DataFrame): Controlled dataframe with energy data
        control_periods (dict): Dictionary of control periods by day
    
    Returns:
        tuple: (baseline_peak_energy, controlled_peak_energy)
    """
    baseline_peak_energy = 0.0
    controlled_peak_energy = 0.0
    
    # Process each day
    for day in sorted(baseline_df['day'].unique()):
        # Skip days that don't have control period information
        if day not in control_periods:
            continue
        
        # Process morning shed period
        m_shed_time, m_shed_duration = control_periods[day]['M_S']
        if m_shed_time != 'N/A' and m_shed_time != '00:00' and float(m_shed_duration) > 0:
            # Calculate energy during morning shed
            m_shed_baseline = calculate_period_energy_for_day(baseline_df, day, m_shed_time, float(m_shed_duration))
            m_shed_controlled = calculate_period_energy_for_day(controlled_df, day, m_shed_time, float(m_shed_duration))
            
            baseline_peak_energy += m_shed_baseline
            controlled_peak_energy += m_shed_controlled
        
        # Process evening shed period
        e_shed_time, e_shed_duration = control_periods[day]['E_S']
        if e_shed_time != 'N/A' and e_shed_time != '00:00' and float(e_shed_duration) > 0:
            # Calculate energy during evening shed
            e_shed_baseline = calculate_period_energy_for_day(baseline_df, day, e_shed_time, float(e_shed_duration))
            e_shed_controlled = calculate_period_energy_for_day(controlled_df, day, e_shed_time, float(e_shed_duration))
            
            baseline_peak_energy += e_shed_baseline
            controlled_peak_energy += e_shed_controlled
    
    return baseline_peak_energy, controlled_peak_energy

def calculate_loadup_energy(baseline_df, controlled_df, control_periods):
    """
    Calculate energy consumption during load-up periods
    
    Args:
        baseline_df (pd.DataFrame): Baseline dataframe with energy data
        controlled_df (pd.DataFrame): Controlled dataframe with energy data
        control_periods (dict): Dictionary of control periods by day
    
    Returns:
        tuple: (baseline_loadup_energy, controlled_loadup_energy)
    """
    baseline_loadup_energy = 0.0
    controlled_loadup_energy = 0.0
    
    # Process each day
    for day in sorted(baseline_df['day'].unique()):
        # Skip days that don't have control period information
        if day not in control_periods:
            continue
        
        # Process morning load-up period
        m_lu_time, m_lu_duration = control_periods[day]['M_LU']
        if m_lu_time != 'N/A' and m_lu_time != '00:00' and float(m_lu_duration) > 0:
            # Calculate energy during morning load-up
            m_lu_baseline = calculate_period_energy_for_day(baseline_df, day, m_lu_time, float(m_lu_duration))
            m_lu_controlled = calculate_period_energy_for_day(controlled_df, day, m_lu_time, float(m_lu_duration))
            
            baseline_loadup_energy += m_lu_baseline
            controlled_loadup_energy += m_lu_controlled
        
        # Process evening advanced load-up period
        e_alu_time, e_alu_duration = control_periods[day]['E_ALU']
        if e_alu_time != 'N/A' and e_alu_time != '00:00' and float(e_alu_duration) > 0:
            # Calculate energy during evening advanced load-up
            e_alu_baseline = calculate_period_energy_for_day(baseline_df, day, e_alu_time, float(e_alu_duration))
            e_alu_controlled = calculate_period_energy_for_day(controlled_df, day, e_alu_time, float(e_alu_duration))
            
            baseline_loadup_energy += e_alu_baseline
            controlled_loadup_energy += e_alu_controlled
    
    return baseline_loadup_energy, controlled_loadup_energy

def calculate_period_energy_for_day(df, day, period_time_str, period_duration):
    """
    Calculate energy consumption for a specific period on a specific day
    
    Args:
        df (pd.DataFrame): Dataframe with energy data
        day (int): Day number
        period_time_str (str): Period start time (HH:MM format)
        period_duration (float): Period duration in hours
    
    Returns:
        float: Total energy consumption during the period
    """
    try:
        # Parse start time
        start_hour, start_minute = map(int, period_time_str.split(':'))
        
        # Calculate end hour and minute
        total_minutes = start_hour * 60 + start_minute + period_duration * 60
        end_hour = int(total_minutes // 60)
        end_minute = int(total_minutes % 60)
        
        # Filter data for the day
        day_df = df[df['day'] == day].copy()
        
        # Create mask for the period
        hour_gt_start = day_df['hour'] > start_hour
        hour_eq_start_and_min_ge_start = (day_df['hour'] == start_hour) & (day_df['minute'] >= start_minute)
        hour_lt_end = day_df['hour'] < end_hour
        hour_eq_end_and_min_lt_end = (day_df['hour'] == end_hour) & (day_df['minute'] < end_minute)
        
        # Complete period mask
        in_period = (hour_gt_start | hour_eq_start_and_min_ge_start) & (hour_lt_end | hour_eq_end_and_min_lt_end)
        
        # Sum energy during the period
        if in_period.any():
            return day_df.loc[in_period, 'interval_energy'].sum()
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error calculating period energy: {str(e)}")
        return 0.0

def calculate_statistics(summary_df, summary_dir, season, simulation_details):
    """
    Calculate and save overall statistics with comprehensive details
    
    Args:
        summary_df (pd.DataFrame): Summary dataframe with results for all simulations
        summary_dir (str): Directory to save statistics
        season (str): Season for analysis
        simulation_details (list): List of strings with details for each simulation
    """
    # Calculate average metrics
    avg_energy_savings = summary_df['Energy Savings (kWh)'].mean()
    avg_energy_savings_pct = summary_df['Energy Savings (%)'].mean()
    avg_cost_savings = summary_df['Cost Savings ($)'].mean()
    avg_cost_savings_pct = summary_df['Cost Savings (%)'].mean()
    avg_peak_reduction = summary_df['Peak Energy Reduction (kWh)'].mean()
    avg_peak_reduction_pct = summary_df['Peak Energy Reduction (%)'].mean()
    avg_loadup_increase = summary_df['Load-up Energy Increase (kWh)'].mean()
    avg_loadup_increase_pct = summary_df['Load-up Energy Increase (%)'].mean()
    
    # Calculate standard deviations
    std_energy_savings = summary_df['Energy Savings (kWh)'].std()
    std_energy_savings_pct = summary_df['Energy Savings (%)'].std()
    std_cost_savings = summary_df['Cost Savings ($)'].std()
    std_cost_savings_pct = summary_df['Cost Savings (%)'].std()
    std_peak_reduction = summary_df['Peak Energy Reduction (kWh)'].std()
    std_peak_reduction_pct = summary_df['Peak Energy Reduction (%)'].std()
    std_loadup_increase = summary_df['Load-up Energy Increase (kWh)'].std() if 'Load-up Energy Increase (kWh)' in summary_df.columns else 0
    std_loadup_increase_pct = summary_df['Load-up Energy Increase (%)'].std() if 'Load-up Energy Increase (%)' in summary_df.columns else 0
    
    # Calculate min/max values
    min_energy_savings = summary_df['Energy Savings (kWh)'].min()
    max_energy_savings = summary_df['Energy Savings (kWh)'].max()
    min_cost_savings = summary_df['Cost Savings ($)'].min()
    max_cost_savings = summary_df['Cost Savings ($)'].max()
    min_peak_reduction = summary_df['Peak Energy Reduction (kWh)'].min()
    max_peak_reduction = summary_df['Peak Energy Reduction (kWh)'].max()
    
    # Calculate quartiles (25th, 50th, 75th percentiles)
    energy_q1, energy_med, energy_q3 = summary_df['Energy Savings (kWh)'].quantile([0.25, 0.5, 0.75])
    cost_q1, cost_med, cost_q3 = summary_df['Cost Savings ($)'].quantile([0.25, 0.5, 0.75])
    peak_q1, peak_med, peak_q3 = summary_df['Peak Energy Reduction (kWh)'].quantile([0.25, 0.5, 0.75])
    
    # Calculate confidence intervals (95% CI)
    n = len(summary_df)
    energy_ci = 1.96 * std_energy_savings / np.sqrt(n) if n > 1 else 0
    cost_ci = 1.96 * std_cost_savings / np.sqrt(n) if n > 1 else 0
    peak_ci = 1.96 * std_peak_reduction / np.sqrt(n) if n > 1 else 0
    
    # Calculate statistical significance using t-tests
    try:
        t_stat_energy, p_val_energy = stats.ttest_1samp(summary_df['Energy Savings (kWh)'], 0)
        energy_sig = "statistically significant" if p_val_energy < 0.05 else "not statistically significant"
        
        t_stat_cost, p_val_cost = stats.ttest_1samp(summary_df['Cost Savings ($)'], 0)
        cost_sig = "statistically significant" if p_val_cost < 0.05 else "not statistically significant"
        
        t_stat_peak, p_val_peak = stats.ttest_1samp(summary_df['Peak Energy Reduction (kWh)'], 0)
        peak_sig = "statistically significant" if p_val_peak < 0.05 else "not statistically significant"
    except:
        energy_sig, cost_sig, peak_sig = "N/A", "N/A", "N/A"
        p_val_energy, p_val_cost, p_val_peak = 0, 0, 0
    
    # Calculate baseline and controlled averages for comprehensive reporting
    avg_baseline_energy = summary_df['Baseline Energy (kWh)'].mean()
    avg_controlled_energy = summary_df['Controlled Energy (kWh)'].mean()
    avg_baseline_cost = summary_df['Baseline Cost ($)'].mean()
    avg_controlled_cost = summary_df['Controlled Cost ($)'].mean()
    avg_baseline_peak = summary_df['Baseline Peak Energy (kWh)'].mean()
    avg_controlled_peak = summary_df['Controlled Peak Energy (kWh)'].mean()
    
    # Create statistics string
    statistics = f"""
{season} Season Comparison Statistics
===================================

Number of simulations: {len(summary_df)}

Individual Simulation Results:
-----------------------------
"""
    
    # Add individual simulation details
    for detail in simulation_details:
        statistics += detail + "\n"
    
    # Add detailed summary table for each simulation
    statistics += f"\nDetailed Summary Table:\n"
    statistics += "-" * 125 + "\n"
    statistics += f"{'Sim':^4} | {'Baseline':^8} | {'Controlled':^8} | {'Savings':^8} | {'Savings':^8} | "
    statistics += f"{'Baseline':^8} | {'Controlled':^8} | {'Savings':^8} | {'Savings':^8} | "
    statistics += f"{'Peak Red.':^8} | {'Peak Red.':^8}\n"
    
    statistics += f"{'#':^4} | {'Energy':^8} | {'Energy':^8} | {'Energy':^8} | {'Energy':^8} | "
    statistics += f"{'Cost':^8} | {'Cost':^8} | {'Cost':^8} | {'Cost':^8} | "
    statistics += f"{'kWh':^8} | {'%':^8}\n"
    
    statistics += f"{' ':^4} | {'(kWh)':^8} | {'(kWh)':^8} | {'(kWh)':^8} | {'(%)':^8} | "
    statistics += f"{'($)':^8} | {'($)':^8} | {'($)':^8} | {'(%)':^8} | "
    statistics += f"{' ':^8} | {' ':^8}\n"
    
    statistics += "-" * 125 + "\n"
    
    for _, row in summary_df.iterrows():
        sim_num = int(row['Simulation'])
        statistics += f"{sim_num:^4} | {row['Baseline Energy (kWh)']:8.2f} | {row['Controlled Energy (kWh)']:8.2f} | "
        statistics += f"{row['Energy Savings (kWh)']:8.2f} | {row['Energy Savings (%)']:8.2f} | "
        statistics += f"{row['Baseline Cost ($)']:8.2f} | {row['Controlled Cost ($)']:8.2f} | "
        statistics += f"{row['Cost Savings ($)']:8.2f} | {row['Cost Savings (%)']:8.2f} | "
        statistics += f"{row['Peak Energy Reduction (kWh)']:8.2f} | {row['Peak Energy Reduction (%)']:8.2f}\n"
    
    statistics += "-" * 125 + "\n"
    
    # Continue with summary statistics
    statistics += f"""
Comprehensive Summary Statistics:
===============================

Baseline vs. Controlled Comparison:
---------------------------------
Baseline Energy:      {avg_baseline_energy:.2f} kWh (avg)
Controlled Energy:    {avg_controlled_energy:.2f} kWh (avg)
Baseline Cost:        ${avg_baseline_cost:.2f} (avg)
Controlled Cost:      ${avg_controlled_cost:.2f} (avg)
Baseline Peak Energy: {avg_baseline_peak:.2f} kWh (avg)
Controlled Peak Energy: {avg_controlled_peak:.2f} kWh (avg)

Energy Savings:
-------------
Average energy savings:          {avg_energy_savings:.2f} kWh (std: {std_energy_savings:.2f} kWh)
Average savings percentage:      {avg_energy_savings_pct:.2f}% (std: {std_energy_savings_pct:.2f}%)
95% Confidence Interval:         [{avg_energy_savings - energy_ci:.2f}, {avg_energy_savings + energy_ci:.2f}] kWh
Standard Deviation Percentage:   {(std_energy_savings/abs(avg_energy_savings)*100) if avg_energy_savings != 0 else 0:.2f}%
±1 Standard Deviation Range:     [{avg_energy_savings - std_energy_savings:.2f}, {avg_energy_savings + std_energy_savings:.2f}] kWh
Min energy savings:              {min_energy_savings:.2f} kWh
Max energy savings:              {max_energy_savings:.2f} kWh
Quartiles (25th, 50th, 75th):    {energy_q1:.2f}, {energy_med:.2f}, {energy_q3:.2f} kWh
Statistical Significance:        {energy_sig} (p = {p_val_energy:.4f})

Cost Savings:
-----------
Average cost savings:            ${avg_cost_savings:.2f} (std: ${std_cost_savings:.2f})
Average cost savings percentage: {avg_cost_savings_pct:.2f}% (std: {std_cost_savings_pct:.2f}%)
95% Confidence Interval:         [${avg_cost_savings - cost_ci:.2f}, ${avg_cost_savings + cost_ci:.2f}]
Standard Deviation Percentage:   {(std_cost_savings/abs(avg_cost_savings)*100) if avg_cost_savings != 0 else 0:.2f}%
±1 Standard Deviation Range:     [${avg_cost_savings - std_cost_savings:.2f}, ${avg_cost_savings + std_cost_savings:.2f}]
Min cost savings:                ${min_cost_savings:.2f}
Max cost savings:                ${max_cost_savings:.2f}
Quartiles (25th, 50th, 75th):    ${cost_q1:.2f}, ${cost_med:.2f}, ${cost_q3:.2f}
Statistical Significance:        {cost_sig} (p = {p_val_cost:.4f})

Peak Energy Reduction (Shed Periods):
-----------------------------------
Average peak energy reduction:      {avg_peak_reduction:.2f} kWh (std: {std_peak_reduction:.2f} kWh)
Average peak reduction percentage:  {avg_peak_reduction_pct:.2f}% (std: {std_peak_reduction_pct:.2f}%)
95% Confidence Interval:            [{avg_peak_reduction - peak_ci:.2f}, {avg_peak_reduction + peak_ci:.2f}] kWh
Standard Deviation Percentage:      {(std_peak_reduction/abs(avg_peak_reduction)*100) if avg_peak_reduction != 0 else 0:.2f}%
±1 Standard Deviation Range:        [{avg_peak_reduction - std_peak_reduction:.2f}, {avg_peak_reduction + std_peak_reduction:.2f}] kWh
Min peak reduction:                 {min_peak_reduction:.2f} kWh
Max peak reduction:                 {max_peak_reduction:.2f} kWh
Quartiles (25th, 50th, 75th):       {peak_q1:.2f}, {peak_med:.2f}, {peak_q3:.2f} kWh
Statistical Significance:           {peak_sig} (p = {p_val_peak:.4f})

Load-up Energy Increase:
----------------------
Average load-up energy increase:     {avg_loadup_increase:.2f} kWh (std: {std_loadup_increase:.2f} kWh)
Average load-up increase percentage: {avg_loadup_increase_pct:.2f}% (std: {std_loadup_increase_pct:.2f}%)

Temperature Maintenance:
----------------------
Average baseline temperature:           {summary_df['Baseline Avg Temp (C)'].mean():.2f} °C (std: {summary_df['Baseline Avg Temp (C)'].std():.2f} °C)
Average controlled temperature:         {summary_df['Controlled Avg Temp (C)'].mean():.2f} °C (std: {summary_df['Controlled Avg Temp (C)'].std():.2f} °C)
Average baseline minimum temperature:   {summary_df['Baseline Min Temp (C)'].mean():.2f} °C (std: {summary_df['Baseline Min Temp (C)'].std():.2f} °C)
Average controlled minimum temperature: {summary_df['Controlled Min Temp (C)'].mean():.2f} °C (std: {summary_df['Controlled Min Temp (C)'].std():.2f} °C)

Water Draw Comparison:
--------------------
Average baseline water draw:   {summary_df['Baseline Water Draw (L)'].mean():.2f} L (std: {summary_df['Baseline Water Draw (L)'].std():.2f} L)
Average controlled water draw: {summary_df['Controlled Water Draw (L)'].mean():.2f} L (std: {summary_df['Controlled Water Draw (L)'].std():.2f} L)

Daily Cost and Usage Comparison:
------------------------------
Average daily baseline energy:  {summary_df['Baseline Energy (kWh)'].mean() / summary_df['Duration (days)'].mean():.2f} kWh
Average daily controlled energy: {summary_df['Controlled Energy (kWh)'].mean() / summary_df['Duration (days)'].mean():.2f} kWh
Average daily energy savings:   {avg_energy_savings / summary_df['Duration (days)'].mean():.2f} kWh
Average daily cost savings:     ${avg_cost_savings / summary_df['Duration (days)'].mean():.2f}

Analysis Generation Information:
------------------------------
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Data source: {len(summary_df)} simulations in {season} season
Analysis version: HPWH Large-Scale Comparison Analysis v2.0
"""
    
    # Save statistics to file
    stats_file = os.path.join(summary_dir, f"{season}_comprehensive_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write(statistics)
    
    print(f"Comprehensive statistics saved to: {stats_file}")
    
    # Also save detailed daily cost data
    daily_cost_file = os.path.join(summary_dir, f"{season}_daily_cost_comparison.csv")
    create_daily_cost_summary(summary_df, daily_cost_file)


def create_summary_plots(summary_df, summary_dir, season):
    """
    Create summary plots for all simulations
    
    Args:
        summary_df (pd.DataFrame): Summary dataframe with results for all simulations
        summary_dir (str): Directory to save plots
        season (str): Season for analysis
    """
    try:
        # Create plots directory
        plots_dir = os.path.join(summary_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Energy Savings Plot - Absolute Values
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Energy Savings (kWh)'])
        plt.axhline(y=summary_df['Energy Savings (kWh)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: {summary_df["Energy Savings (kWh)"].mean():.2f} kWh')
        plt.xlabel('Simulation Number')
        plt.ylabel('Energy Savings (kWh)')
        plt.title(f'{season} Season Energy Savings')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_energy_savings.png'))
        plt.close()
        
        # Energy Savings Plot - Percentage Values
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Energy Savings (%)'])
        plt.axhline(y=summary_df['Energy Savings (%)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: {summary_df["Energy Savings (%)"].mean():.2f}%')
        plt.xlabel('Simulation Number')
        plt.ylabel('Energy Savings (%)')
        plt.title(f'{season} Season Energy Savings Percentage')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_energy_savings_percentage.png'))
        plt.close()
        
        # Cost Savings Plot - Absolute Values
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Cost Savings ($)'])
        plt.axhline(y=summary_df['Cost Savings ($)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: ${summary_df["Cost Savings ($)"].mean():.2f}')
        plt.xlabel('Simulation Number')
        plt.ylabel('Cost Savings ($)')
        plt.title(f'{season} Season Cost Savings')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_cost_savings.png'))
        plt.close()
        
        # Cost Savings Plot - Percentage Values
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Cost Savings (%)'])
        plt.axhline(y=summary_df['Cost Savings (%)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: {summary_df["Cost Savings (%)"].mean():.2f}%')
        plt.xlabel('Simulation Number')
        plt.ylabel('Cost Savings (%)')
        plt.title(f'{season} Season Cost Savings Percentage')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_cost_savings_percentage.png'))
        plt.close()
        
        # Peak Energy Reduction Plot - Absolute Values
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Peak Energy Reduction (kWh)'])
        plt.axhline(y=summary_df['Peak Energy Reduction (kWh)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: {summary_df["Peak Energy Reduction (kWh)"].mean():.2f} kWh')
        plt.xlabel('Simulation Number')
        plt.ylabel('Peak Energy Reduction (kWh)')
        plt.title(f'{season} Season Peak Energy Reduction')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_peak_reduction.png'))
        plt.close()
        
        # Peak Energy Reduction Plot - Percentage Values
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Peak Energy Reduction (%)'])
        plt.axhline(y=summary_df['Peak Energy Reduction (%)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: {summary_df["Peak Energy Reduction (%)"].mean():.2f}%')
        plt.xlabel('Simulation Number')
        plt.ylabel('Peak Energy Reduction (%)')
        plt.title(f'{season} Season Peak Energy Reduction Percentage')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_peak_reduction_percentage.png'))
        plt.close()
        
        # Continue with other plots...
        # Create a summary image with key metrics
        create_summary_image(summary_df, season, plots_dir)
        
        # Try to create distribution plots
        try:
            create_distribution_plots(summary_df, season, plots_dir)
        except Exception as e:
            print(f"Could not create distribution plots: {str(e)}")
        
        print(f"Summary plots saved to: {plots_dir}")
    
    except Exception as e:
        print(f"Error creating summary plots: {str(e)}")
        import traceback
        traceback.print_exc()

def create_daily_cost_summary(summary_df, output_file):
    """
    Create a summary of daily costs for all simulations
    
    Args:
        summary_df (pd.DataFrame): Summary dataframe with results for all simulations
        output_file (str): Path to output CSV file
    """
    # Create a dataframe for daily costs
    daily_df = pd.DataFrame()
    
    # Calculate average daily values
    daily_df['Simulation'] = summary_df['Simulation']
    daily_df['Baseline Energy (kWh/day)'] = summary_df['Baseline Energy (kWh)'] / summary_df['Duration (days)']
    daily_df['Controlled Energy (kWh/day)'] = summary_df['Controlled Energy (kWh)'] / summary_df['Duration (days)']
    daily_df['Energy Savings (kWh/day)'] = summary_df['Energy Savings (kWh)'] / summary_df['Duration (days)']
    daily_df['Energy Savings (%/day)'] = summary_df['Energy Savings (%)']
    daily_df['Baseline Cost ($/day)'] = summary_df['Baseline Cost ($)'] / summary_df['Duration (days)']
    daily_df['Controlled Cost ($/day)'] = summary_df['Controlled Cost ($)'] / summary_df['Duration (days)']
    daily_df['Cost Savings ($/day)'] = summary_df['Cost Savings ($)'] / summary_df['Duration (days)']
    daily_df['Cost Savings (%/day)'] = summary_df['Cost Savings (%)']
    
    # Round values
    for col in daily_df.columns:
        if col != 'Simulation':
            daily_df[col] = daily_df[col].round(2)
    
    # Save to CSV
    daily_df.to_csv(output_file, index=False)
    
    print(f"Daily cost summary saved to: {output_file}")

def create_summary_plots(summary_df, summary_dir, season):
    """
    Create summary plots for all simulations
    
    Args:
        summary_df (pd.DataFrame): Summary dataframe with results for all simulations
        summary_dir (str): Directory to save plots
        season (str): Season for analysis
    """
    try:
        # Create plots directory
        plots_dir = os.path.join(summary_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Energy Savings Plot
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Energy Savings (kWh)'])
        plt.axhline(y=summary_df['Energy Savings (kWh)'].mean(), color='r', linestyle='-', label=f'Mean: {summary_df["Energy Savings (kWh)"].mean():.2f} kWh')
        plt.xlabel('Simulation Number')
        plt.ylabel('Energy Savings (kWh)')
        plt.title(f'{season} Season Energy Savings')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_energy_savings.png'))
        plt.close()
        
        # Cost Savings Plot
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Cost Savings ($)'])
        plt.axhline(y=summary_df['Cost Savings ($)'].mean(), color='r', linestyle='-', label=f'Mean: ${summary_df["Cost Savings ($)"].mean():.2f}')
        plt.xlabel('Simulation Number')
        plt.ylabel('Cost Savings ($)')
        plt.title(f'{season} Season Cost Savings')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_cost_savings.png'))
        plt.close()
        
        # Peak Energy Reduction Plot
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Peak Energy Reduction (kWh)'])
        plt.axhline(y=summary_df['Peak Energy Reduction (kWh)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: {summary_df["Peak Energy Reduction (kWh)"].mean():.2f} kWh')
        plt.xlabel('Simulation Number')
        plt.ylabel('Peak Energy Reduction (kWh)')
        plt.title(f'{season} Season Peak Energy Reduction')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_peak_reduction.png'))
        plt.close()
        
        # # Energy Savings Percentage Plot
        # plt.figure(figsize=(12, 6))
        # plt.bar(summary_df['Simulation'], summary_df['Energy Savings (%)'])
        # plt.axhline(y=summary_df['Energy Savings (%)'].mean(), color='r', linestyle='-', 
        #             label=f'Mean: {summary_df["Energy Savings (%)"].mean():.2f}%')
        # plt.xlabel('Simulation Number')
        # plt.ylabel('Energy Savings (%)')
        # plt.title(f'{season} Season Energy Savings Percentage')
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(plots_dir, f'{season}_energy_savings_pct.png'))
        # plt.close()
        
        # # Scatter plot for Energy vs. Cost Savings
        # plt.figure(figsize=(8, 8))
        # plt.scatter(summary_df['Energy Savings (kWh)'], summary_df['Cost Savings ($)'])
        # plt.xlabel('Energy Savings (kWh)')
        # plt.ylabel('Cost Savings ($)')
        # plt.title(f'{season} Season Energy vs. Cost Savings')
        # plt.grid(True, alpha=0.3)
        
        # # Add trend line
        # z = np.polyfit(summary_df['Energy Savings (kWh)'], summary_df['Cost Savings ($)'], 1)
        # p = np.poly1d(z)
        # plt.plot(summary_df['Energy Savings (kWh)'], p(summary_df['Energy Savings (kWh)']), 
        #          "r--", label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(plots_dir, f'{season}_energy_vs_cost.png'))
        # plt.close()
        
        # # Temperature comparison boxplot
        # plt.figure(figsize=(10, 6))
        # temperature_data = [summary_df['Baseline Avg Temp (C)'], summary_df['Controlled Avg Temp (C)']]
        # plt.boxplot(temperature_data, labels=['Baseline', 'Controlled'])
        # plt.ylabel('Average Temperature (°C)')
        # plt.title(f'{season} Season Temperature Comparison')
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.savefig(os.path.join(plots_dir, f'{season}_temperature_comparison.png'))
        # plt.close()
        
        # Create a summary image with key metrics
        create_summary_image(summary_df, season, plots_dir)
        
        # Try to create distribution plots
        try:
            create_distribution_plots(summary_df, season, plots_dir)
        except Exception as e:
            print(f"Could not create distribution plots: {str(e)}")
        
        # # Try to create correlation matrix
        # try:
        #     create_correlation_matrix(summary_df, season, plots_dir)
        # except Exception as e:
        #     print(f"Could not create correlation matrix: {str(e)}")
        
        # # Try to create quantile analysis
        # try:
        #     create_quantile_analysis(summary_df, season, plots_dir)
        # except Exception as e:
        #     print(f"Could not create quantile analysis: {str(e)}")
        
        print(f"Summary plots saved to: {plots_dir}")
    
    except Exception as e:
        print(f"Error creating summary plots: {str(e)}")
        import traceback
        traceback.print_exc()

def create_distribution_plots(summary_df, season, plots_dir):
    """
    Create bell curve distribution plots for key metrics with min and max values
    
    Args:
        summary_df (pd.DataFrame): Summary dataframe with results for all simulations
        season (str): Season for analysis
        plots_dir (str): Directory to save plots
    """
    # Key metrics to plot with color assignments
    metrics = [
        ('Energy Savings (kWh)', '#4C78A8'),
        ('Cost Savings ($)', '#F58518'),
        ('Peak Energy Reduction (kWh)', '#72B7B2')
    ]
    
    # Create bell curve for each metric in a separate figure
    for metric, color in metrics:
        if metric not in summary_df.columns:
            continue
            
        data = summary_df[metric]
        mean = data.mean()
        std = data.std()
        min_val = data.min()
        max_val = data.max()
        
        # Calculate standard deviation percentage and CI
        std_percentage = (std/abs(mean)*100)  # Same calculation as CV but labeled differently
        
        # Calculate mean percentage based on the metric
        if metric == 'Peak Energy Reduction (kWh)':
            # Use the corresponding percentage column if available
            if 'Peak Energy Reduction (%)' in summary_df.columns:
                mean_percentage = summary_df['Peak Energy Reduction (%)'].mean()
            else:
                # Estimate from baseline peak data if available
                if 'Baseline Peak Energy (kWh)' in summary_df.columns:
                    baseline_peak_mean = summary_df['Baseline Peak Energy (kWh)'].mean()
                    mean_percentage = (mean / baseline_peak_mean) * 100 if baseline_peak_mean > 0 else 0
                else:
                    # Default fallback if no baseline data is available
                    mean_percentage = 0
        elif metric == 'Energy Savings (kWh)':
            if 'Energy Savings (%)' in summary_df.columns:
                mean_percentage = summary_df['Energy Savings (%)'].mean()
            else:
                mean_percentage = 0
        elif metric == 'Cost Savings ($)':
            if 'Cost Savings (%)' in summary_df.columns:
                mean_percentage = summary_df['Cost Savings (%)'].mean()
            else:
                mean_percentage = 0
        else:
            mean_percentage = 0
        
        ci_lower = mean - 1.96*std
        ci_upper = mean + 1.96*std
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create x values for the bell curve, ensuring min and max are included
        x_min = min(min_val - 0.5*std, mean - 4*std)
        x_max = max(max_val + 0.5*std, mean + 4*std)
        x = np.linspace(x_min, x_max, 1000)
        
        # Create the bell curve
        y = stats.norm.pdf(x, mean, std)
        
        # Plot the bell curve
        plt.plot(x, y, color=color, linewidth=2)
        
        # Fill the area under the curve
        plt.fill_between(x, y, color=color, alpha=0.2)
        
        # Add vertical lines for mean and standard deviations
        plt.axvline(mean, color='red', linestyle='-', linewidth=1.5)
        plt.axvline(mean - std, color='red', linestyle='--', linewidth=1)
        plt.axvline(mean + std, color='red', linestyle='--', linewidth=1)
        
        # Add vertical lines for min and max values
        plt.axvline(min_val, color='green', linestyle='-.', linewidth=1)
        plt.axvline(max_val, color='green', linestyle='-.', linewidth=1)
        
        # Mark the 95% confidence interval area
        ci_x = np.linspace(ci_lower, ci_upper, 100)
        ci_y = stats.norm.pdf(ci_x, mean, std)
        plt.fill_between(ci_x, ci_y, color='red', alpha=0.2)
        
        # Plot the actual data points as a rug plot at the bottom
        plt.plot(data.values, np.zeros_like(data.values), '|', color=color, ms=20, markeredgewidth=2)
        
        # Create a custom legend in a logical order
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        legend_elements = [
            # First group: Central tendency
            Line2D([0], [0], color='red', linestyle='-', lw=1.5, 
                   label=f'Mean: {mean:.2f}'),
            Line2D([0], [0], color='white', marker='_', 
                   label=f'Mean: {mean_percentage:.1f}%'),
            
            # Second group: Variation metrics
            Line2D([0], [0], color='white', marker='_', label=f'Standard Deviation: {std:.2f}'),
            Line2D([0], [0], color='white', marker='_', label=f'Standard Deviation: {std_percentage:.1f}%'),
            
            # Third group: Statistical intervals
            Line2D([0], [0], color='red', linestyle='--', lw=1, 
                   label=f'±1 SD: [{mean-std:.2f}, {mean+std:.2f}]'),
            Patch(facecolor='red', alpha=0.2, 
                  label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]'),
                  
            # Fourth group: Min and Max values
            Line2D([0], [0], color='green', linestyle='-.', lw=1,
                   label=f'Min: {min_val:.2f}'),
            Line2D([0], [0], color='green', linestyle='-.', lw=1,
                   label=f'Max: {max_val:.2f}'),
        ]
        
        # Add title and labels
        plt.title(f'{season} Season {metric} Distribution', fontsize=16)
        plt.xlabel(metric, fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        
        # Add custom legend
        plt.legend(handles=legend_elements, fontsize=10, loc='best')
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add grid for readability
        plt.grid(True, alpha=0.3)
        
        # Tighten layout
        plt.tight_layout()
        
        # Save the figure
        safe_name = metric.replace(" ", "_").replace("(", "").replace(")", "").replace("$", "dollar")
        plt.savefig(os.path.join(plots_dir, f'{season}_season_{safe_name}_bell_curve.png'), 
                    dpi=300, bbox_inches='tight')
        
        plt.close()

#def create_correlation_matrix(summary_df, season, plots_dir):
    # """
    # Create correlation matrix to show relationships between different variables
    
    # Args:
    #     summary_df (pd.DataFrame): Summary dataframe with results for all simulations
    #     season (str): Season for analysis
    #     plots_dir (str): Directory to save plots
    # """
    # # Select numeric columns for correlation
    # numeric_cols = [
    #     'Baseline Energy (kWh)', 'Controlled Energy (kWh)', 
    #     'Energy Savings (kWh)', 'Energy Savings (%)',
    #     'Baseline Cost ($)', 'Controlled Cost ($)', 
    #     'Cost Savings ($)', 'Cost Savings (%)',
    #     'Baseline Peak Energy (kWh)', 'Controlled Peak Energy (kWh)',
    #     'Peak Energy Reduction (kWh)', 'Peak Energy Reduction (%)',
    #     'Baseline Avg Temp (C)', 'Controlled Avg Temp (C)'
    # ]
    
    # # Filter columns that exist in the dataframe
    # numeric_cols = [col for col in numeric_cols if col in summary_df.columns]
    
    # # Calculate correlation matrix
    # corr_matrix = summary_df[numeric_cols].corr()
    
    # # Create heatmap
    # plt.figure(figsize=(14, 12))
    # plt.title(f'{season} Season Correlation Matrix of Key Metrics', fontsize=16)
    
    # # Plot heatmap
    # cmap = plt.cm.RdBu_r  # Red-Blue colormap
    # sns_heatmap = plt.pcolor(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    # plt.colorbar(sns_heatmap, label='Correlation Coefficient')
    
    # # Add correlation values
    # for i in range(len(corr_matrix.columns)):
    #     for j in range(len(corr_matrix.columns)):
    #         plt.text(j + 0.5, i + 0.5, f'{corr_matrix.iloc[i, j]:.2f}',
    #                 ha='center', va='center', color='black')
    
    # # Set ticks and labels
    # plt.xticks(np.arange(len(numeric_cols)) + 0.5, numeric_cols, rotation=90)
    # plt.yticks(np.arange(len(numeric_cols)) + 0.5, numeric_cols)
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f'{season}_correlation_matrix.png'))
    # plt.close()
    
    # # Attempt to create scatter matrix for select variables
    # try:
    #     key_vars = [
    #         'Energy Savings (kWh)', 'Cost Savings ($)', 
    #         'Peak Energy Reduction (kWh)', 'Baseline Avg Temp (C)', 
    #         'Controlled Avg Temp (C)'
    #     ]
    #     key_vars = [col for col in key_vars if col in summary_df.columns]
        
    #     if hasattr(pd, 'plotting') and hasattr(pd.plotting, 'scatter_matrix'):
    #         pd.plotting.scatter_matrix(summary_df[key_vars], figsize=(14, 12), 
    #                                 diagonal='kde', alpha=0.7)
    #         plt.suptitle(f'{season} Season Scatter Matrix of Key Metrics', fontsize=16)
    #         plt.tight_layout(rect=[0, 0, 1, 0.95])
    #         plt.savefig(os.path.join(plots_dir, f'{season}_scatter_matrix.png'))
    #         plt.close()
    # except Exception as e:
    #     print(f"Could not create scatter matrix: {str(e)}")

#def create_quantile_analysis(summary_df, season, plots_dir):
    # """
    # Create quantile analysis plots for key metrics
    
    # Args:
    #     summary_df (pd.DataFrame): Summary dataframe with results for all simulations
    #     season (str): Season for analysis
    #     plots_dir (str): Directory to save plots
    # """
    # # Key metrics to analyze
    # metrics = [
    #     'Energy Savings (kWh)', 'Energy Savings (%)',
    #     'Cost Savings ($)', 'Cost Savings (%)',
    #     'Peak Energy Reduction (kWh)', 'Peak Energy Reduction (%)'
    # ]
    
    # # Filter metrics that exist in the dataframe
    # metrics = [metric for metric in metrics if metric in summary_df.columns]
    
    # # Calculate quantiles
    # quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    # quantile_values = {}
    
    # for metric in metrics:
    #     quantile_values[metric] = [summary_df[metric].quantile(q) for q in quantiles]
    
    # # Create quantile plot
    # plt.figure(figsize=(14, 8))
    
    # # Set width of bars
    # barWidth = 0.15
    
    # # Set position of bars on X axis
    # positions = np.arange(len(metrics))
    
    # # Create bars
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # for i, q in enumerate(quantiles):
    #     plt.bar(positions + i*barWidth, 
    #             [quantile_values[metric][i] for metric in metrics], 
    #             width=barWidth, 
    #             alpha=0.7, 
    #             color=colors[i],
    #             label=f'{int(q*100)}th Percentile')
    
    # # Add labels and title
    # plt.xlabel('Metrics', fontweight='bold')
    # plt.ylabel('Values', fontweight='bold')
    # plt.title(f'{season} Season Quantile Analysis of Key Metrics')
    
    # # Adjust x-axis labels
    # formatted_metrics = [m.replace(' (', '\n(') for m in metrics]
    # plt.xticks([r + barWidth*2 for r in range(len(metrics))], formatted_metrics)
    
    # # Create legend
    # plt.legend()
    
    # # Show plot
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f'{season}_quantile_analysis.png'))
    # plt.close()
    
    # # Create table with quantile values
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.axis('off')
    
    # # Table data
    # table_data = []
    # for metric in metrics:
    #     row = [metric] + [f"{val:.2f}" for val in quantile_values[metric]]
    #     table_data.append(row)
    
    # # Column labels
    # col_labels = ['Metric'] + [f"{int(q*100)}th Percentile" for q in quantiles]
    
    # # Create table
    # table = ax.table(cellText=table_data, 
    #                 colLabels=col_labels,
    #                 loc='center',
    #                 cellLoc='center')
    
    # # Adjust font size
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)
    # table.scale(1, 1.5)
    
    # # Set title
    # plt.title(f'{season} Season Quantile Values of Key Metrics')
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f'{season}_quantile_table.png'))
    # plt.close()

def create_summary_image(summary_df, season, plots_dir):
    """
    Create a summary image with key performance metrics
    
    Args:
        summary_df (pd.DataFrame): Summary dataframe with results for all simulations
        season (str): Season for analysis
        plots_dir (str): Directory to save plots
    """
    try:
        # Calculate key metrics
        avg_energy_savings = summary_df['Energy Savings (kWh)'].mean()
        avg_energy_savings_pct = summary_df['Energy Savings (%)'].mean()
        avg_cost_savings = summary_df['Cost Savings ($)'].mean()
        avg_cost_savings_pct = summary_df['Cost Savings (%)'].mean()
        avg_peak_reduction = summary_df['Peak Energy Reduction (kWh)'].mean()
        avg_peak_reduction_pct = summary_df['Peak Energy Reduction (%)'].mean()
        
        # Calculate 95% confidence intervals
        n = len(summary_df)
        
        energy_std = summary_df['Energy Savings (kWh)'].std()
        energy_ci = 1.96 * energy_std / np.sqrt(n) if n > 1 else 0
        
        cost_std = summary_df['Cost Savings ($)'].std()
        cost_ci = 1.96 * cost_std / np.sqrt(n) if n > 1 else 0
        
        peak_std = summary_df['Peak Energy Reduction (kWh)'].std()
        peak_ci = 1.96 * peak_std / np.sqrt(n) if n > 1 else 0
        
        # Calculate quartiles
        energy_q1, energy_med, energy_q3 = summary_df['Energy Savings (kWh)'].quantile([0.25, 0.5, 0.75])
        cost_q1, cost_med, cost_q3 = summary_df['Cost Savings ($)'].quantile([0.25, 0.5, 0.75])
        peak_q1, peak_med, peak_q3 = summary_df['Peak Energy Reduction (kWh)'].quantile([0.25, 0.5, 0.75])
        
        # Create figure
        plt.figure(figsize=(12, 10))
        plt.axis('off')
        
        # Add title
        plt.suptitle(f'{season} Season Performance Summary', fontsize=20, y=0.98)
        
        # Add metrics
        y_pos = 0.90
        plt.text(0.5, y_pos, f'Total Simulations: {len(summary_df)}', 
                 horizontalalignment='center', fontsize=14)
        
        # Energy Savings with CI
        y_pos -= 0.06
        plt.text(0.5, y_pos, f'Energy Savings: {avg_energy_savings:.2f} kWh ± {energy_ci:.2f} (95% CI)', 
                 horizontalalignment='center', fontsize=14)
        
        # Energy Savings Percentage
        y_pos -= 0.04
        plt.text(0.5, y_pos, f'Energy Savings Percentage: {avg_energy_savings_pct:.2f}%', 
                 horizontalalignment='center', fontsize=14)
        
        # Cost Savings with CI
        y_pos -= 0.06
        plt.text(0.5, y_pos, f'Cost Savings: ${avg_cost_savings:.2f} ± ${cost_ci:.2f} (95% CI)', 
                 horizontalalignment='center', fontsize=14)
        
        # Cost Savings Percentage
        y_pos -= 0.04
        plt.text(0.5, y_pos, f'Cost Savings Percentage: {avg_cost_savings_pct:.2f}%', 
                 horizontalalignment='center', fontsize=14)
        
        # Peak Reduction with CI
        y_pos -= 0.06
        plt.text(0.5, y_pos, f'Peak Energy Reduction: {avg_peak_reduction:.2f} kWh ± {peak_ci:.2f} (95% CI)', 
                 horizontalalignment='center', fontsize=14)
        
        # Peak Reduction Percentage
        y_pos -= 0.04
        plt.text(0.5, y_pos, f'Peak Reduction Percentage: {avg_peak_reduction_pct:.2f}%', 
                 horizontalalignment='center', fontsize=14)
        
        # Add quartile information
        y_pos -= 0.08
        plt.text(0.5, y_pos, 'Quartile Analysis (25th | 50th | 75th percentiles):', 
                 horizontalalignment='center', fontsize=14, fontweight='bold')
        
        y_pos -= 0.05
        plt.text(0.5, y_pos, f'Energy Savings: {energy_q1:.2f} | {energy_med:.2f} | {energy_q3:.2f} kWh', 
                 horizontalalignment='center', fontsize=12)
        
        y_pos -= 0.04
        plt.text(0.5, y_pos, f'Cost Savings: ${cost_q1:.2f} | ${cost_med:.2f} | ${cost_q3:.2f}', 
                 horizontalalignment='center', fontsize=12)
        
        y_pos -= 0.04
        plt.text(0.5, y_pos, f'Peak Reduction: {peak_q1:.2f} | {peak_med:.2f} | {peak_q3:.2f} kWh', 
                 horizontalalignment='center', fontsize=12)
        
        # Add range information
        y_pos -= 0.08
        plt.text(0.5, y_pos, 'Performance Ranges (min to max):', 
                 horizontalalignment='center', fontsize=14, fontweight='bold')
        
        y_pos -= 0.05
        plt.text(0.5, y_pos, f'Energy Savings: {summary_df["Energy Savings (kWh)"].min():.2f} to {summary_df["Energy Savings (kWh)"].max():.2f} kWh', 
                 horizontalalignment='center', fontsize=12)
        
        y_pos -= 0.04
        plt.text(0.5, y_pos, f'Cost Savings: ${summary_df["Cost Savings ($)"].min():.2f} to ${summary_df["Cost Savings ($)"].max():.2f}', 
                 horizontalalignment='center', fontsize=12)
        
        y_pos -= 0.04
        plt.text(0.5, y_pos, f'Peak Reduction: {summary_df["Peak Energy Reduction (kWh)"].min():.2f} to {summary_df["Peak Energy Reduction (kWh)"].max():.2f} kWh', 
                 horizontalalignment='center', fontsize=12)
        
        # Add statistical significance note
        if n >= 5:  # Only add if sample size is reasonably large
            try:
                # Test if energy savings are significant
                t_stat, p_val_energy = stats.ttest_1samp(summary_df['Energy Savings (kWh)'], 0)
                energy_sig = "statistically significant" if p_val_energy < 0.05 else "not statistically significant"
                
                # Test if cost savings are significant
                t_stat, p_val_cost = stats.ttest_1samp(summary_df['Cost Savings ($)'], 0)
                cost_sig = "statistically significant" if p_val_cost < 0.05 else "not statistically significant"
                
                # Test if peak reduction is significant
                t_stat, p_val_peak = stats.ttest_1samp(summary_df['Peak Energy Reduction (kWh)'], 0)
                peak_sig = "statistically significant" if p_val_peak < 0.05 else "not statistically significant"
                
                y_pos -= 0.08
                plt.text(0.5, y_pos, 'Statistical Significance (one-sample t-test):',
                        horizontalalignment='center', fontsize=14, fontweight='bold')
                
                y_pos -= 0.05
                plt.text(0.5, y_pos, f'Energy Savings: {energy_sig} (p = {p_val_energy:.4f})',
                        horizontalalignment='center', fontsize=12)
                
                y_pos -= 0.04
                plt.text(0.5, y_pos, f'Cost Savings: {cost_sig} (p = {p_val_cost:.4f})',
                        horizontalalignment='center', fontsize=12)
                
                y_pos -= 0.04
                plt.text(0.5, y_pos, f'Peak Reduction: {peak_sig} (p = {p_val_peak:.4f})',
                        horizontalalignment='center', fontsize=12)
            except:
                print("Could not calculate statistical significance.")
        
        # Add timestamp
        plt.text(0.5, 0.02, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                 horizontalalignment='center', fontsize=10)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_summary.png'))
        plt.close()
    
    except Exception as e:
        print(f"Error creating summary image: {str(e)}")

def create_summary_plots(summary_df, summary_dir, season):
    """
    Create summary plots for all simulations
    
    Args:
        summary_df (pd.DataFrame): Summary dataframe with results for all simulations
        summary_dir (str): Directory to save plots
        season (str): Season for analysis
    """
    try:
        # Create plots directory
        plots_dir = os.path.join(summary_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Energy Savings Plot - Absolute Values
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Energy Savings (kWh)'])
        plt.axhline(y=summary_df['Energy Savings (kWh)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: {summary_df["Energy Savings (kWh)"].mean():.2f} kWh')
        plt.xlabel('Simulation Number')
        plt.ylabel('Energy Savings (kWh)')
        plt.title(f'{season} Season Energy Savings')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_energy_savings.png'))
        plt.close()
        
        # Energy Savings Plot - Percentage Values
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Energy Savings (%)'])
        plt.axhline(y=summary_df['Energy Savings (%)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: {summary_df["Energy Savings (%)"].mean():.2f}%')
        plt.xlabel('Simulation Number')
        plt.ylabel('Energy Savings (%)')
        plt.title(f'{season} Season Energy Savings Percentage')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_energy_savings_percentage.png'))
        plt.close()
        
        # Cost Savings Plot - Absolute Values
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Cost Savings ($)'])
        plt.axhline(y=summary_df['Cost Savings ($)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: ${summary_df["Cost Savings ($)"].mean():.2f}')
        plt.xlabel('Simulation Number')
        plt.ylabel('Cost Savings ($)')
        plt.title(f'{season} Season Cost Savings')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_cost_savings.png'))
        plt.close()
        
        # Cost Savings Plot - Percentage Values
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Cost Savings (%)'])
        plt.axhline(y=summary_df['Cost Savings (%)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: {summary_df["Cost Savings (%)"].mean():.2f}%')
        plt.xlabel('Simulation Number')
        plt.ylabel('Cost Savings (%)')
        plt.title(f'{season} Season Cost Savings Percentage')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_cost_savings_percentage.png'))
        plt.close()
        
        # Peak Energy Reduction Plot - Absolute Values
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Peak Energy Reduction (kWh)'])
        plt.axhline(y=summary_df['Peak Energy Reduction (kWh)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: {summary_df["Peak Energy Reduction (kWh)"].mean():.2f} kWh')
        plt.xlabel('Simulation Number')
        plt.ylabel('Peak Energy Reduction (kWh)')
        plt.title(f'{season} Season Peak Energy Reduction')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_peak_reduction.png'))
        plt.close()
        
        # Peak Energy Reduction Plot - Percentage Values
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['Simulation'], summary_df['Peak Energy Reduction (%)'])
        plt.axhline(y=summary_df['Peak Energy Reduction (%)'].mean(), color='r', linestyle='-', 
                    label=f'Mean: {summary_df["Peak Energy Reduction (%)"].mean():.2f}%')
        plt.xlabel('Simulation Number')
        plt.ylabel('Peak Energy Reduction (%)')
        plt.title(f'{season} Season Peak Energy Reduction Percentage')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{season}_peak_reduction_percentage.png'))
        plt.close()
        
        # Continue with other plots...
        # Create a summary image with key metrics
        create_summary_image(summary_df, season, plots_dir)
        
        # Try to create distribution plots
        try:
            create_distribution_plots(summary_df, season, plots_dir)
        except Exception as e:
            print(f"Could not create distribution plots: {str(e)}")
        
        print(f"Summary plots saved to: {plots_dir}")
    
    except Exception as e:
        print(f"Error creating summary plots: {str(e)}")
        import traceback
        traceback.print_exc()


def compare_results(season="Winter"):
    """
    Main function to compare baseline and controlled results across all simulations
    
    Args:
        season (str): Season for analysis ("Winter", "Spring", "Summer", "Fall")
    """
    try:
        # Define base directory
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        LARGE_SCALE_OUTPUT = os.path.join(BASE_DIR, 'outputs', 'LargeScale')
        
        # Check if the directory exists
        if not os.path.exists(LARGE_SCALE_OUTPUT):
            print(f"Error: LargeScale directory not found at {LARGE_SCALE_OUTPUT}")
            return
        
        # Compare simulations
        compare_simulation_results(LARGE_SCALE_OUTPUT, season)
        
    except Exception as e:
        print(f"\nError in compare_results: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    season = "Summer"  # Can be "Winter", "Spring", "Summer", or "Fall"
    compare_results(season)
