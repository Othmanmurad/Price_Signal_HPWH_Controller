"""
=====================================================================
Large-Scale Multi-Simulation Framework for 
HPWH Control Strategy Evaluation
=====================================================================

This script implements a batch processing system to evaluate the price-signal-based control 
strategy for Heat Pump Water Heaters (HPWHs) across multiple seasons, water draw profiles, 
and start dates. It systematically executes the day-ahead control algorithm to generate a 
statistically significant dataset for validating control performance under diverse conditions.

The framework orchestrates:
1. Multiple water draw profiles - Executes simulations across 8 different water draw 
   schedules representing diverse household usage patterns
   
2. Multiple start dates - For each schedule, runs 6 different start dates per season 
   (1st and 15th of each month), capturing intra-seasonal variations
   
3. Seasonal testing - Organizes simulations by season (Winter, Spring, Summer, Fall)
   to evaluate performance across different weather conditions and price patterns
   
4. Systematic result collection - Creates an organized directory structure for storing
   and analyzing simulation outputs

The batch processor manages the simulation environment by:
- Importing the day-ahead control algorithm from actual_forecast_MD.py
- Temporarily modifying dwelling parameters for each simulation
- Disabling visualization during batch execution for performance
- Capturing runtime statistics and error information
- Maintaining a clean directory structure

This framework generated the 192 annual simulations (48 per season) used in Case Study II 
to evaluate the control strategy's robustness across diverse residential usage patterns.
Performance metrics validated through this large-scale simulation include:
- Energy consumption changes (6.5-7.5% increase across seasons)
- Electricity cost savings (11.6-32.9% across seasons)
- Peak-period load reductions (86.3-90.7% across seasons)

Functions:
    get_fixed_start_dates: Generates simulation start dates for a given season
    setup_directories: Creates the directory structure for storing results
    run_seasonal_simulations: Executes simulations for one season
    main: Coordinates the overall batch execution process

Usage:
    python md_large_scale_simulations.py
    
    season: Optional parameter to specify which season to simulate
            (Winter, Spring, Summer, Fall). Defaults to Fall.

Dependencies:
    The script requires the actual_forecast_MD.py module containing the
    day-ahead control algorithm implementation and the run_dwelling.py
    module for OCHRE simulation parameters.

Author: Othman A. Murad
=====================================================================
"""


import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
import time
import shutil

# Add the OCHRE bin directory to the path if needed
sys.path.append('/Users/othmanmurad/Documents/OCHRE/bin')

# Import from existing scripts
from run_dwelling import dwelling_args
from actual_forecast_MD import (
    read_and_process_data,
    get_forecast_and_actual_data,
    generate_control_schedules,
    run_simulation,
    visualize_daily_schedule_comparison
)

# Constants
NUM_SIMULATIONS = 1
SIMULATION_DAYS = 14
RESULTS_DIR = "/Users/othmanmurad/Documents/OCHRE/results/Large-scale_wf"

# Define seasonal date ranges
SEASONAL_RANGES = {
    "Winter": {
        "start": dt.datetime(2019, 1, 1),
        "end": dt.datetime(2019, 3, 31)
    },
    "Spring": {
        "start": dt.datetime(2019, 4, 1),
        "end": dt.datetime(2019, 6, 30)
    },
    "Summer": {
        "start": dt.datetime(2019, 7, 1),
        "end": dt.datetime(2019, 9, 30)
    },
    "Fall": {
        "start": dt.datetime(2019, 10, 1),
        "end": dt.datetime(2019, 12, 31)
    }
}

def get_fixed_start_dates(season):
    season_months = {
        "Winter": [1, 2, 3],
        "Spring": [4, 5, 6],
        "Summer": [7, 8, 9],
        "Fall": [10, 11, 12]
    }
    year = SEASONAL_RANGES[season]['start'].year
    months = season_months[season]
    return [dt.datetime(year, m, d) for m in months for d in [1, 15]]

def setup_directories(season):
    season_dir = os.path.join(RESULTS_DIR, season)
    os.makedirs(season_dir, exist_ok=True)
    for i in range(1, NUM_SIMULATIONS + 1):
        sim_dir = os.path.join(season_dir, f"simulation_{i}")
        os.makedirs(sim_dir, exist_ok=True)
    return season_dir

def run_seasonal_simulations(season):
    print(f"Running large-scale simulations for {season} season...")
    season_dir = setup_directories(season)
    temp_plots_dir = os.path.join(season_dir, "temp_plots")
    os.makedirs(temp_plots_dir, exist_ok=True)

    input_dir = "/Users/othmanmurad/Documents/OCHRE/ochre/defaults/Input Files"
    price_file_path = os.path.join(input_dir, f"{season}_14Days-results.csv")
    base_schedule_path = os.path.join(input_dir, "schedules_{}.csv")

    start_dates = get_fixed_start_dates(season)
    simulation_counter = 1

    from actual_forecast_MD import dwelling_args as forecast_dwelling_args
    original_start_time = forecast_dwelling_args['start_time']
    original_duration = forecast_dwelling_args['duration']
    original_schedule_file = forecast_dwelling_args['schedule_input_file'] if 'schedule_input_file' in forecast_dwelling_args else None

    for schedule_num in range(1, 9):
        schedule_file = base_schedule_path.format(schedule_num)
        for start_date in start_dates:
            print(f"\nStarting Simulation {simulation_counter}/{NUM_SIMULATIONS}")
            print(f"Start Date: {start_date.strftime('%Y-%m-%d')} | Schedule: schedules_{schedule_num}.csv")
            sim_dir = os.path.join(season_dir, f"simulation_{simulation_counter}")
            start_time = time.time()

            try:
                # Set the schedule_input_file in the global forecast_dwelling_args
                forecast_dwelling_args['start_time'] = start_date
                forecast_dwelling_args['duration'] = dt.timedelta(days=SIMULATION_DAYS)
                forecast_dwelling_args['schedule_input_file'] = schedule_file

                print("Processing price data and generating control schedules...")
                grouped_data = read_and_process_data(price_file_path, season)

                def dummy_visualize(*args, **kwargs):
                    pass

                import actual_forecast_MD
                original_visualize = visualize_daily_schedule_comparison
                actual_forecast_MD.visualize_daily_schedule_comparison = dummy_visualize

                all_schedules = generate_control_schedules(grouped_data, season, temp_plots_dir)
                actual_forecast_MD.visualize_daily_schedule_comparison = original_visualize

                schedule_df = pd.DataFrame(all_schedules)
                schedule_path = os.path.join(sim_dir, "control_schedule.csv")
                schedule_df.to_csv(schedule_path, index=False)

                load_up_hours = 0
                shed_hours = 0
                if 'M_LU_duration' in schedule_df.columns:
                    load_up_hours += schedule_df['M_LU_duration'].sum()
                if 'E_ALU_duration' in schedule_df.columns:
                    load_up_hours += schedule_df['E_ALU_duration'].sum()
                if 'M_S_duration' in schedule_df.columns:
                    shed_hours += schedule_df['M_S_duration'].sum()
                if 'E_S_duration' in schedule_df.columns:
                    shed_hours += schedule_df['E_S_duration'].sum()

                df_controlled, df_baseline = run_simulation(schedule_df, season)

                baseline_energy = df_baseline['Water Heating Electric Power (kW)'].sum() / 60
                controlled_energy = df_controlled['Water Heating Electric Power (kW)'].sum() / 60
                energy_savings = baseline_energy - controlled_energy
                energy_savings_percent = (energy_savings / baseline_energy) * 100 if baseline_energy > 0 else 0

                df_baseline.to_csv(os.path.join(sim_dir, f"baseline_results.csv"))
                df_controlled.to_csv(os.path.join(sim_dir, f"controlled_results.csv"))

                end_time = time.time()
                runtime = end_time - start_time

                print(f"Simulation {simulation_counter} completed in {runtime:.2f} seconds")
                simulation_counter += 1

            except Exception as e:
                print(f"Error in simulation {simulation_counter}: {str(e)}")
                import traceback
                traceback.print_exc()

    forecast_dwelling_args['start_time'] = original_start_time
    forecast_dwelling_args['duration'] = original_duration
    if original_schedule_file is not None:
        forecast_dwelling_args['schedule_input_file'] = original_schedule_file

    print(f"\nAll {season} simulations completed.")
    print(f"Results stored in: {season_dir}")

def main(season):
    if season not in SEASONAL_RANGES:
        valid_seasons = list(SEASONAL_RANGES.keys())
        print(f"Error: Invalid season '{season}'. Please choose from: {valid_seasons}")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Starting large-scale simulation for {season} season...")
    print(f"Will run {NUM_SIMULATIONS} simulations using 8 schedules × 6 start dates")
    print(f"Results will be stored in: {os.path.join(RESULTS_DIR, season)}")

    try:
        run_seasonal_simulations(season)
        temp_plots_dir = os.path.join(RESULTS_DIR, season, "temp_plots")
        if os.path.exists(temp_plots_dir):
            shutil.rmtree(temp_plots_dir)
        print("\nProcess completed successfully!")
    except Exception as e:
        print(f"\nError in main process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        season = sys.argv[1]
    else:
        season = "Fall"         # Can be "Winter", "Spring", "Summer", or "Fall"
    main(season)
