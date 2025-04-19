# HPWH-Model/single_day/analysis/compare_results.py

"""
=====================================================================
HPWH Single-Day Simulation Results Analysis Framework
=====================================================================

This module implements a detailed analysis system for evaluating the performance of 
price-signal-based control strategies for Heat Pump Water Heaters (HPWHs) in a single-day 
simulation. It processes OCHRE simulation outputs to compare baseline and controlled 
operation, providing visualizations and comprehensive performance metrics.

The analysis framework features:
1. CalFlexHub integration - Processes 24-hour CalFlexHub price signals to 
   calculate time-varying electricity costs and identify control periods
   
2. Multi-metric evaluation - Quantifies performance across three key dimensions:
   - Energy efficiency (hourly consumption, total usage)
   - Economic impact (hourly and total costs, savings percentage)
   - Grid support capability (peak period load reduction)
   
3. Period-specific analysis - Evaluates performance during distinct control periods:
   - Load-up periods: Standard preheating (130°F setpoint)
   - Advanced Load-up periods: Enhanced preheating (145°F setpoint)
   - Shed periods: Load reduction (120°F setpoint)
   
4. High-resolution visualization - Creates detailed 24-hour operation plots showing:
   - Power consumption profiles for baseline and controlled cases
   - Tank temperature dynamics in Fahrenheit
   - 15-minute resolution water draw patterns
   - Hourly price signals with color-coded control period indicators
   
5. Comprehensive reporting - Generates detailed text reports with:
   - Hourly breakdown of energy use and costs
   - Period-specific performance metrics
   - Energy shifting efficiency calculations
   - Overall cost and consumption summaries

This analysis framework processes the results from Case Study I where a fixed water 
draw profile was used to evaluate control performance for a single day with different 
seasonal price patterns. The visualization and analysis highlight the temporal relationship 
between price signals, hot water usage, and the resulting control actions.

Functions:
    load_control_schedule: Loads schedule data from CSV files
    create_timestamps: Creates minute-resolution timestamps
    normalize_timestamps: Adds timestamps to simulation data
    calculate_hourly_data: Calculates hourly metrics and costs
    create_power_temperature_plot: Creates power and temperature plots
    create_water_flow_price_plot: Creates water flow and price plots
    setup_plot_formatting: Formats plots for readability
    save_comparison_results: Generates detailed text reports
    compare_results: Main function coordinating overall analysis

Usage:
    python sd_results_analysis.py
    
    The season can be modified in the main function to analyze different
    seasonal simulation results (Winter, Spring, Summer, Fall).

Author: Othman A. Murad
=====================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time, timedelta


def load_control_schedule(schedule_file):
    """
    Load control periods from schedule file
    """
    try:
        schedule_df = pd.read_csv(schedule_file)
        
        # Convert times to datetime and calculate end times based on durations
        control_periods = {}
        
        # Morning Load-up
        if schedule_df['M_LU_time'].iloc[0] != 'N/A':
            m_lu_start = pd.to_datetime(schedule_df['M_LU_time'].iloc[0]).time()
            m_lu_duration = schedule_df['M_LU_duration'].iloc[0]
            control_periods['load_up_1'] = (
                schedule_df['M_LU_time'].iloc[0],
                (pd.to_datetime(schedule_df['M_LU_time'].iloc[0]) + 
                 pd.Timedelta(hours=m_lu_duration)).strftime('%H:%M')
            )
        
        # Morning Shed
        if schedule_df['M_S_time'].iloc[0] != 'N/A':
            m_s_start = pd.to_datetime(schedule_df['M_S_time'].iloc[0]).time()
            m_s_duration = schedule_df['M_S_duration'].iloc[0]
            control_periods['shed_1'] = (
                schedule_df['M_S_time'].iloc[0],
                (pd.to_datetime(schedule_df['M_S_time'].iloc[0]) + 
                 pd.Timedelta(hours=m_s_duration)).strftime('%H:%M')
            )
        
        # Evening Advanced Load-up
        if schedule_df['E_ALU_time'].iloc[0] != 'N/A':
            e_alu_start = pd.to_datetime(schedule_df['E_ALU_time'].iloc[0]).time()
            e_alu_duration = schedule_df['E_ALU_duration'].iloc[0]
            control_periods['advanced_load_up'] = (
                schedule_df['E_ALU_time'].iloc[0],
                (pd.to_datetime(schedule_df['E_ALU_time'].iloc[0]) + 
                 pd.Timedelta(hours=e_alu_duration)).strftime('%H:%M')
            )
        
        # Evening Shed
        if schedule_df['E_S_time'].iloc[0] != 'N/A':
            e_s_start = pd.to_datetime(schedule_df['E_S_time'].iloc[0]).time()
            e_s_duration = schedule_df['E_S_duration'].iloc[0]
            control_periods['shed_2'] = (
                schedule_df['E_S_time'].iloc[0],
                (pd.to_datetime(schedule_df['E_S_time'].iloc[0]) + 
                 pd.Timedelta(hours=e_s_duration)).strftime('%H:%M')
            )
        
        return control_periods
    except Exception as e:
        print(f"Error loading control schedule: {str(e)}")
        raise


def create_timestamps():
    """
    Create timestamps for a full day at minute intervals
    """
    base_date = datetime.now().date()
    timestamps = []
    for hour in range(24):
        for minute in range(60):
            timestamps.append(datetime.combine(base_date, time(hour, minute)))
    return timestamps

def normalize_timestamps(df):
    """
    Add normalized timestamps to the dataframe
    """
    timestamps = create_timestamps()
    df['Timestamp'] = timestamps
    return df

def calculate_hourly_data(df, price_df):
    """
    Calculate hourly averages for energy consumption, water flow, and cost
    """
    try:
        # Convert power from kW to W for consistency
        df['Power_W'] = df['Water Heating Electric Power (kW)'] * 1000

        # Calculate time differences and energy
        df['time_diff'] = 1/60  # one minute in hours
        df['interval_energy'] = df['Power_W'] * df['time_diff'] / 1000  # Convert Wh to kWh

        # Calculate water volume per minute (L)
        df['water_volume'] = df['Hot Water Delivered (L/min)'] * 1  # 1 minute intervals

        # Add quarter-hour calculation for water flow visualization
        df['quarter_hour'] = df['Timestamp'].dt.minute // 15

        # Group by hour and calculate aggregates
        hourly_energy = df.groupby(df['Timestamp'].dt.hour)['interval_energy'].sum()

        # Calculate cost for each hour
        hourly_costs = []
        hourly_prices = []
        total_cost = 0

        for hour in range(24):
            energy = hourly_energy.get(hour, 0)
            price = float(price_df[price_df['start_time'].dt.hour == hour]['price'].iloc[0])
            hour_cost = energy * price

            hourly_costs.append(hour_cost)
            hourly_prices.append(price)
            total_cost += hour_cost

        hourly_data = pd.DataFrame({
            'energy': hourly_energy,
            'water_flow': df.groupby(df['Timestamp'].dt.hour)['Hot Water Delivered (L/min)'].mean(),
            'water_volume': df.groupby(df['Timestamp'].dt.hour)['water_volume'].sum(),
            'cost': hourly_costs,
            'price': hourly_prices
        })

        return hourly_data, df, total_cost
    except Exception as e:
        print(f"Error in calculate_hourly_data: {str(e)}")
        raise

def create_power_temperature_plot(ax1, baseline_df, controlled_df, date, control_periods):
    """
    Create power and temperature comparison plot
    """
    # Create twin axis for temperature
    ax1_twin = ax1.twinx()

    # Add control period shading
    for period_type, (start, end) in control_periods.items():
        start_time = pd.Timestamp.combine(date, pd.to_datetime(start).time())
        end_time = pd.Timestamp.combine(date, pd.to_datetime(end).time())
        
        if 'advanced_load_up' in period_type:
            color = 'lightgreen'
            label = 'Advanced Load-up Period' if '1' in period_type else None
        elif 'load_up' in period_type:
            color = 'lightblue'
            label = 'Load-up Period' if '1' in period_type else None
        else:  # shed period
            color = 'lightpink'
            label = 'Shed Period' if '1' in period_type else None
            
        ax1.axvspan(start_time, end_time, alpha=0.2, color=color, label=label)

    # Plot Power
    line1 = ax1.plot(baseline_df['Timestamp'], baseline_df['Power_W'], 
                     color='blue', label='Baseline Power', alpha=0.5, linewidth=1)
    line2 = ax1.plot(controlled_df['Timestamp'], controlled_df['Power_W'],
                     color='red', label='Controlled Power', linewidth=1)
    ax1.set_ylabel('Power (W)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.2)

    # Plot Temperature
    line3 = ax1_twin.plot(baseline_df['Timestamp'], 
                         baseline_df['Hot Water Outlet Temperature (C)'] * 9/5 + 32,
                         color='blue', linestyle='--', label='Baseline Temperature', 
                         alpha=0.5, linewidth=1)
    line4 = ax1_twin.plot(controlled_df['Timestamp'],
                         controlled_df['Hot Water Outlet Temperature (C)'] * 9/5 + 32,
                         color='red', linestyle='--', label='Controlled Temperature', 
                         linewidth=1)
    ax1_twin.set_ylabel('Temperature (°F)', color='black')
    ax1_twin.tick_params(axis='y', labelcolor='black')
    ax1_twin.set_ylim(bottom=110, top=150)

    return line1, line2, line3, line4, ax1_twin

def create_water_flow_price_plot(ax2, baseline_df, baseline_hourly, date, hourly_timestamps):
    """
    Create water flow and price plot with 15-minute resolution water flow data
    """
    L_TO_GAL = 0.264172
    
    # Calculate 15-minute water flow data
    quarter_hour_key = baseline_df['Timestamp'].dt.hour * 4 + baseline_df['quarter_hour']
    quarter_hour_water_flow = (baseline_df.groupby(quarter_hour_key)['Hot Water Delivered (L/min)']
                             .mean() * L_TO_GAL)
    
    # Ensure all 96 quarter-hour periods exist (24 hours × 4 quarters)
    full_day_quarters = pd.Series(index=range(96), data=0.0)
    quarter_hour_water_flow = quarter_hour_water_flow.combine_first(full_day_quarters)
    
    # Sort by quarter-hour index
    quarter_hour_water_flow = quarter_hour_water_flow.sort_index().values
    
    # Create 15-minute timestamps for water flow plotting
    quarter_hour_timestamps = []
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            quarter_hour_timestamps.append(pd.Timestamp.combine(date, time(hour, minute)))
    
    # Plot Water Flow at 15-minute resolution
    line5 = ax2.plot(quarter_hour_timestamps, quarter_hour_water_flow,
             color='green', label='Water Flow', linewidth=1,
             drawstyle='steps-post')
    ax2.set_ylabel('Water Flow (gal/min)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid(True, alpha=0.2)
    ax2.set_xlabel('Time (h)')
    
    # Add Price to second subplot with twin axis (hourly resolution)
    ax2_twin = ax2.twinx()
    color_price = 'orange'
    line6 = ax2_twin.plot(hourly_timestamps, baseline_hourly['price'],
                       color=color_price, label='Price', linewidth=1,
                       drawstyle='steps-post')
    ax2_twin.set_ylabel('Price ($/kWh)', color='black')
    ax2_twin.tick_params(axis='y', labelcolor='black')
    
    # Add legends for water flow and price only
    ax2.legend(line5, ['Water Flow'], loc='upper left')
    ax2_twin.legend(line6, ['Price'], loc='upper right')
    
    return line6, ax2_twin

def setup_plot_formatting(fig, ax1, ax2, ax2_twin, date, line1, line2, line3, line4, line5):
    """
    Set up the formatting for plots
    """
    # Set x-axis format for both subplots
    for ax in [ax1, ax2]:
        ax.set_xlim(pd.Timestamp.combine(date, time(0, 0)),
                   pd.Timestamp.combine(date, time(23, 59)))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Hide x-axis labels for top subplot
    ax1.set_xticklabels([])
    ax2.set_xlabel('Time (h)')

    # Combine legends for top plot only
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    
    # Add control period labels to legend
    handles = [plt.Rectangle((0,0),1,1, color='lightblue', alpha=0.2),
              plt.Rectangle((0,0),1,1, color='lightgreen', alpha=0.2),
              plt.Rectangle((0,0),1,1, color='lightpink', alpha=0.2)]
    ax1.legend(lines + handles, 
              labels + ['Load-up Period', 'Advanced Load-up Period', 'Shed Period'], 
              loc='upper left')

    # Set title
    plt.suptitle(f'24-Hour Performance Comparison: Baseline vs. Controlled HPWH Operation - {season} Season')
    plt.tight_layout()

def save_comparison_results(baseline_hourly, controlled_hourly, baseline_cost, controlled_cost, output_path, control_periods):
    """
    Save comparison analysis results to a text file with dynamic control periods
    """
    L_TO_GAL = 0.264172
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("Single Day Water Heater Performance Comparison\n")
        f.write(f"Analysis Date: {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        
        # Hourly Breakdown
        f.write("Hourly Energy and Cost Comparison:\n")
        f.write("Hour | Baseline (kWh) | Controlled (kWh) | Savings (kWh) | Savings (%)\n")
        f.write("-" * 75 + "\n")

        for hour in range(24):
            baseline_energy = baseline_hourly.loc[hour, 'energy']
            controlled_energy = controlled_hourly.loc[hour, 'energy']
            energy_savings = baseline_energy - controlled_energy
            savings_percent = (energy_savings / baseline_energy * 100) if baseline_energy != 0 else 0

            f.write(f"{hour:02d}:00 | {baseline_energy:13.3f} | {controlled_energy:15.3f} | "
                   f"{energy_savings:12.3f} | {savings_percent:10.1f}\n")

        # Write period-specific analyses
        f.write("\nDR PERFORMANCE ANALYSIS:\n")
        f.write("=" * 50 + "\n\n")

        # 1. Shed Period Analysis
        f.write("SHED PERIOD ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        
        total_shed_baseline = 0
        total_shed_controlled = 0
        
        for period in ['shed_1', 'shed_2']:
            start_time = control_periods[period][0]
            end_time = control_periods[period][1]
            start_hour = int(start_time.split(':')[0])
            end_hour = int(end_time.split(':')[0])
            
            period_baseline = baseline_hourly.loc[start_hour:end_hour-1, 'energy'].sum()
            period_controlled = controlled_hourly.loc[start_hour:end_hour-1, 'energy'].sum()
            period_reduction = period_baseline - period_controlled
            reduction_percent = (period_reduction / period_baseline * 100) if period_baseline != 0 else 0
            
            total_shed_baseline += period_baseline
            total_shed_controlled += period_controlled
            
            f.write(f"\n{period} ({start_time}-{end_time}):\n")
            f.write(f"  Baseline Energy: {period_baseline:.3f} kWh\n")
            f.write(f"  Controlled Energy: {period_controlled:.3f} kWh\n")
            f.write(f"  Energy Reduction: {period_reduction:.3f} kWh ({reduction_percent:.1f}%)\n")

        total_shed_reduction = total_shed_baseline - total_shed_controlled
        total_reduction_percent = (total_shed_reduction / total_shed_baseline * 100) if total_shed_baseline != 0 else 0
        
        f.write("\nTotal Shed Period Performance:\n")
        f.write(f"Total Energy Reduction: {total_shed_reduction:.3f} kWh\n")
        f.write(f"Overall Reduction Percentage: {total_reduction_percent:.1f}%\n")

        # 2. Advanced Load-up Analysis
        f.write("\nADVANCED LOAD-UP ANALYSIS:\n")
        f.write("-" * 25 + "\n")

        alu_time = control_periods['advanced_load_up'][0]
        alu_duration = control_periods['advanced_load_up'][1]
        alu_start = int(alu_time.split(':')[0])
        # Fix the end time calculation
        alu_end = int(alu_time.split(':')[0]) + 2  # Since it's a 2-hour period (12:00-14:00)

        alu_baseline = baseline_hourly.loc[alu_start:alu_end-1, 'energy'].sum()
        alu_controlled = controlled_hourly.loc[alu_start:alu_end-1, 'energy'].sum()
        alu_increase = alu_controlled - alu_baseline
        increase_percent = (alu_increase / alu_baseline * 100) if alu_baseline != 0 else 0
        
        f.write(f"\nAdvanced Load-up Period ({alu_time}-{alu_end:02d}:00):\n")
        f.write(f"  Baseline Energy: {alu_baseline:.3f} kWh\n")
        f.write(f"  Controlled Energy: {alu_controlled:.3f} kWh\n")
        f.write(f"  Energy Increase: {alu_increase:.3f} kWh ({increase_percent:.1f}%)\n")

        # 3. Combined Impact Analysis
        f.write("\nCOMBINED IMPACT ANALYSIS:\n")
        f.write("-" * 23 + "\n")
        f.write("Energy Shifting Strategy Performance:\n")
        f.write(f"Total Energy Shifted in Shed Periods: {total_shed_reduction:.3f} kWh\n")
        f.write(f"Total Additional Energy Used in ALU: {alu_increase:.3f} kWh\n")
        net_impact = total_shed_reduction - alu_increase
        f.write(f"Net Energy Impact: {net_impact:.3f} kWh\n")
        shifting_efficiency = total_shed_reduction / alu_increase if alu_increase != 0 else 0
        f.write(f"Shifting Efficiency Ratio: {shifting_efficiency:.2f}\n\n")

        # Overall Summary
        total_baseline_energy = baseline_hourly['energy'].sum()
        total_controlled_energy = controlled_hourly['energy'].sum()
        total_energy_savings = total_baseline_energy - total_controlled_energy
        total_energy_savings_percent = (total_energy_savings / total_baseline_energy * 100)
        
        total_baseline_water = baseline_hourly['water_volume'].sum() * L_TO_GAL
        total_controlled_water = controlled_hourly['water_volume'].sum() * L_TO_GAL

        f.write("\nOVERALL SUMMARY:\n")
        f.write("-" * 30 + "\n")
        
        f.write("Energy Consumption:\n")
        f.write(f"  Baseline: {total_baseline_energy:.3f} kWh\n")
        f.write(f"  Controlled: {total_controlled_energy:.3f} kWh\n")
        f.write(f"  Total Savings: {total_energy_savings:.3f} kWh ({total_energy_savings_percent:.1f}%)\n\n")
        
        f.write("Cost Analysis:\n")
        f.write(f"  Baseline Cost: ${baseline_cost:.2f}\n")
        f.write(f"  Controlled Cost: ${controlled_cost:.2f}\n")
        f.write(f"  Cost Savings: ${baseline_cost - controlled_cost:.2f}\n")
        f.write(f"  Percentage Savings: {((baseline_cost - controlled_cost) / baseline_cost * 100):.1f}%\n\n")

        f.write("Water Usage:\n")
        f.write(f"  Baseline: {total_baseline_water:.2f} gallons\n")
        f.write(f"  Controlled: {total_controlled_water:.2f} gallons\n")
        f.write(f"  Difference: {total_controlled_water - total_baseline_water:.2f} gallons\n")


def compare_results(season="Spring"):
    """
    Main function to compare baseline and controlled results
    """
    try:
        # Define paths
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        BASELINE_DATA = os.path.join(BASE_DIR, 'outputs/baseline/SD-baseline_results.csv')
        CONTROLLED_DATA = os.path.join(BASE_DIR, 'outputs/controlled/SD-controlled_results.csv')
        PRICE_DATA = os.path.join(BASE_DIR, 'analysis',f'{season}_24Hrs-results.csv')
        SCHEDULE_DATA = os.path.join(BASE_DIR, 'outputs/schedule', f'Schedule_SD_{season}.csv')
        
        # Create analysis output directory
        ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis/results')
        os.makedirs(ANALYSIS_DIR, exist_ok=True)

        # Load control periods from schedule
        control_periods = load_control_schedule(SCHEDULE_DATA)
        print(f"Loaded control periods from schedule: {control_periods}")

        # Read and process data
        baseline_df = normalize_timestamps(pd.read_csv(BASELINE_DATA))
        controlled_df = normalize_timestamps(pd.read_csv(CONTROLLED_DATA))
        date = baseline_df['Timestamp'].dt.date.iloc[0]

        # Process price data
        price_df = pd.read_csv(PRICE_DATA)
        price_df['start_time'] = pd.to_datetime(price_df['start_time'])
        price_df['start_time'] = price_df['start_time'] - pd.Timedelta(hours=8)

        # Calculate hourly data for both cases
        baseline_hourly, baseline_df, baseline_cost = calculate_hourly_data(baseline_df, price_df)
        controlled_hourly, controlled_df, controlled_cost = calculate_hourly_data(controlled_df, price_df)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
        plt.subplots_adjust(hspace=0.3)

        # Create hourly timestamps for price plot
        hourly_timestamps = [pd.Timestamp.combine(date, time(hour, 0)) for hour in range(24)]

        line1, line2, line3, line4, ax1_twin = create_power_temperature_plot(
            ax1, baseline_df, controlled_df, date, control_periods)
        line5, ax2_twin = create_water_flow_price_plot(
            ax2, baseline_df, baseline_hourly, date, hourly_timestamps)

        # Add control period shading to both plots
        for ax in [ax1, ax2]:
            for period_type, (start, end) in control_periods.items():
                start_time = pd.Timestamp.combine(date, pd.to_datetime(start).time())
                end_time = pd.Timestamp.combine(date, pd.to_datetime(end).time())
                
                if 'advanced_load_up' in period_type:
                    color = 'lightgreen'
                    label = 'Advanced Load-up Period' if '1' in period_type else None
                elif 'load_up' in period_type:
                    color = 'lightblue'
                    label = 'Load-up Period' if '1' in period_type else None
                else:  # shed period
                    color = 'lightpink'
                    label = 'Shed Period' if '1' in period_type else None
                
                ax.axvspan(start_time, end_time, alpha=0.2, color=color, label=label)
                ax.grid(True, alpha=0.2)

        # Set up plot formatting
        setup_plot_formatting(fig, ax1, ax2, ax2_twin, date, line1, line2, line3, line4, line5)

        # Save plots
        plots_dir = os.path.join(BASE_DIR, 'outputs/analysis_plots')
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(os.path.join(plots_dir, 'baseline_vs_controlled_comparison.png'), 
                    bbox_inches='tight')
        plt.show()

        # Save comparison results
        results_file = os.path.join(ANALYSIS_DIR, f'comparison_results-{season}.txt')
        save_comparison_results(baseline_hourly, controlled_hourly, baseline_cost, 
                              controlled_cost, results_file, control_periods)
        
        print(f"Analysis completed. Results saved to: {results_file}")

    except Exception as e:
        print(f"\nError in compare_results: {str(e)}")
        import traceback
        traceback.print_exc()

# Example usage:
if __name__ == "__main__":
    season = "Winter"
    compare_results(season)
