# HPWH-Model/multi_day/analysis/comparison_15mins.py


"""
=====================================================================
HPWH 14-Day Simulation Results Analysis Framework
=====================================================================

This module implements a comprehensive analysis system for evaluating the performance of 
price-signal-based control strategies for Heat Pump Water Heaters (HPWHs) over 14-day 
simulation periods. It processes simulation outputs from OCHRE, compares baseline and 
controlled operation, and generates detailed visualizations and statistical reports.

The analysis framework features:
1. CalFlexHub integration - Imports and processes CalFlexHub price signals to calculate 
   time-varying electricity costs and identify price-responsive control periods
   
2. Multi-metric evaluation - Quantifies performance across three key dimensions:
   - Energy efficiency (daily consumption, percentage change)
   - Economic impact (electricity costs, savings percentage)
   - Grid support capability (peak period load reduction)
   
3. Statistical robustness - Calculates means and standard deviations for all metrics
   to characterize performance consistency across the simulation period
   
4. High-resolution visualization - Creates detailed daily operation plots showing:
   - Power consumption profiles for baseline and controlled cases
   - Tank temperature dynamics with smoothed visualization
   - 15-minute resolution water draw patterns
   - Real-time price signals with control period indicators
   
5. Performance reporting - Generates comprehensive text reports with:
   - Day-by-day tabular comparisons
   - Statistical summaries for all metrics
   - Overall performance indicators

This analysis framework processes the results from Case Study I in which a fixed water draw 
profile was used to evaluate control performance across baseline, perfect knowledge, 
day-ahead, and two-day-ahead scenarios. The code accommodates seasonal variation and can 
process different control schedules with time-varying price signals.

Functions:
    load_control_schedule: Loads schedule data from CSV files
    calculate_period_energy: Calculates energy use during specific periods
    analyze_load_shifting: Analyzes load shifting during shed periods
    calculate_daily_data: Calculates daily totals for energy and costs
    create_power_temperature_plot: Creates power and temperature plots
    create_water_flow_plot: Creates water flow and price plots
    plot_single_day_comparison: Creates daily comparison visualizations
    create_daily_comparison_plots: Creates all daily comparison plots
    create_total_energy_plot: Creates energy summary plot
    create_shed_period_plots: Creates shed period analysis plots
    print_analysis_results: Generates detailed text reports
    compare_results: Main function coordinating overall analysis

Usage:
    python md_results_analysis.py
    
    The season can be modified in the main function to analyze different
    seasonal simulation results (Winter, Spring, Summer, Fall).

Author: Othman A. Murad
=====================================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time, timedelta
import numpy as np

def load_control_schedule(season="Winter"):
    """
    Load and process control schedule from Schedule_MD_{season}.csv
    """
    try:
        schedule_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   #f'outputs/schedule/Schedule_MD_{season}.csv')
                                   f'outputs/schedule/Schedule_MD_{season}_forecast.csv')

        schedule_df = pd.read_csv(schedule_path)

        control_periods = {}
        for day in range(len(schedule_df)):
            day_schedule = schedule_df.iloc[day]
            control_periods[day] = {
                'M_LU': (day_schedule['M_LU_time'], day_schedule['M_LU_duration']),
                'M_S': (day_schedule['M_S_time'], day_schedule['M_S_duration']),
                'E_ALU': (day_schedule['E_ALU_time'], day_schedule['E_ALU_duration']),
                'E_S': (day_schedule['E_S_time'], day_schedule['E_S_duration'])
            }
        return control_periods
    except Exception as e:
        print(f"Error loading control schedule: {str(e)}")
        raise

def calculate_period_energy(df, day, period_time, period_duration):
    """Calculate energy consumption for a specific period"""
    try:
        # Handle various skip conditions
        if period_time == 'N/A' or pd.isna(period_time):
            return 0
            
        # Convert period_duration to float to ensure it's a number
        try:
            period_duration = float(period_duration)
        except (ValueError, TypeError):
            return 0
            
        # Skip periods with zero or negative duration
        if period_duration <= 0:
            return 0
            
        # Convert time string to time object
        period_start = pd.to_datetime(period_time).time()
        period_end_dt = (datetime.combine(datetime.min, period_start) + 
                        timedelta(hours=period_duration)).time()
        
        # Filter data for the specific day and time period
        day_mask = df['day'] == day
        
        # Handle periods spanning midnight
        if period_end_dt < period_start:
            time_mask = ((df['Timestamp'].dt.time >= period_start) | 
                        (df['Timestamp'].dt.time < period_end_dt))
        else:
            time_mask = ((df['Timestamp'].dt.time >= period_start) & 
                        (df['Timestamp'].dt.time < period_end_dt))
        
        period_energy = df[day_mask & time_mask]['interval_energy'].sum()
        return period_energy
        
    except Exception as e:
        print(f"Error calculating period energy for period at {period_time}: {str(e)}")
        return 0

def analyze_load_shifting(baseline_df, controlled_df, control_periods):
    """Analyze load shifting during shed periods and energy consumption during advanced load-up"""
    load_shift_results = {
        'day': [], 'period': [],
        'baseline_energy': [], 'controlled_energy': [],
        'energy_reduction': [], 'reduction_percentage': []
    }
    
    advanced_loadup_results = {
        'day': [], 
        'baseline_energy': [], 'controlled_energy': [],
        'energy_increase': [], 'increase_percentage': []
    }
    
    total_baseline_shed = 0
    total_controlled_shed = 0
    total_baseline_alu = 0
    total_controlled_alu = 0
    
    for day in range(min(14, len(control_periods))):
        periods = control_periods[day]
        
        # Analyze morning and evening shed periods
        for period_name in ['M_S', 'E_S']:
            period_time, period_duration = periods[period_name]
            
            # Convert period_duration to float for comparison
            try:
                period_duration_float = float(period_duration)
            except (ValueError, TypeError):
                period_duration_float = 0
            
            # Only process periods with positive duration
            if period_time != 'N/A' and period_duration_float > 0:
                baseline_energy = calculate_period_energy(
                    baseline_df, day, period_time, period_duration)
                controlled_energy = calculate_period_energy(
                    controlled_df, day, period_time, period_duration)
                
                energy_reduction = baseline_energy - controlled_energy
                reduction_percentage = (energy_reduction / baseline_energy * 100 
                                     if baseline_energy > 0 else 0)
                
                load_shift_results['day'].append(day + 1)
                load_shift_results['period'].append(period_name)
                load_shift_results['baseline_energy'].append(baseline_energy)
                load_shift_results['controlled_energy'].append(controlled_energy)
                load_shift_results['energy_reduction'].append(energy_reduction)
                load_shift_results['reduction_percentage'].append(reduction_percentage)
                
                total_baseline_shed += baseline_energy
                total_controlled_shed += controlled_energy
        
        # Analyze evening advanced load-up period
        alu_time, alu_duration = periods['E_ALU']
        
        # Convert ALU duration to float for comparison
        try:
            alu_duration_float = float(alu_duration)
        except (ValueError, TypeError):
            alu_duration_float = 0
            
        if alu_time != 'N/A' and alu_duration_float > 0:
            baseline_alu = calculate_period_energy(
                baseline_df, day, alu_time, alu_duration)
            controlled_alu = calculate_period_energy(
                controlled_df, day, alu_time, alu_duration)
            
            energy_increase = controlled_alu - baseline_alu
            increase_percentage = (energy_increase / baseline_alu * 100 
                                 if baseline_alu > 0 else 0)
            
            advanced_loadup_results['day'].append(day + 1)
            advanced_loadup_results['baseline_energy'].append(baseline_alu)
            advanced_loadup_results['controlled_energy'].append(controlled_alu)
            advanced_loadup_results['energy_increase'].append(energy_increase)
            advanced_loadup_results['increase_percentage'].append(increase_percentage)
            
            total_baseline_alu += baseline_alu
            total_controlled_alu += controlled_alu
    
    # Create DataFrames for results
    shift_df = pd.DataFrame(load_shift_results)
    alu_df = pd.DataFrame(advanced_loadup_results)
    
    # Calculate overall statistics
    total_reduction = total_baseline_shed - total_controlled_shed
    total_percentage = (total_reduction / total_baseline_shed * 100 
                       if total_baseline_shed > 0 else 0)
    
    # Calculate advanced load-up statistics
    total_alu_increase = total_controlled_alu - total_baseline_alu
    total_alu_percentage = (total_alu_increase / total_baseline_alu * 100 
                           if total_baseline_alu > 0 else 0)
    
    return (shift_df, total_reduction, total_percentage, 
            alu_df, total_alu_increase, total_alu_percentage)
  
def calculate_daily_data(df, price_df):
    """Calculate daily totals for energy consumption, water flow, and cost"""
    try:
        # Calculate interval energy (1-minute intervals)
        df['interval_energy'] = df['Power_W'] * (1/60) / 1000  # Convert Wh to kWh
        df['water_volume'] = df['Hot Water Delivered (L/min)'] * 1  # 1-minute intervals

        # Create minute-level price data
        minute_prices = []
        for day in range(14):
            # Get prices for this day
            day_prices = price_df[price_df['day'] == day]['price'].values
            # Repeat each hourly price 60 times for minutes
            day_minute_prices = np.repeat(day_prices, 60)
            minute_prices.extend(day_minute_prices)
        
        # Add minute-level prices to main dataframe
        df['price'] = minute_prices[:len(df)]  # Ensure length matches
        
        # Calculate minute-by-minute costs
        df['interval_cost'] = df['interval_energy'] * df['price']

        # Initialize daily data
        days = range(14)
        daily_data = pd.DataFrame(index=days)
        
        # Calculate daily totals
        for day in days:
            day_mask = df['day'] == day
            day_df = df[day_mask]
            
            daily_data.loc[day, 'energy'] = day_df['interval_energy'].sum()
            daily_data.loc[day, 'water_flow'] = day_df['Hot Water Delivered (L/min)'].mean()
            daily_data.loc[day, 'water_volume'] = day_df['water_volume'].sum()
            daily_data.loc[day, 'cost'] = day_df['interval_cost'].sum()
            daily_data.loc[day, 'avg_price'] = day_df['price'].mean()

        total_cost = daily_data['cost'].sum()

        return daily_data, df, total_cost

    except Exception as e:
        print(f"Error in calculate_daily_data: {str(e)}")
        raise

def add_control_periods(ax, date, day_control_periods):
    """Add control period shading to plot"""
    for period_type, (start_time, duration) in day_control_periods.items():
        if start_time != 'N/A':
            try:
                start = pd.Timestamp.combine(date, pd.to_datetime(start_time).time())
                end = start + pd.Timedelta(hours=duration)
                
                if period_type == 'E_ALU':
                    color = 'lightgreen'
                    label = 'Advanced Load-up Period' if period_type.endswith('1') else None
                elif period_type == 'M_LU':
                    color = 'lightblue'
                    label = 'Load-up Period' if period_type.endswith('1') else None
                else:  # Shed periods (M_S or E_S)
                    color = 'lightpink'
                    label = 'Shed Period' if period_type.endswith('1') else None
                
                ax.axvspan(start, end, alpha=0.2, color=color, label=label)
                
            except Exception as e:
                print(f"Error adding control period {period_type}: {str(e)}")
                continue

def create_power_temperature_plot(ax1, baseline_df, controlled_df):
    """Create power and temperature comparison plot"""
    # Plot power for both cases
    line1 = ax1.plot(baseline_df['Timestamp'], baseline_df['Power_W'],
                     color='blue', label='Baseline Power', linewidth=1)
    line2 = ax1.plot(controlled_df['Timestamp'], controlled_df['Power_W'],
                     color='red', label='Controlled Power', linewidth=1)
    ax1.set_ylabel('Power (W)')

    # Plot temperature on twin axis
    ax1_twin = ax1.twinx()
    
    # Process and plot temperatures
    baseline_temp_F = baseline_df['Hot Water Outlet Temperature (C)'] * 9/5 + 32
    controlled_temp_F = controlled_df['Hot Water Outlet Temperature (C)'] * 9/5 + 32
    
    baseline_temp_smooth = pd.Series(baseline_temp_F).ewm(span=20, adjust=False).mean()
    controlled_temp_smooth = pd.Series(controlled_temp_F).ewm(span=20, adjust=False).mean()
    
    line3 = ax1_twin.plot(baseline_df['Timestamp'], baseline_temp_smooth,
                         color='blue', label='Baseline Temp', linewidth=1)
    line4 = ax1_twin.plot(controlled_df['Timestamp'], controlled_temp_smooth,
                         color='red', label='Controlled Temp', linewidth=1)
    ax1_twin.set_ylabel('Temperature (°F)')
    ax1_twin.set_ylim(bottom=110, top=150)

    return line1, line2, line3, line4, ax1_twin

def create_water_flow_plot(ax2, quarter_hour_timestamps, quarter_hour_water_flow, price_df):
    """
    Create water flow and price plot with 15-minute resolution water flow data
    """
    # Plot Water Flow
    color_water = 'green'
    line5 = ax2.plot(quarter_hour_timestamps, quarter_hour_water_flow,
                    color=color_water, label='Water Flow', linewidth=1.5,
                    drawstyle='steps-post')
    ax2.set_ylabel('Water Flow (gal/min)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_xlabel('Time (h)')
    
    # Add Price to second subplot with twin axis
    ax2_twin = ax2.twinx()
    color_price = 'orange'
    
    # Get the date from the timestamp
    date = quarter_hour_timestamps[0].date()
    
    # Create properly formatted price data with datetime index
    # First create hourly timestamps
    hourly_timestamps = [datetime.combine(date, time(hour, 0)) for hour in range(24)]
    
    # Create a DataFrame with the hourly price data and datetime index
    price_data = pd.DataFrame({'price': price_df['price'].values}, index=hourly_timestamps)
    
    # Add a final point at 23:59:59 to extend the price curve to the end of the day
    last_price = price_data['price'].iloc[-1]
    price_data.loc[datetime.combine(date, time(23, 59, 59))] = last_price
    
    # Sort the index to ensure proper order
    price_data = price_data.sort_index()
    
    # Plot the price data
    line6 = ax2_twin.plot(price_data.index, price_data['price'],
                       color=color_price, label='Price', linewidth=1.5,
                       drawstyle='steps-post')
    ax2_twin.set_ylabel('Price ($/kWh)', color='black')
    ax2_twin.tick_params(axis='y', labelcolor='black')
    
    # Set price axis range - ensure minimum upper bound is 0.5
    current_max = ax2_twin.get_ylim()[1]
    ax2_twin.set_ylim(0, max(current_max, 0.5))
    
    # Add legends for water flow and price only
    ax2.legend(line5, ['Water Flow'], loc='upper left')
    ax2_twin.legend(line6, ['Price'], loc='upper right')
    
    return line5, line6, ax2_twin

def setup_plot_formatting(fig, ax1, ax2, date, lines1, lines2, lines3, lines4, lines5, lines6):
    """Set up plot formatting with updated legend handling"""
    for ax in [ax1, ax2]:
        # Set x-axis limits from 00:00 to 23:59
        ax.set_xlim(datetime.combine(date, time(0, 0)),
                   datetime.combine(date, time(23, 59)))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.2)

    ax1.set_xticklabels([])

    # Combine all lines for legend
    all_lines = lines1 + lines2 + lines3 + lines4
    labels = [l.get_label() for l in all_lines]
    
    # Add control period legend elements
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.2) 
              for c in ['lightblue', 'lightgreen', 'lightpink']]
    labels.extend(['Load-up Period', 'Advanced Load-up Period', 'Shed Period'])
    
    # Create legend
    ax1.legend(all_lines + handles, labels, loc='upper left', ncol=1)

def plot_single_day_comparison(baseline_df, controlled_df, price_df, day_number, 
                             control_periods, output_dir, fig_title="HPWH Baseline and Controlled Operation Analysis Comparison"):
    """Create comparison plot for a single day with 15-minute water flow resolution"""
    try:
        # Filter data for the specific day
        day_mask = lambda df: df['day'] == day_number
        baseline_day = baseline_df[day_mask(baseline_df)]
        controlled_day = controlled_df[day_mask(controlled_df)]
        date = baseline_day['Timestamp'].dt.date.iloc[0]

        # Add quarter-hour to both dataframes
        baseline_day['minute'] = baseline_day['Timestamp'].dt.minute
        baseline_day['quarter_hour'] = baseline_day['minute'] // 15
        
        # Calculate 15-minute water flow data
        L_TO_GAL = 0.264172
        
        # Create a unique key for each 15-minute period (0-95 for a full day)
        quarter_hour_key = baseline_day['hour'] * 4 + baseline_day['quarter_hour']
        
        # Calculate average water flow for each 15-minute period
        quarter_hour_water_flow = (baseline_day.groupby(quarter_hour_key)['Hot Water Delivered (L/min)']
                               .mean() * L_TO_GAL)
        
        # Ensure we have all 96 15-minute periods (24 hours × 4 quarters)
        full_day_quarters = pd.Series(index=range(96), data=0.0)
        quarter_hour_water_flow = quarter_hour_water_flow.combine_first(full_day_quarters)
        
        # Sort by quarter-hour index to ensure proper order
        quarter_hour_water_flow = quarter_hour_water_flow.sort_index().values
        
        # Create timestamps for 15-minute intervals
        quarter_hour_timestamps = []
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                quarter_hour_timestamps.append(datetime.combine(date, time(hour, minute)))

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
        plt.subplots_adjust(hspace=0.3)

        # Add control period shading
        for ax in [ax1, ax2]:
            add_control_periods(ax, date, control_periods[day_number])
            ax.grid(True, alpha=0.2)

        # Create power and temperature plot
        lines1, lines2, lines3, lines4, ax1_twin = create_power_temperature_plot(
            ax1, baseline_day, controlled_day)
        
        # Get price data for the day
        day_price_mask = price_df['day'] == day_number
        day_price_data = price_df[day_price_mask].copy()
        
        # Create water flow and price plot
        lines5, lines6, ax2_twin = create_water_flow_plot(
            ax2, quarter_hour_timestamps, quarter_hour_water_flow, day_price_data)
        
        # Set up plot formatting
        setup_plot_formatting(fig, ax1, ax2, date, 
                            lines1, lines2, lines3, lines4,
                            lines5, lines6)
        
        # Add title
        fig.suptitle(f'{fig_title} - Day {day_number+1}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'day_{day_number+1}_comparison.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error plotting day {day_number}: {str(e)}")
        plt.close()

def create_daily_comparison_plots(baseline_df, controlled_df, price_df, control_periods, output_dir, season):
    """
    Create comparison plots for all days
    """
    for day in range(14):
        plot_single_day_comparison(
            baseline_df, controlled_df, price_df, day, control_periods, output_dir,
            fig_title=f"HPWH Baseline and Controlled Operation Analysis Comparison - {season} Season")

def create_total_energy_plot(baseline_daily, controlled_daily, plots_dir, season):
    """Create plot for total daily energy consumption"""
    plt.figure(figsize=(12, 6))
    
    days = range(1, len(baseline_daily) + 1)
    x = np.array(days)
    width = 0.35
    
    plt.bar(x - width/2, baseline_daily['energy'], width, 
           label='Baseline', color='blue', alpha=0.8)
    plt.bar(x + width/2, controlled_daily['energy'], width, 
           label='Controlled', color='red', alpha=0.8)
    
    plt.xlabel('Day')
    plt.ylabel('Energy Consumption (kWh)')
    plt.title(f'HPWH Daily Energy Consumption Comparison - {season} Season')
    plt.xticks(x, [f'Day {i}' for i in days])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'total_energy_comparison.png'))
    plt.close()

def create_shed_period_plots(shift_df, plots_dir, season):
    """Create plots for shed periods with handling for missing periods"""
    # Group by day
    days = shift_df['day'].unique()
    
    # Check if we have any morning and evening shed data
    has_morning_data = 'M_S' in shift_df['period'].values
    has_evening_data = 'E_S' in shift_df['period'].values
    
    # Width for bar plots
    width = 0.35
    
    # Morning shed plot - only if data exists
    if has_morning_data:
        plt.figure(figsize=(12, 6))
        morning_data = shift_df[shift_df['period'] == 'M_S']
        x = np.array(morning_data['day'])
        
        plt.bar(x - width/2, morning_data['baseline_energy'], width,
               label='Baseline', color='blue', alpha=0.8)
        plt.bar(x + width/2, morning_data['controlled_energy'], width,
               label='Controlled', color='red', alpha=0.8)
        
        plt.xlabel('Day')
        plt.ylabel('Energy Consumption (kWh)')
        plt.title(f'HPWH Morning Shed Period Energy Consumption - {season} Season')
        plt.xticks(x, [f'Day {i}' for i in x])
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'morning_shed_comparison.png'))
        plt.close()
    else:
        print(f"No morning shed periods found for {season} season, skipping morning shed plot.")
    
    # Evening shed plot - only if data exists
    if has_evening_data:
        plt.figure(figsize=(12, 6))
        evening_data = shift_df[shift_df['period'] == 'E_S']
        x = np.array(evening_data['day'])
        
        plt.bar(x - width/2, evening_data['baseline_energy'], width,
               label='Baseline', color='blue', alpha=0.8)
        plt.bar(x + width/2, evening_data['controlled_energy'], width,
               label='Controlled', color='red', alpha=0.8)
        
        plt.xlabel('Day')
        plt.ylabel('Energy Consumption (kWh)')
        plt.title(f'HPWH Evening Shed Period Energy Consumption - {season} Season')
        plt.xticks(x, [f'Day {i}' for i in x])
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'evening_shed_comparison.png'))
        plt.close()
    else:
        print(f"No evening shed periods found for {season} season, skipping evening shed plot.")
    
    # Combined shed periods plot - only create if we have any data
    if len(shift_df) > 0:
        plt.figure(figsize=(12, 6))
        
        # Group by day and sum the energies
        combined_data = shift_df.groupby('day').agg({
            'baseline_energy': 'sum',
            'controlled_energy': 'sum',
            'energy_reduction': 'sum'
        }).reset_index()
        
        x = np.array(combined_data['day'])
        
        plt.bar(x - width/2, combined_data['baseline_energy'], width,
               label='Baseline', color='blue', alpha=0.8)
        plt.bar(x + width/2, combined_data['controlled_energy'], width,
               label='Controlled', color='red', alpha=0.8)
        
        plt.xlabel('Day')
        plt.ylabel('Energy Consumption (kWh)')
        plt.title(f'HPWH Combined Shed Periods Energy Consumption - {season} Season')
        plt.xticks(x, [f'Day {i}' for i in x])
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'combined_shed_comparison.png'))
        plt.close()
    else:
        print(f"No shed periods found for {season} season, skipping combined shed plot.")


def print_analysis_results(baseline_daily, controlled_daily, baseline_cost, controlled_cost, shift_df, output_file, season="Winter"):
    """
    Print reorganized analysis results to both console and file with handling for missing periods
    """
    # Calculate statistics for the analysis
    L_TO_GAL = 0.264172
    
    # Daily energy statistics
    baseline_energy = baseline_daily['energy'].values
    controlled_energy = controlled_daily['energy'].values
    energy_diff = baseline_energy - controlled_energy
    energy_diff_pct = energy_diff / baseline_energy * 100
    
    energy_stats = {
        "baseline_mean": np.mean(baseline_energy),
        "baseline_std": np.std(baseline_energy),
        "controlled_mean": np.mean(controlled_energy),
        "controlled_std": np.std(controlled_energy),
        "diff_mean": np.mean(energy_diff),
        "diff_std": np.std(energy_diff),
        "diff_pct_mean": np.mean(energy_diff_pct),
        "diff_pct_std": np.std(energy_diff_pct)
    }
    
    # Daily cost statistics
    baseline_cost_daily = baseline_daily['cost'].values
    controlled_cost_daily = controlled_daily['cost'].values
    cost_diff = baseline_cost_daily - controlled_cost_daily
    cost_diff_pct = cost_diff / baseline_cost_daily * 100
    
    cost_stats = {
        "baseline_mean": np.mean(baseline_cost_daily),
        "baseline_std": np.std(baseline_cost_daily),
        "controlled_mean": np.mean(controlled_cost_daily),
        "controlled_std": np.std(controlled_cost_daily),
        "diff_mean": np.mean(cost_diff),
        "diff_std": np.std(cost_diff),
        "diff_pct_mean": np.mean(cost_diff_pct),
        "diff_pct_std": np.std(cost_diff_pct)
    }
    
    # Check if we have shed period data
    has_shed_data = len(shift_df) > 0
    
    # Shed period energy statistics - only if we have data
    if has_shed_data:
        # Group by day to get daily total shed energy
        shed_daily = shift_df.groupby('day').agg({
            'baseline_energy': 'sum',
            'controlled_energy': 'sum',
            'energy_reduction': 'sum'
        })
        
        shed_baseline = shed_daily['baseline_energy'].values
        shed_controlled = shed_daily['controlled_energy'].values
        shed_reduction = shed_daily['energy_reduction'].values
        shed_reduction_pct = shed_reduction / shed_baseline * 100
        
        shed_stats = {
            "baseline_mean": np.mean(shed_baseline),
            "baseline_std": np.std(shed_baseline),
            "controlled_mean": np.mean(shed_controlled),
            "controlled_std": np.std(shed_controlled),
            "reduction_mean": np.mean(shed_reduction),
            "reduction_std": np.std(shed_reduction),
            "reduction_pct_mean": np.mean(shed_reduction_pct),
            "reduction_pct_std": np.std(shed_reduction_pct)
        }
    
    # Write to file
    with open(output_file, 'w') as f:
        # 1. Daily Energy Consumption Comparison
        f.write(f"Daily Energy Consumption Comparison - {season} Season:\n")
        f.write("Day | Baseline (kWh) | Controlled (kWh) | Energy Difference (kWh) | Difference (%)\n")
        f.write("-" * 80 + "\n")
        
        for day in range(len(baseline_daily)):
            baseline_day_energy = baseline_daily.loc[day, 'energy']
            controlled_day_energy = controlled_daily.loc[day, 'energy']
            energy_diff = baseline_day_energy - controlled_day_energy
            energy_diff_pct = (energy_diff / baseline_day_energy * 100) if baseline_day_energy > 0 else 0
            
            f.write(f"{day+1:3d} | {baseline_day_energy:13.3f} | {controlled_day_energy:15.3f} | "
                  f"{energy_diff:22.3f} | {energy_diff_pct:13.1f}\n")
        
        f.write("\n")  # Add blank line after table
        
        # 2. Daily Cost Comparison
        f.write(f"Daily Cost Comparison - {season} Season:\n")
        f.write("Day | Baseline Cost ($) | Controlled Cost ($) | Cost Difference ($) | Difference (%)\n")
        f.write("-" * 80 + "\n")
        
        for day in range(len(baseline_daily)):
            baseline_day_cost = baseline_daily.loc[day, 'cost']
            controlled_day_cost = controlled_daily.loc[day, 'cost']
            cost_diff = baseline_day_cost - controlled_day_cost
            cost_diff_pct = (cost_diff / baseline_day_cost * 100) if baseline_day_cost > 0 else 0
            
            f.write(f"{day+1:3d} | {baseline_day_cost:16.2f} | {controlled_day_cost:18.2f} | "
                  f"{cost_diff:19.2f} | {cost_diff_pct:13.1f}\n")
        
        f.write("\n")  # Add blank line after table
        
        # 3. Daily Peak-Time Energy Consumption Comparison (Shed Periods) - only if we have data
        if has_shed_data:
            combined_data = shift_df.groupby('day').agg({
                'baseline_energy': 'sum',
                'controlled_energy': 'sum',
                'energy_reduction': 'sum'
            }).reset_index()
            
            f.write(f"Daily Peak-Time Energy Consumption Comparison (Shed Periods) - {season} Season:\n")
            f.write("Day | Baseline (kWh) | Controlled (kWh) | Reduction (kWh) | Reduction (%)\n")
            f.write("-" * 80 + "\n")
            
            for _, row in combined_data.iterrows():
                day = int(row['day'])
                baseline_shed = row['baseline_energy']
                controlled_shed = row['controlled_energy']
                reduction = row['energy_reduction']
                reduction_pct = (reduction / baseline_shed * 100) if baseline_shed > 0 else 0
                
                f.write(f"{day:3d} | {baseline_shed:13.3f} | {controlled_shed:15.3f} | "
                      f"{reduction:14.3f} | {reduction_pct:12.1f}\n")
            
            f.write("\n")  # Add blank line after table
        else:
            f.write(f"No active shed periods found for {season} season.\n\n")
        
        # 4. Summary Analysis with Statistics
        f.write(f"\nSummary Analysis with Statistics - {season} Season:\n")
        f.write("=" * 60 + "\n\n")
        
        # Energy consumption statistics
        f.write("Energy Consumption Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Baseline Energy: {energy_stats['baseline_mean']:.3f} ± {energy_stats['baseline_std']:.3f} kWh (mean ± std)\n")
        f.write(f"Controlled Energy: {energy_stats['controlled_mean']:.3f} ± {energy_stats['controlled_std']:.3f} kWh (mean ± std)\n")
        f.write(f"Energy Difference: {energy_stats['diff_mean']:.3f} ± {energy_stats['diff_std']:.3f} kWh (mean ± std)\n")
        f.write(f"Percentage Difference: {energy_stats['diff_pct_mean']:.1f} ± {energy_stats['diff_pct_std']:.1f}% (mean ± std)\n\n")
        
        # Cost statistics
        f.write("Cost Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Baseline Cost: ${cost_stats['baseline_mean']:.2f} ± ${cost_stats['baseline_std']:.2f} (mean ± std)\n")
        f.write(f"Controlled Cost: ${cost_stats['controlled_mean']:.2f} ± ${cost_stats['controlled_std']:.2f} (mean ± std)\n")
        f.write(f"Cost Savings: ${cost_stats['diff_mean']:.2f} ± ${cost_stats['diff_std']:.2f} (mean ± std)\n")
        f.write(f"Percentage Savings: {cost_stats['diff_pct_mean']:.1f} ± {cost_stats['diff_pct_std']:.1f}% (mean ± std)\n\n")
        
        # Shed period statistics - only if we have data
        if has_shed_data:
            f.write("Peak-Time (Shed Period) Energy Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Baseline Shed Energy: {shed_stats['baseline_mean']:.3f} ± {shed_stats['baseline_std']:.3f} kWh (mean ± std)\n")
            f.write(f"Controlled Shed Energy: {shed_stats['controlled_mean']:.3f} ± {shed_stats['controlled_std']:.3f} kWh (mean ± std)\n")
            f.write(f"Energy Reduction: {shed_stats['reduction_mean']:.3f} ± {shed_stats['reduction_std']:.3f} kWh (mean ± std)\n")
            f.write(f"Percentage Reduction: {shed_stats['reduction_pct_mean']:.1f} ± {shed_stats['reduction_pct_std']:.1f}% (mean ± std)\n\n")
        
        # Overall totals
        total_baseline_energy = baseline_daily['energy'].sum()
        total_controlled_energy = controlled_daily['energy'].sum()
        total_energy_diff = total_baseline_energy - total_controlled_energy
        
        total_baseline_water = (baseline_daily['water_volume'].sum() * L_TO_GAL)
        total_controlled_water = (controlled_daily['water_volume'].sum() * L_TO_GAL)
        
        f.write("Overall Totals:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Energy: Baseline = {total_baseline_energy:.3f} kWh, Controlled = {total_controlled_energy:.3f} kWh\n")
        f.write(f"Energy Difference: {total_energy_diff:.3f} kWh ({total_energy_diff/total_baseline_energy*100:.1f}%)\n\n")
        
        f.write(f"Total Cost: Baseline = ${baseline_cost:.2f}, Controlled = ${controlled_cost:.2f}\n")
        f.write(f"Cost Savings: ${baseline_cost - controlled_cost:.2f} ({(baseline_cost - controlled_cost) / baseline_cost * 100:.1f}%)\n\n")
        
        f.write(f"Total Water Usage: Baseline = {total_baseline_water:.2f} gallons, Controlled = {total_controlled_water:.2f} gallons\n")
        f.write(f"Water Difference: {total_controlled_water - total_baseline_water:.2f} gallons\n\n")
        
        # Shed period totals - only if we have data
        if has_shed_data:
            total_shed_baseline = combined_data['baseline_energy'].sum()
            total_shed_controlled = combined_data['controlled_energy'].sum()
            total_shed_reduction = combined_data['energy_reduction'].sum()
            
            f.write(f"Total Shed Period Energy: Baseline = {total_shed_baseline:.3f} kWh, Controlled = {total_shed_controlled:.3f} kWh\n")
            f.write(f"Total Shed Period Reduction: {total_shed_reduction:.3f} kWh ({total_shed_reduction/total_shed_baseline*100:.1f}%)\n")
        
    # Print to console summary
    print(f"\nAnalysis Summary - {season} Season:")
    print("=" * 40)
    
    print("\nDaily Energy Consumption Comparison (mean ± std):")
    print(f"Baseline: {energy_stats['baseline_mean']:.3f} ± {energy_stats['baseline_std']:.3f} kWh")
    print(f"Controlled: {energy_stats['controlled_mean']:.3f} ± {energy_stats['controlled_std']:.3f} kWh")
    print(f"Difference: {energy_stats['diff_mean']:.3f} ± {energy_stats['diff_std']:.3f} kWh ({energy_stats['diff_pct_mean']:.1f} ± {energy_stats['diff_pct_std']:.1f}%)")
    
    print("\nDaily Cost Comparison (mean ± std):")
    print(f"Baseline: ${cost_stats['baseline_mean']:.2f} ± ${cost_stats['baseline_std']:.2f}")
    print(f"Controlled: ${cost_stats['controlled_mean']:.2f} ± ${cost_stats['controlled_std']:.2f}")
    print(f"Savings: ${cost_stats['diff_mean']:.2f} ± ${cost_stats['diff_std']:.2f} ({cost_stats['diff_pct_mean']:.1f} ± {cost_stats['diff_pct_std']:.1f}%)")
    
    # Shed period statistics - only if we have data
    if has_shed_data:
        print("\nPeak-Time Energy Reduction (mean ± std):")
        print(f"Baseline: {shed_stats['baseline_mean']:.3f} ± {shed_stats['baseline_std']:.3f} kWh")
        print(f"Controlled: {shed_stats['controlled_mean']:.3f} ± {shed_stats['controlled_std']:.3f} kWh")
        print(f"Reduction: {shed_stats['reduction_mean']:.3f} ± {shed_stats['reduction_std']:.3f} kWh ({shed_stats['reduction_pct_mean']:.1f} ± {shed_stats['reduction_pct_std']:.1f}%)")
    else:
        print("\nNo active shed periods found for this season.")

def compare_results(season="Winter"):
    """
    Main function to compare baseline and controlled results
    """
    try:
        # Define paths
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        BASELINE_DATA = os.path.join(BASE_DIR, 'outputs/baseline/MD-baseline_results.csv')
        #CONTROLLED_DATA = os.path.join(BASE_DIR, 'outputs/controlled/MD-controlled_results.csv')
        CONTROLLED_DATA = os.path.join(BASE_DIR, 'outputs/controlled/MD-controlled_forecast_results.csv')
        PRICE_DATA = os.path.join(BASE_DIR, f'inputs/{season}_14Days-results.csv')
        
        # Create output directories
        PLOTS_DIR = os.path.join(BASE_DIR, 'outputs/analysis_plots/comparison')
        ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis/results')
        os.makedirs(PLOTS_DIR, exist_ok=True)
        os.makedirs(ANALYSIS_DIR, exist_ok=True)

        # Load and process data
        print(f"\nReading and processing data files for {season} season...")
        baseline_df = pd.read_csv(BASELINE_DATA)
        controlled_df = pd.read_csv(CONTROLLED_DATA)
        
        # Process price data
        price_df = pd.read_csv(PRICE_DATA)
        date_col = f'{season.lower()}_period_data_origination_date'
        price_col = f'{season.lower()}_period_price'
        price_df = price_df.rename(columns={
            date_col: 'timestamp',
            price_col: 'price'
        })

        # Process timestamps and days
        for df in [baseline_df, controlled_df]:
            df['Timestamp'] = pd.to_datetime(df['Time'])
            df['day'] = df.index // 1440  # 1440 minutes per day
            df['Power_W'] = df['Water Heating Electric Power (kW)'] * 1000
            df['hour'] = df['Timestamp'].dt.hour
            df['minute'] = df['Timestamp'].dt.minute
            df['quarter_hour'] = df['minute'] // 15

        price_df['Timestamp'] = pd.to_datetime(price_df['timestamp'])
        price_df['day'] = price_df.index // 24
        price_df['hour'] = price_df['Timestamp'].dt.hour

        # Load control schedule
        control_periods = load_control_schedule(season)

        # Calculate metrics and create summary plots
        print("\nCalculating performance metrics...")
        baseline_daily, baseline_df, baseline_cost = calculate_daily_data(baseline_df, price_df)
        controlled_daily, controlled_df, controlled_cost = calculate_daily_data(controlled_df, price_df)

        print("\nAnalyzing load shifting performance...")
        shift_df, total_reduction, total_percentage, alu_df, total_alu_increase, total_alu_percentage = \
            analyze_load_shifting(baseline_df, controlled_df, control_periods)

        # Create daily comparison plots
        print("\nGenerating daily comparison plots...")
        create_daily_comparison_plots(
            baseline_df, controlled_df, price_df, control_periods, PLOTS_DIR, season)

        # Generate reports and plots
        results_file = os.path.join(ANALYSIS_DIR, f'comparison_results_{season}.txt')
        
        print("\nGenerating analysis reports...")
        # Save main comparison results with the new format
        print_analysis_results(baseline_daily, controlled_daily, baseline_cost, 
                           controlled_cost, shift_df, results_file, season)

        print("\nCreating summary plots...")
        # Create total energy comparison plot
        create_total_energy_plot(baseline_daily, controlled_daily, PLOTS_DIR, season)
        
        # Create shed period plots (morning, evening, and combined)
        create_shed_period_plots(shift_df, PLOTS_DIR, season)

        print(f"\nAnalysis completed successfully for {season} season.")
        print(f"Results saved to: {results_file}")
        print(f"Plots saved in: {PLOTS_DIR}")

    except Exception as e:
        print(f"\nError in compare_results: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    season = "Winter"  # Change to "Summer", "Spring", or "Fall" as needed
    compare_results(season)
