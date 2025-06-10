"""
=====================================================================
Simplified HPWH Control System (Controlled Operation Only)
=====================================================================

This module implements a simplified control system for Heat Pump Water Heaters (HPWHs)
that allows manual entry of CTA-2045-B-compatible commands with specified times and durations.

Only controlled operation is performed (no baseline comparison).

Author: Othman A. Murad
=====================================================================
"""

import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time

try:
    from ochre import Dwelling
    from bin.hpwh_model import dwelling_args
    OCHRE_AVAILABLE = True
except ImportError:
    print("Warning: OCHRE package not found. Simulation functionality will be limited.")
    OCHRE_AVAILABLE = False


def get_user_schedule():
    schedule_data = {}

    print("\nEnter Morning Load Up (LU) information:")
    use_morning_lu = input("Do you want to use Morning Load Up? (y/n): ").lower() == 'y'
    if use_morning_lu:
        schedule_data['M_LU_time'] = input("Start time (HH:MM): ")
        schedule_data['M_LU_duration'] = float(input("Duration (hours): "))
    else:
        schedule_data['M_LU_time'] = 'N/A'
        schedule_data['M_LU_duration'] = 0.0

    print("\nEnter Morning Shed (S) information:")
    use_morning_shed = input("Do you want to use Morning Shed? (y/n): ").lower() == 'y'
    if use_morning_shed:
        schedule_data['M_S_time'] = input("Start time (HH:MM): ")
        schedule_data['M_S_duration'] = float(input("Duration (hours): "))
    else:
        schedule_data['M_S_time'] = 'N/A'
        schedule_data['M_S_duration'] = 0.0

    print("\nEnter Evening Advanced Load Up (ALU) information:")
    use_evening_alu = input("Do you want to use Evening Advanced Load Up? (y/n): ").lower() == 'y'
    if use_evening_alu:
        schedule_data['E_ALU_time'] = input("Start time (HH:MM): ")
        schedule_data['E_ALU_duration'] = float(input("Duration (hours): "))
    else:
        schedule_data['E_ALU_time'] = 'N/A'
        schedule_data['E_ALU_duration'] = 0.0

    print("\nEnter Evening Shed (S) information:")
    use_evening_shed = input("Do you want to use Evening Shed? (y/n): ").lower() == 'y'
    if use_evening_shed:
        schedule_data['E_S_time'] = input("Start time (HH:MM): ")
        schedule_data['E_S_duration'] = float(input("Duration (hours): "))
    else:
        schedule_data['E_S_time'] = 'N/A'
        schedule_data['E_S_duration'] = 0.0

    return schedule_data


def get_water_heater_controls(hour_of_day, current_setpoint, schedule_data, **_):
    control = {
        'Water Heating': {
            'Setpoint': current_setpoint,
            'Deadband': 2.8,
            'Load Fraction': 1
        }
    }

    if schedule_data['M_LU_time'] != 'N/A':
        m_lu_time = pd.to_datetime(schedule_data['M_LU_time'], format='%H:%M').hour
        if m_lu_time <= hour_of_day < m_lu_time + schedule_data['M_LU_duration']:
            control['Water Heating']['Setpoint'] = 54.4

    if schedule_data['M_S_time'] != 'N/A':
        m_s_time = pd.to_datetime(schedule_data['M_S_time'], format='%H:%M').hour
        if m_s_time <= hour_of_day < m_s_time + schedule_data['M_S_duration']:
            control['Water Heating']['Setpoint'] = 48.9

    if schedule_data['E_ALU_time'] != 'N/A':
        e_alu_time = pd.to_datetime(schedule_data['E_ALU_time'], format='%H:%M').hour
        if e_alu_time <= hour_of_day < e_alu_time + schedule_data['E_ALU_duration']:
            control['Water Heating']['Setpoint'] = 62.8

    if schedule_data['E_S_time'] != 'N/A':
        e_s_time = pd.to_datetime(schedule_data['E_S_time'], format='%H:%M').hour
        if e_s_time <= hour_of_day < e_s_time + schedule_data['E_S_duration']:
            control['Water Heating']['Setpoint'] = 48.9

    return control


def run_simulation(schedule_data, results_dir):
    if not OCHRE_AVAILABLE:
        print("Error: OCHRE package not available. Cannot run simulation.")
        return None

    os.makedirs(results_dir, exist_ok=True)

    schedule_df = pd.DataFrame([schedule_data])
    schedule_file = os.path.join(results_dir, 'Schedule_SD.csv')
    schedule_df.to_csv(schedule_file, index=False)
    print(f"Control schedule saved to {schedule_file}")

    dwelling = Dwelling(name="Water Heater Control Test", **dwelling_args)
    water_heater = dwelling.get_equipment_by_end_use('Water Heating')

    for t in dwelling.sim_times:
        current_setpoint = water_heater.schedule.loc[t, 'Water Heating Setpoint (C)']
        control_signal = get_water_heater_controls(
            hour_of_day=t.hour,
            current_setpoint=current_setpoint,
            schedule_data=schedule_data
        )
        dwelling.update(control_signal=control_signal)

    df_controlled, _, _ = dwelling.finalize()

    # Filter only HPWH-relevant columns
    hpwh_columns = [
        "Hot Water Outlet Temperature (C)",
        "Hot Water Average Temperature (C)",
        "Water Heating Deadband Upper Limit (C)",
        "Water Heating Deadband Lower Limit (C)",
        "Water Heating Electric Power (kW)",
        "Hot Water Unmet Demand (kW)",
        "Hot Water Delivered (L/min)"
    ]
    df_controlled = df_controlled[hpwh_columns]
    df_controlled.to_csv(os.path.join(results_dir, 'SD-controlled_results.csv'))
    print(f"\nSimulation complete. Results saved to {results_dir}")

    # Plot with shaded control periods
    df_controlled['Timestamp'] = df_controlled.index
    df_controlled['Power_W'] = df_controlled['Water Heating Electric Power (kW)'] * 1000
    df_controlled['Energy_Wh'] = df_controlled['Power_W'] / 60
    total_energy_kwh = df_controlled['Energy_Wh'].sum() / 1000

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    date = df_controlled['Timestamp'].dt.date.iloc[0]

    def add_shade(start_str, duration_hr, color, label):
        if start_str != 'N/A':
            start_time = pd.to_datetime(start_str).time()
            end_time = (datetime.combine(date, start_time) + pd.Timedelta(hours=duration_hr)).time()
            ax1.axvspan(
                pd.Timestamp.combine(date, start_time),
                pd.Timestamp.combine(date, end_time),
                alpha=0.2, color=color, label=label
            )

    add_shade(schedule_data['M_LU_time'], schedule_data['M_LU_duration'], 'lightblue', 'Load-up Period')
    add_shade(schedule_data['M_S_time'], schedule_data['M_S_duration'], 'lightpink', 'Shed Period')
    add_shade(schedule_data['E_ALU_time'], schedule_data['E_ALU_duration'], 'lightgreen', 'Advanced Load-up Period')
    add_shade(schedule_data['E_S_time'], schedule_data['E_S_duration'], 'lightpink', 'Shed Period')

    ax1.plot(df_controlled['Timestamp'], df_controlled['Power_W'], label='Power (W)', color='red')
    ax2.plot(df_controlled['Timestamp'], df_controlled['Hot Water Outlet Temperature (C)'] * 9/5 + 32,
             label='Temperature (°F)', color='blue', linestyle='--')

    ax1.set_ylabel('Power (W)')
    ax2.set_ylabel('Temperature (°F)')
    ax1.set_xlabel('Time')
    ax1.grid(True, alpha=0.3)

    ax1.set_xlim(pd.Timestamp.combine(date, time(0, 0)), pd.Timestamp.combine(date, time(23, 59)))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Controlled HPWH Performance with CTA-2045 Schedule')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'hpwh_performance_controlled.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

    summary_path = os.path.join(results_dir, 'performance_summary_controlled.txt')
    with open(summary_path, 'w') as f:
        f.write("HPWH Controlled Operation Summary\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
        f.write(f"Total Energy Consumption: {total_energy_kwh:.3f} kWh\n")
    print(f"Summary saved to {summary_path}")

    return df_controlled


def main():
    print("\n===== HPWH Manual Control System =====\n")

    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    print("\n===== HPWH Control Schedule Input =====")
    print("Enter the start time and duration for each control period.")
    print("For times, use 24-hour format (HH:MM)")


    schedule_data = get_user_schedule()
    print("\nRunning OCHRE simulation with manual schedule...")
    df_controlled = run_simulation(schedule_data, results_dir)

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
