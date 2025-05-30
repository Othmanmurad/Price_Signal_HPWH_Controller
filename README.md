# Price-Signal-Based HPWH Controller

This repository contains the code and input files used to simulate a price-responsive control strategy for Heat Pump Water Heaters (HPWHs) using the [OCHRE](https://github.com/NREL/OCHRE) simulation framework.

---

## Getting Started

### 1. Install OCHRE

To begin, install the OCHRE platform by following the steps in the official user tutorial:

[OCHRE User Tutorial (Google Colab)](https://colab.research.google.com/github/NREL/OCHRE/blob/main/notebook/user_tutorial.ipynb)

> **Note:** It is highly recommended to install OCHRE locally (e.g., in Visual Studio Code) rather than running it in Colab. All simulations in this repository were performed using a local setup.

---

### 2. Download Input Files

The repository includes input files required to run the simulations:

- `.xml`: Defines the building and equipment properties.
- `.csv`: Contains the water usage schedule.
- `.epw`: Includes weather data in EnergyPlus Weather format.

Make sure to place these files in the correct directories as specified in each script.

---
### 3. Download the HPWH model
-  `hpwh_model.py`: HPWH model script that has to be placed in the same directory as the `run_dwelling.py` script placed in the default path of OCHRE


### 4. Run the Controller

The controller includes both single-day and multi-day simulation options:

#### Single-Day Simulation
-  `sd_price_based_control.py`: A basic script that runs a simulation using one water draw profile.

#### Multi-Day Simulation
- `md_price_based_control.py`: Simulates using a single water draw profile across multiple days.
- `md_large_scale_simulations.py`: Simulates using eight diverse water draw profiles for a broader evaluation.

---

### 5. Analyze Results

Once a simulation is complete:

- Move the output results to your preferred analysis environment.
- Run the appropriate result analysis script (provided in this repo) to assess:
  - Energy savings
  - Cost reduction
  - Load shifting performance
  - Statistical significance

---

---

## Additional OCHRE Resources

For more information about OCHRE, visit the following links:

-  [NREL OCHRE Project Page](https://www.nrel.gov/grid/ochre)
-  [OCHRE GitHub Repository](https://github.com/NREL/OCHRE)
-  [OCHRE Documentation](https://ochre-nrel.readthedocs.io/en/latest/index.html)

