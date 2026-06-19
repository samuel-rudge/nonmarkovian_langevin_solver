# Postprocessing

This document describes the **postprocessing stage** of the workflow. Postprocessing converts raw trajectory data and electronic-force datasets into physically meaningful observables, diagnostic plots, and voltage-resolved steady-state results.

Postprocessing is logically divided into two independent but complementary components:

1. **Dynamical postprocessing** (transient + steady-state statistics)
2. **Electronic-force postprocessing** (position-dependent forces and decompositions)

Both operate on data generated during preprocessing and propagation.

---

## Overview of Postprocessing Tasks

### Dynamical postprocessing
- Computes ensemble-averaged observables from propagated trajectories
- Diagnoses whether vibrational steady state has been reached
- Produces steady-state transport observables (e.g. current–voltage data)

### Electronic-force postprocessing
- Visualizes precomputed electronic forces
- Plots Markovian friction and noise
- Plots non-Markovian decomposition weights and frequencies

These two tasks are run via **separate scripts** and can be executed independently.

Throughout this document, paths are written schematically as:

results / ... / voltage_<voltage>eV / ...

The placeholder `...` denotes the **system-specific directory hierarchy** defined by the preprocessing stage, namely:

data_identifier /  
gamma_<molecule_lead_coupling>meV_temp_<temperature>K /  
delta_<Delta>eV /  
tier_<electronic_tier> /

This directory structure is constructed automatically from `config/settings.yaml` and is shared consistently across preprocessing, propagation, and postprocessing.

Before reading this document, the user is expected to be familiar with:

- `electronic_force_dataset_spec.md` — definition of electronic-force datasets
- `preprocess.md` — generation and decomposition of electronic forces
- `propagate.md` — Markovian and non-Markovian trajectory propagation

---

## Dynamical Postprocessing: Steady-State and Transient Analysis

### Script

```bash

python3 -m scripts.ss_postprocessing_pipeline

``` 

## Usage

Postprocessing is performed using a single driver script that handles **both transient diagnostics and steady-state averaging**. The user typically runs postprocessing after completing preprocessing and propagation.

Only the **stationary (steady-state) workflow** is documented here.

---

## Postprocess a Single Voltage

To postprocess a specific bias voltage, run with the --voltage flag:

```bash
python3 -m scripts.ss_postprocessing_pipeline --voltage "<voltage>eV"
```

In a single run, this command performs:

1. Transient ensemble diagnostics (if available)
2. Steady-state ensemble averaging (mandatory)

Transient diagnostics are used only to verify convergence to the stationary regime. Steady-state averages are the primary scientific output.

---

## Postprocess All Voltages

If no voltage is specified, the script iterates over all voltage directories found in the results tree:

python3 -m scripts.ss_postprocessing_pipeline

This runs postprocessing independently for each voltage.

---

## Collect Steady-State Results Across Voltages

After steady-state postprocessing has been completed for all voltages, the results can be combined into a single dataset using:

python3 -m scripts.ss_postprocessing_pipeline --collect

This searches all voltage directories, reads their steady-state output files, and concatenates them into a single voltage-resolved table.

---

## Inputs

The postprocessing stage assumes that propagation has already been completed and that the following data exist for each voltage:

- Transient trajectories (subset of trajectories retained during propagation)
- Steady-state sample trajectories
- Initial conditions used to generate steady-state samples

These are produced automatically during propagation and stored under the voltage-specific results directory.

---

## Transient Ensemble Diagnostics

Transient postprocessing is diagnostic and is used to verify that the vibrational degrees of freedom have reached a stationary regime before steady-state sampling.

Only a subset of trajectories is processed, as defined during propagation.

### Computed observables (as functions of time)

- Ensemble-averaged position \langle x(t) \rangle
- Ensemble-averaged momentum \langle p(t) \rangle
- Ensemble-averaged kinetic energy \langle KE(t)\rangle
- Ensemble-averaged potential energy \langle PE(t)\rangle
- Ensemble-averaged adiabatic current \langle  I(t) \rangle
- Correlation-function-related observables for the non-Markovian solver

### Outputs

Transient diagnostics are saved as PDF plots in:

results / ... / voltage_<voltage>eV / transient / plots /

Typical files include:

- energy_transient.pdf  
- physical_state_transient.pdf  
- current_ad_transient.pdf  
- corrfunc_transient.pdf  

If transient diagnostics fail for a given voltage, a warning is issued and steady-state analysis continues.

---

## Steady-State Ensemble Averaging

Steady-state postprocessing computes time-independent ensemble averages using statistically independent samples drawn from the stationary distribution.

These samples are generated during the propagation stage after the system has equilibrated.

### Observables

Steady-state averages include:

- Electric current
- Kinetic energy
- Potential energy
- Mean position \langle x \rangle
- Mean momentum \langle p \rangle
- Stochastic-force statistics (non-Markovian solver only)

### Output

For each voltage, steady-state results are written to a single CSV file:

results / ... / voltage_<voltage>eV / ss / ss_av.csv

This file contains one row corresponding to that voltage and all computed steady-state observables.

---

## Combined Voltage-Resolved Output

When running postprocessing with the collect option, all steady-state CSV files are concatenated into a single voltage-resolved dataset.

The combined file is saved in:

results / ... /

This dataset is typically used to construct current–voltage curves and other bias-dependent observables.

---

## Electronic-Force Postprocessing

Electronic-force postprocessing is independent of dynamical postprocessing and can be run separately.

It visualizes position-dependent electronic quantities produced during preprocessing.

### Usage

To postprocess electronic forces at a given voltage:

python3 -m scripts.run_postprocessing_cifs --voltage "<voltage>eV"

---

## Electronic-Force Outputs

Electronic-force plots are saved as PDF files in:

results / ... / voltage_<voltage>eV / el_forces /

These include:

- Position-dependent Markovian friction and noise
- Position-dependent adiabatic electronic forces
- Non-Markovian exponential weights and frequencies for friction
- Non-Markovian exponential weights and frequencies for correlation functions

These plots provide direct insight into how electronic environments couple to nuclear motion.

---

## Notes on Physical Interpretation

- Simulations are performed at finite bias voltage, where the fluctuation–dissipation theorem is not generally satisfied.
- Friction and noise are therefore always treated as independent inputs.
- At zero bias, the fluctuation–dissipation relation is recovered.
- The framework remains formally exact even out of equilibrium.
- Units (dimensionless or atomic) are handled consistently with preprocessing and propagation.

---

## Summary

Postprocessing converts raw trajectory data and electronic-force datasets into:

- Diagnostic transient ensemble plots
- Statistically converged steady-state observables
- Voltage-resolved transport data
- Interpretable electronic-force visualizations

It represents the final analysis layer of the simulation workflow.