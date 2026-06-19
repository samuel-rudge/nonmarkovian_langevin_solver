# Electronic Force Dataset Specification (EFD Spec)

This document defines the required structure, naming conventions, and file formats for **electronic-force datasets** used in Markovian and non-Markovian Langevin / electronic-friction-based nuclear dynamics simulations.

The simulation framework **does not compute electronic forces on-the-fly**. Instead, it loads **precomputed electronic response data** defined on a discrete grid of nuclear configurations and constructs interpolated force fields and memory kernels for dynamical propagation.

---

## 1. Overview

Each dataset represents a fixed electronic environment for a given molecular system, voltage bias, and electronic parameter regime.

A dataset is identified by the directory hierarchy:

data_identifier /
    voltage_* /
        x_*

where:

- data_identifier: physical system label (arbitrary but unique)
- voltage_*: external bias condition
- x_*: nuclear coordinate grid point

---

# 2. Directory Structure

## 2.1 Dataset root

data / 
    data_identifier / 
        gamma_<molecule_lead_coupling>meV_temp_<temperature>K / 
            <Delta>eV / 
                <electronic_tier>

Example:

dataset_root = data / 
    2l1m_negative_friction_w_time_dependence / 
        gamma_100meV_temp_300K / 
            delta_0.2eV / 
                tier_2 / 

This folder contains all electronic-force data for a single physical system, potentially for several voltages. The user is free to the data_identifier and system specifications that go  into the dataset root (e.g. gamma, electronic tier), but it must remain consistent with the code. These can be changed in source/utils/config.py .

---

## 2.2 Voltage directories

dataset_root / voltage_<V>eV /

Example:

voltage_0.0eV /
voltage_0.05eV /

Each directory corresponds to a fixed external voltage bias.

Voltage values are defined in config/settings.yaml under:

simulation -> voltage

---

## 2.3 Nuclear coordinate grid

Inside each voltage directory:

x_<value>> /

Examples:

x_.0 /
x_.1 /
x_-.2 /
x_10.2 /

Each folder corresponds to a discrete nuclear configuration point.

### Naming rules

- Positive values:
  x_0.1, x_1.0, x_10.2

- Negative values:
  x_-0.1, x_-1.0, x_-10.2

- Values in the interval (-1, 1):
  0.2 maps to x_.2
  -0.2 maps to x_-.2

---

# 3. Required files per x-directory

Each x_* folder must contain the following files:

---

## 3.1 Adiabatic force

adiabatic_force.dat

Represents the mean electronic force:

F_ad(x)

Format:
- scalar or vector per line
- must be consistent across the full dataset

---

## 3.2 Electronic friction

friction.dat
friction_integrand_heom.dat

Represents the Markovian and non-Markovian electronic friction tensor, respectively:

\tilde{gamma}(x,0) and \gamma(x,t)

Format (friction_integrand_heom.dat):
- first column: time
- second column: friction value(s)

Format (friction.dat):
- single x-value line, first columns
- friction value remaining columns
---

## 3.3 Force-force correlation function

corrfunc.dat
corrfunc_integrand_heom.dat

Represents the Markovian and non-Markovian electronic force autocorrelation function:

\tilde{D}(x,0),
D(t, x) = < delta F(t) delta F(0) >

Used for:
- Markovian and non-Markovian memory effects

Format (corrfunc_integrand_heom.dat):
- first column: time
- second column: correlation value(s)

Format (corrfunc.dat):
- single x-value line, first columns
- corrfunc value remaining columns

---

# 4. Physical interpretation

Each x_* directory encodes a locally frozen electronic subsystem conditioned on nuclear coordinate x.

Each grid point defines:

- mean electronic force
- dissipative friction
- fluctuation statistics

Together, these define a position-dependent electronic bath model.

---

# 5. Dataset assembly procedure

At runtime, the simulation performs:

1. Traverse voltage directories
2. Traverse x-coordinate directories
3. Load:
   - adiabatic_force.dat
   - friction.dat
   - friction_integrand_heom.dat
   - corrfunc.dat
   - corrfunc_integrand_heom.dat

4. Construct discrete mappings:
   - F_ad(x)
   - \tilde{\gamma}(x,0), \gamma(x,t)
   - \tilde{D}(x,0), D(x,t)

5. Interpolate these quantities onto continuous nuclear coordinate space

---

# 6. Markovian Langevin equation

In the Markovian limit, nuclear dynamics are given by:

M \ddot{x}(t) = - dU_vib / dx + F_ad(x) - \tilde{gamma}(x,0) x_dot(t) + xi 

where:

- gamma(x): position-dependent friction
- \langle xi(t) \xi(t') \rangle_ens = \tilde{D}(x,0): stochastic noise term

---

# 7. Non-Markovian Langevin equation

In the non-Markovian limit, nuclear dynamics are given by:

M \ddot{x}(t) = - dU_vib / dx + F_ad(x) - \int_0^t dt' \gamma(x,t) x_dot(t - t') + xi(t)

where:

- \gamma(x,t): position- and time-dependent friction
- \langle xi(t) \xi(t') \rangle_ens = D(x,t): stochastic noise term

---

# 8. Data consistency requirements

## 8.1 Grid consistency
All voltage directories must share identical x-grid structure.

## 8.2 File completeness
Each x_* directory must contain all required files.

## 8.3 Dimensional consistency
- friction tensors must have identical shape across grid
- correlation functions must share compatible time grids

## 8.4 Units
- energy: eV
- voltage: V
- coordinates: dimensionless or atomic units
- temperature: K

---

# 8. Scope and assumptions

This dataset format is agnostic to:

- electronic structure method used to generate forces
- whether data comes from NEGF, HEOM, TDDFT, or other approaches
- Markovian or non-Markovian interpretation of dynamics

It only defines the interface between precomputed electronic data and nuclear dynamics.

---

# 9. Failure modes

Common issues include:

- missing x_* directories
- incomplete force files
- inconsistent grid spacing
- mismatched friction tensor shapes
- incompatible correlation time grids

---

# 10. Summary

This dataset defines a discretized electronic environment over nuclear configuration space.

It enables Langevin propagation with:

- position-dependent mean forces
- position-dependent friction
- position-dependent fluctuations

It forms the foundational input layer for electronic friction based molecular dynamics.