# Propagation Module (Markovian and Non-Markovian Langevin Dynamics)

This document describes the **trajectory propagation layer** of the electronic friction framework.

It assumes that:

- electronic-force datasets have been precomputed via the preprocessing pipeline
- all forces and kernels are provided as interpolated functions of nuclear coordinate
- dynamical variables are rescaled internally according to the selected unit system

The propagation layer generates nuclear trajectories under either:

- Markovian electronic friction Langevin dynamics
- non-Markovian generalized Langevin dynamics with auxiliary-variable embedding

The propagation module implements a unified framework for nuclear dynamics in the presence of an electronic environment.

All dynamics are driven by:

- adiabatic mean forces
- electronic dissipation (friction)
- electronic fluctuations (noise)

The key feature of this framework is that:

Friction and noise are treated independently and explicitly.

At equilibrium (zero bias), they satisfy fluctuation-dissipation relations.

Out of equilibrium (finite bias), they are fully independent and the theory remains valid.

# 1. USAGE

## 1.1 Run all voltages

python3 -m scripts.ss_pipeline

This will:

- iterate over all voltage directories defined in config/settings.yaml
- run steady-state propagation for each voltage
- generate ensemble-averaged steady-state observables

## 1.2 Run a single voltage

python3 -m scripts.ss_pipeline --voltage "<value>eV"

e.g.

python3 -m scripts.ss_pipeline --voltage "0.0eV"

This will:

- only process the specified voltage directory
- overwrite or update steady-state outputs for that voltage

IMPORTANT: See docs/hpc.md for information on how to optimize these calculations on a High-Performance Computing environment. 

---

# 2. Input configuration (config/settings.yaml)

Propagation is controlled by:

simulation:
    markovian: true | false
    traj_type: transient | stationary

and general simulation parameters:

- number of trajectories
- timestep dt
- number of steps
- number of cores
- random seed

---

## 2.1 Unit system

The system supports:

units_type: dimensionless / atomic

All dynamical variables are explicitly rescaled before propagation.

The integrator always operates in a consistent internal representation.

---

# 3. Electronic force interface

All electronic effects are provided through interpolated functions constructed during preprocessing.

For a given nuclear coordinate x:

## Markovian case

- \( F_{\mathrm{ad}}(x) \)
- \( \tilde{\gamma}(x,0) \)
- \( \tilde{D}(x,0) \)

## Non-Markovian case

- \( F_{\mathrm{ad}}(x) \)
- \( \gamma(x,t) \rightarrow \{w_k(x), \nu_k(x)\} \)
- \( D(x,t) \rightarrow \{a_k(x), \lambda_k(x)\} \)


These represent exponential decompositions of memory/noise kernels.

---

# 4. Markovian propagation

Activated when:

simulation.markovian = true

Performed via the ABOBA 

The nuclear dynamics follow:

M \ddot{x} = - dU_vib/dx + F_ad(x) - \tilde{\gamma}(x) \dot{x}(t) / M + f(t)

and is solved via the method outlined in 

https://pubs.aip.org/aip/jcp/article/161/6/064106/3307575/Nonadiabatic-dynamics-of-molecules-interacting

# 5. Non-Markovian propagation

Activated when:

simulation.markovian = true

Performed via the ABOBA 

The nuclear dynamics follow:

M \ddot{x} = - dU_vib/dx + F_ad(x) - \int^t_0 dt' \gamma(x,t) \dot{x}(t - t') / M + f(t)

w/ \langle f(t) f(t') \rangle = D(x(t),t)

and is solved via the method outlined in 

https://pubs.acs.org/doi/10.1021/acs.nanolett.6c01839

# 6. Steady-state and transient protocols

The system supports two propagation modes:

---

## 6.1 Transient mode

simulation.traj_type = transient


- full trajectory propagation
- full time series storage
- used for dynamical analysis

Output:

- full trajectory propagation
- full time series storage
- used for dynamical analysis

results/.../transient/traj_#.npz

---

## 6.2 Steady-state mode

simulation.traj_type = stationary


This is a two-stage procedure executed in a single run.

---

### Stage 1: equilibration

- propagate until ss_time
- retain a subset of trajectories
- store diagnostic transient trajectories

Stored in:

results/.../ss/ss_samples as ss_sample_#.npz

and 

results/.../ss/initial_conditions as ss_ic_#.npz


---

### Stage 2: steady-state sampling

After equilibration:

- final states of all trajectories are saved as initial conditions
- system is re-advanced
- trajectories are sampled n_sample times with spacing dt_ind

---

### Output storage

#### sampled steady-state trajectories

results/.../ss/ss_samples/ss_sample_traj#.npz


---

# 7. Parallel execution

- trajectory-level parallelism via joblib
- controlled by n_cores
- reproducibility enforced via base_seed

---

# 8. Key physical structure

This propagation framework consistently enforces:

- explicit electronic dissipation
- explicit electronic fluctuations
- correct equilibrium limit (FDT at zero bias)
- controlled non-equilibrium generalization at finite bias

---

# 9. Summary

The propagation module provides a unified ABOBA-based integrator for:

- Markovian Langevin dynamics
- non-Markovian memory-resolved dynamics
- steady-state and transient sampling protocols
- equilibrium and non-equilibrium electronic environments

All electronic effects enter through precomputed datasets and are never evaluated on-the-fly.