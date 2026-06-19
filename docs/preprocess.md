# Preprocessing Pipeline (Electronic Force Dataset Construction)

This document describes the **preprocessing stage** of the simulation framework, which converts raw electronic structure data into a compact, structured representation suitable for both **Markovian and non-Markovian Langevin dynamics**.

The preprocessing pipeline is responsible for transforming **grid-resolved electronic response data** into smooth, interpolable force representations and (in the non-Markovian case) decomposed memory kernels.

---

# 1. Purpose of Preprocessing

The preprocessing stage has three main objectives:

1. **Data ingestion**
   - Traverse the electronic-force dataset defined on a discrete nuclear coordinate grid
   - Load all required electronic observables at each grid point

2. **Model reduction**
   - Convert raw time-dependent functions into compact representations
   - In the Markovian case: extract static friction and noise strength
   - In the non-Markovian case: decompose memory kernels into exponential sums

3. **Interpolation readiness**
   - Organize all quantities as smooth functions of nuclear coordinate x
   - Prepare data structures for runtime interpolation in the propagation stage

---

# 2. Input Data Structure

The preprocessing pipeline operates on datasets of the form:

data /
    data_identifier /
        gamma_<...> /
            delta_<...> /
                tier_<...> /
                    voltage_<V>eV /
                        x_<value> /


Each `x_<value>` directory contains:

## Required files (Markovian and non-Markovian shared)

- `adiabatic_force.dat`
- `friction.dat`
- `corrfunc.dat`

## Required files (non-Markovian only)

- `friction_integrand_heom.dat`
- `corrfunc_integrand_heom.dat`

---

# 3. Configuration Control

All preprocessing behavior is controlled via:

config/settings.yaml

Key control flag:

    simulation:
        markovian: true | false

- `true`  → Markovian friction and noise extraction
- `false` → non-Markovian kernel decomposition using HEOM integrands

Additional parameters define:
- voltage grid
- time discretization
- decomposition accuracy
- numerical tolerances

---

# 4. Preprocessing Workflow

The pipeline is executed via:

python3 -m scripts.run_preprocessing

or for a single voltage:

python3 -m scripts.run_preprocessing --voltage <value>

e.g. 
python3 -m scripts.run_preprocessing --voltage "0.00eV"
python3 -m scripts.run_preprocessing --voltage "0.40eV"


---

## 4.1 Directory traversal

The preprocessing system performs hierarchical traversal:

1. Iterate over voltage directories
2. Iterate over x-coordinate directories
3. Load electronic data at each grid point

The directory structure is defined in:

- `source/utils/file_walker.py`

---

## 4.2 Data loading stage

At each nuclear coordinate x:

The following objects are loaded:

- mean electronic force:
  - `F_ad(x)`

- friction:
  - Markovian: `gamma(x)`
  - non-Markovian: `gamma(x, t)`

- correlation function:
  - Markovian: `D(x)`
  - non-Markovian: `D(x, t)`

---

## 4.3 Markovian preprocessing path

When `markovian = true`, the pipeline:

1. Extracts static friction:
   \[
   \tilde{\gamma}(x, 0)
   \]

2. Extracts noise strength:
   \[
   \tilde{D}(x, 0)
   \]

3. Constructs continuous mappings:
   - \( F_{\mathrm{ad}}(x) \)
   - \( \tilde{\gamma}(x, 0) \)
   - \( \tilde{D}(x, 0) \)

4. Stores results in compact arrays for interpolation

---

## 4.4 Non-Markovian preprocessing path

When `markovian = false`, the pipeline performs a full kernel decomposition.

### Step 1: Kernel extraction

From HEOM integrands:

- friction kernel:
  \[
  \gamma(x, t)
  \]

- correlation function:
  \[
  D(x, t)
  \]

---

### Step 2: Exponential decomposition

Both kernels are approximated as sums of exponentials:

\[
\gamma(x, t) \approx \sum_{k} w_k(x)\,e^{-\nu_k(x)t}
\]

\[
D(x, t) \approx \sum_{k} a_k(x)\,e^{-\lambda_k(x)t}
\]

where:
- weights depend smoothly on x
- frequencies are extracted using spectral decomposition methods

---

### Step 3: Numerical methods

The decomposition uses:

- ESPRIT for friction kernel decomposition
- Lorentzian spectral fitting for correlation functions

Constraints:
- friction decomposition may produce complex exponentials
- correlation decomposition enforces real-valued weights for stochastic consistency

---

### Step 4: Output construction

The following arrays are constructed:

- friction kernel representation:
  - weights and frequencies as functions of x

- correlation kernel representation:
  - weights and frequencies as functions of x

---

## 4.5 Output files

Preprocessing generates:

results /
    data_identifier /
        gamma_<...> /
            delta_<...> /
                tier_<...> /
                    voltage_<V>eV /
                        x_<value> /
                            el_forces /
                                unprocessed_weights_frequencies_friction.npz
                                unprocessed_weights_frequencies_corrfunc.npz


These files contain:

- x-grid values
- decomposition weights
- exponential decay rates / frequencies
- kernel reconstruction data

They are used directly by the propagation engine.

---

# 5. File system abstraction

Dataset traversal and structure interpretation is handled by:

- `source/utils/file_walker.py`

Key responsibilities:

- Identify voltage directories
- Identify x-coordinate subdirectories
- Map floating-point coordinates to directory names
- Enforce consistent grid structure across voltages

---

# 6. Output structure (logical view)

After preprocessing, the system produces:

## Markovian case

- \( F_{\mathrm{ad}}(x) \)
- \( \tilde{\gamma}(x,0) \)
- \( \tilde{D}(x,0) \)

## Non-Markovian case

- \( F_{\mathrm{ad}}(x) \)
- \( \gamma(x,t) \rightarrow \{w_k(x), \nu_k(x)\} \)
- \( D(x,t) \rightarrow \{a_k(x), \lambda_k(x)\} \)

All stored in interpolable numerical form.

---

# 7. Design assumptions

This preprocessing stage assumes:

- Electronic data is precomputed and fixed on a discrete grid
- Nuclear coordinate dependence is smooth enough for interpolation
- Time-dependent kernels admit exponential decomposition
- Each voltage dataset is independent but structurally identical

---

# 8. Failure modes

Common issues include:

- missing x-grid points
- inconsistent time grids across x
- poorly conditioned kernel decomposition
- non-smooth weight functions across x
- inconsistent voltage directory structure

---

# 9. Role in full simulation pipeline

Preprocessing is the **interface layer between electronic structure and dynamics**:

electronic structure data -> preprocessing -> Langevin / GLE propagation


It does not perform time evolution itself.

It only constructs:

- friction models
- noise models
- interpolable force fields

---

# 10. Summary

The preprocessing pipeline transforms raw electronic response data into a compact representation of a **position-dependent quantum electronic bath**.

It enables:

- Markovian Langevin dynamics via static friction and noise
- Non-Markovian dynamics via exponential kernel decompositions
- Smooth interpolation over nuclear configuration space

This stage defines the entire data interface between electronic structure calculations and nuclear dynamics simulation.