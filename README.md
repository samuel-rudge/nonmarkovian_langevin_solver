# Electronic Friction and Non-Markovian Langevin Dynamics (EFLD) Framework

This repository implements a Python **electronic friction and Langevin dynamics framework** for nonadiabatic nuclear motion in molecular systems interacting with metallic environments. Electronic degrees of freedom are treated quantum mechanically and encoded through precomputed response functions obtained from HEOM-based calculations, including mean forces, friction kernels, and force–force correlation functions.

Nuclear (vibrational) degrees of freedom are propagated using either Markovian Langevin dynamics or non-Markovian generalized Langevin equations. In both cases, the electronic subsystem is integrated out and enters the nuclear equations of motion through position-dependent friction and stochastic noise terms that are reconstructed from the underlying quantum electronic data.

Relevant literature is found in - 

https://pubs.aip.org/aip/jcp/article/161/6/064106/3307575/Nonadiabatic-dynamics-of-molecules-interacting

https://pubs.acs.org/doi/10.1021/acs.nanolett.6c01839

https://pubs.aip.org/aip/jcp/article/161/6/064106/3307575/Nonadiabatic-dynamics-of-molecules-interacting

https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.115416

The code is structured into three conceptual stages:

1. **Electronic-force dataset definition and preprocessing** (docs/preprocess.md)
2. **Nuclear / vibrational propagation** (docs/propagation.md)
3. **Postprocessing and analysis** (docs/postprocess.md)

The current structure of the input data is outlined in docs/ELECTRONIC_FORCE_DATASET_SPEC.md.

---

## 1. Conceptual Overview

The simulation framework **does not compute electronic forces on-the-fly**. Instead, it:

- Loads **precomputed electronic response data** defined on a discrete grid of nuclear coordinates
- Interprets this data as a **position-dependent electronic bath**
- Constructs interpolated force fields and (optionally) non-Markovian memory kernels
- Supplies these objects to Langevin or generalized Langevin solvers

Two regimes are supported:

- **Markovian**: instantaneous friction and white noise
- **Non-Markovian**: time-dependent memory kernels and colored noise

The choice is controlled via `config/settings.yaml`. 

---

## 2. Dataset Identification and Hierarchy

Each dataset corresponds to a fixed electronic environment for a given molecular system, bias voltage, and electronic parameter regime. See docs/electronic_force_dataset_spec.md for detailed information.

## 3. Prerequisites and Architecture

The code is written and run entirely in Python. An Anaconda installation is recommended, with the relevant packages needed to build the conda environment given in BUILD.md

## 3. Running the code

The various scripts to run the three pipelines (preprocessing, propagation, and postprocessing) are found in scripts/ and run simply from the top-level of the working directory by 

python3 -m scripts.script_name --flag <value>

The flags generally refer to specific voltages, with details on the three pipelines found in their relevant documentation.

