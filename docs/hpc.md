# Running on HPC Systems

This document provides guidance for running the code on **high-performance computing (HPC)** systems.  
The focus is on **voltage-parallel execution**, which is the recommended strategy for large parameter sweeps.

This file is complementary to:

- `preprocess.md`
- `propagate.md`
- `postprocess.md`
- `electronic_force_dataset_spec.md`

and assumes that the reader understands the overall workflow:
preprocessing → propagation → postprocessing.

---

## General Parallelization Strategy

The codebase is designed around the idea that **different voltages are independent simulations**.

As a result, the most robust and scalable strategy on HPC systems is:

- Assign **one voltage per job**
- Allocate **one node (or one job slot)** per voltage
- Let each job internally parallelize over trajectories using `joblib`

This avoids unnecessary synchronization, minimizes filesystem contention, and maps naturally onto job arrays.

---

## Why Parallelize Over Voltage?

- Preprocessing, propagation, and postprocessing are all voltage-local
- No communication is required between voltages
- Failures at one voltage do not affect others
- Results can be combined afterward using the postprocessing `--collect` option

This strategy works equally well for:

- Markovian simulations
- Non-Markovian simulations
- Transient-only runs
- Steady-state transport calculations

---

## Typical Workflow on HPC

A typical production workflow looks like:

1. **Preprocessing**
   - Run once (or once per voltage if desired)
   - Produces electronic-force datasets in `results/.../el_forces/`

2. **Propagation**
   - Run independently for each voltage
   - Produces trajectory data in `results/.../transient/` and `results/.../ss/`

3. **Postprocessing**
   - Run independently for each voltage
   - Optionally followed by a global collection step

Each stage can be submitted as a separate job array.

---

## SLURM Job Arrays: Concept

In SLURM, job arrays allow you to submit many nearly identical jobs that differ only by an index.

Each array task:
- Reads a voltage from a predefined list
- Runs preprocessing, propagation, or postprocessing for that voltage
- Writes output to a voltage-specific directory

This is the recommended approach for voltage sweeps. Example scripts are shown in scripts / hpc_scripts.

---

## Example: SLURM Job Array for Steady-State Postprocessing

Below is an **example SLURM submission script** that postprocesses steady-state data, assigning one voltage per array job.

Each array element runs completely independently.

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=10GB
#SBATCH --array=0-2
#SBATCH --export=ALL
#SBATCH --output=logs/steady_state/slurm_%A_%a.out
#SBATCH --error=logs/steady_state/slurm_%A_%a.out
#SBATCH --job-name=postprc

# ------------- LOAD ENVIRONMENT -------------
source ~/anaconda3/bin/activate
ulimit -s 400000

#---------------- Project root ---------------------
PROJECT_ROOT="<YOUR ROOT>"
cd $PROJECT_ROOT

# --------------- Voltage list --------------------
VOLTAGE_ARR=(0.00 0.025 0.05 0.075 0.10 0.125 0.15 0.175 0.20 0.225 0.25 0.275 0.30 0.325 0.35 0.375 0.40)

VOLTAGE=${VOLTAGE_ARR[$SLURM_ARRAY_TASK_ID]}

python3 -m scripts.ss_postprocessing_pipeline --voltage "$VOLTAGE"
```

## Notes on the Example

- `--array=0-2` controls how many voltages are processed  
- Increase this range to match the length of `VOLTAGE_ARR`

Each task:
- Uses its own voltage  
- Writes to its own results directory  
- Logs are separated by job ID and array index  
- Memory and walltime should be adjusted based on system size and observable selection  

---

## Propagation Jobs on HPC

For propagation jobs, the same array-based strategy applies.

### Important considerations

- Use `simulation.n_cores` in `settings.yaml` to control trajectory-level parallelism  
- Match `n_cores` to `cpus-per-task`  
- Avoid oversubscription (do not request more CPUs than used)

### Example mapping

- SLURM: `--cpus-per-task=20`  
- settings.yaml: `simulation.n_cores: 20`

---

## Preprocessing on HPC

Preprocessing can be:

- Run once serially (recommended for small systems)
- Parallelized over voltages using the same job array strategy

Preprocessing is typically much cheaper than propagation.

---

## Filesystem Considerations

- Each voltage writes to its own directory  
- No file locking or shared writes occur  
- Safe for parallel execution on shared filesystems  

However:

- Large trajectory outputs can generate many files  
- Consider compressing or archiving old runs  
- Ensure sufficient inode availability  

---

## After All Jobs Finish

Once all voltage jobs have completed:

Combine steady-state results with:

```bash
python3 -m scripts.ss_postprocessing_pipeline --collect
```

