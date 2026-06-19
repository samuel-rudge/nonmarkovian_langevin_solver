#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --mem=10GB
#SBATCH --export=ALL
#SBATCH --output=logs/steady_state/slurm_%j.out
#SBATCH --error=logs/steady_state/slurm_%j.out
#SBATCH --job-name=ss

# ------------- LOAD ENVIRONMENT -------------
source ~/anaconda3/bin/activate
ulimit -s 400000

#---------------- Project root ---------------------
PROJECT_ROOT="<YOUR ROOT>"
cd $PROJECT_ROOT

# --------------- Read voltage

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 -m scripts.ss_pipeline --voltage "0.325"
