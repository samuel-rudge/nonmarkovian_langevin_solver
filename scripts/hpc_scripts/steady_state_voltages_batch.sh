#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --mem=10GB
#SBATCH --array=0-2
#SBATCH --export=ALL
#SBATCH --output=logs/steady_state/slurm_%A_%a.out
#SBATCH --error=logs/steady_state/slurm_%A_%a.out
#SBATCH --job-name=ss

# ------------- LOAD ENVIRONMENT -------------
source ~/anaconda3/bin/activate
ulimit -s 400000

#---------------- Project root ---------------------
PROJECT_ROOT="<YOUR ROOT>"

cd $PROJECT_ROOT

# --------------- Read voltage
VOLTAGE_ARR=(0.275 0.30 0.325)
#0.00 0.025 0.05 0.075 0.10 0.125 0.15 0.175 0.20 0.225 0.25 0.275 0.30 0.325 0.35 0.375 0.40)
#0.45 0.50 0.55 0.60 0.65 0.70)
VOLTAGE=${VOLTAGE_ARR[$SLURM_ARRAY_TASK_ID]}


python3 -m scripts.ss_pipeline --voltage "$VOLTAGE"
