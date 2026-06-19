#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=4GB
#SBATCH --array=0-2
#SBATCH --export=ALL
#SBATCH --output=logs/preprocessing/slurm_%A_%a.out
#SBATCH --error=logs/preprocessing/slurm_%A_%a.out
#SBATCH --job-name=preproc

# ------------- LOAD ENVIRONMENT -------------
source ~/anaconda3/bin/activate
ulimit -s 400000

#---------------- Project root ---------------------
PROJECT_ROOT="<YOUR ROOT>"
cd $PROJECT_ROOT

# --------------- Read voltage
VOLTAGE_ARR=(0.00 0.025 0.05 0.075 0.10 0.125 0.15 0.175 0.20 0.225 0.25 0.275 0.30 0.325 0.35 0.375 0.40)

VOLTAGE=${VOLTAGE_ARR[$SLURM_ARRAY_TASK_ID]}
python3 -m scripts.run_preprocessing --voltage "$VOLTAGE"
