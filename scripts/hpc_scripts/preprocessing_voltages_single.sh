#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=4GB
#SBATCH --export=ALL
#SBATCH --output=logs/preprocessing/slurm_%j.out
#SBATCH --error=logs/preprocessing/slurm_%j.out
#SBATCH --job-name=preproc

# ------------- LOAD ENVIRONMENT -------------
source ~/anaconda3/bin/activate
ulimit -s 400000

#---------------- Project root ---------------------
PROJECT_ROOT="<YOUR ROOT>"
cd $PROJECT_ROOT

# --------------- Read voltage

python3 -m scripts.run_preprocessing --voltage "0.30"
