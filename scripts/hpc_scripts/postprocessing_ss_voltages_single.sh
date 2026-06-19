#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=10GB
#SBATCH --export=ALL
#SBATCH --output=logs/steady_state/slurm_%j.out
#SBATCH --error=logs/steady_state/slurm_%j.out
#SBATCH --job-name=postprc

# ------------- LOAD ENVIRONMENT -------------
source ~/anaconda3/bin/activate
ulimit -s 400000

#---------------- Project root ---------------------
PROJECT_ROOT="<YOUR ROOT>"
cd $PROJECT_ROOT

# --------------- Read voltage

python3 -m scripts.ss_postprocessing_pipeline --voltage "0.35"
