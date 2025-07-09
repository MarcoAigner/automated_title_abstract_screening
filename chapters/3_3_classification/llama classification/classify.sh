#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45gb
#SBATCH --time=01:00:00 # ADJUST THE TIME LIMIT!
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

export OMP_NUM_THREADS=${SLURM_NTASKS}

# load the anaconda module
module load devel/cuda

# activate the project environment

python classify.py -d $1

# run with
# sbatch --job-name [dataset] classify.sh [dataset]