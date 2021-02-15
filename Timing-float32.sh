#!/bin/bash
#
#SBATCH --job-name=pylops_cupy_timing_f32
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --error=pylops_cupy_timing_f32.%J.err 
#SBATCH --output=pylops_cupy_timing_f32.%J.out
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --constraint=[v100]

## Move to directory:
cd /ibex/scratch/ravasim/PyLops_GPU_EAGEMilan

## Setup environment
module load cuda/10.2.89/gcc-7.5.0-jr6kobf
echo 'Loaded cuda:' $(which nvcc) $(which nvcc)
echo $CUDA_HOME

source activate pylops_cupy_cusignal

## Check specs
nvidia-smi
lscpu

## Run the application:
srun python Timing-float32.py
