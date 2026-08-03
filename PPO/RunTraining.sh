#!/bin/bash

#SBATCH --job-name="PPO"
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

nvidia-smi
source ./../.venv/bin/activate

python3 Training.py
