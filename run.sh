#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=fin
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time 999:99:99


python main.py