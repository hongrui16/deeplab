#!/bin/bash
#SBATCH -J db
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH -t 600000000:00
#SBATCH -o db.out

python train_main.py 