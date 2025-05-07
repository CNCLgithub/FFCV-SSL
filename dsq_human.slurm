#!/bin/bash
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=12
#SBATCH --error /gpfs/radev/project/yildirim/hy348/dev/FFCV-SSL/logs/%A_%2a.err
#SBATCH --gpus-per-node=2
#SBATCH --job-name=ffcv-ssl
#SBATCH --mail-type=ALL
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --open-mode=append
#SBATCH --partition=gpu
#SBATCH --time=02-00
#SBATCH --output /gpfs/radev/project/yildirim/hy348/dev/FFCV-SSL/logs/%A_%2a.out
#SBATCH --array 0-7

# DO NOT EDIT LINE BELOW
/gpfs/radev/apps/avx512/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/radev/project/yildirim/hy348/dev/FFCV-SSL/examples/joblist.txt --status-dir /gpfs/radev/project/yildirim/hy348/dev/FFCV-SSL/logs
