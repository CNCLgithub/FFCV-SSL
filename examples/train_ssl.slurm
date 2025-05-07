#!/bin/bash

# Parameters
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4
#SBATCH --error=/gpfs/radev/project/yildirim/hy348/dev/FFCV-SSL/delete_me/%j_0_log.err
#SBATCH --gpus-per-node=2
#SBATCH --job-name=ffcv-ssl
#SBATCH --mail-type=ALL
#SBATCH --mem=50GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/radev/project/yildirim/hy348/dev/FFCV-SSL/delete_me/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --time=00-03

# command
module add miniconda
conda activate ffcv-ssl
python examples/train_ssl.py --config-file examples/rn50_configs/rn50_spin.yaml --logging.folder delete_me --logging.new_folder delete_me 0
