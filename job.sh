#!/bin/bash
#SBATCH --job-name=test-std-node
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=1gb
#SBATCH --time=02:00:00
#SBATCH --account=yuekai1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=smaity@umich.edu
#SBATCH --partition=standard
python3 adversarial_sample.py