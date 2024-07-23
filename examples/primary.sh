#!/bin/bash

#SBATCH -n 30
#SBATCH -p batch
#SBATCH --mem=128G
#SBATCH -o "primary-job-%j.out"
#SBATCH -e "primary-job-%j.out"

module load anaconda
. "/hive/users/hive/anaconda3/etc/profile.d/conda.sh"
conda activate /hive/users/hive/icaoberg/py-hubmapbags/env

export TOKEN=""
python ./primary.py
