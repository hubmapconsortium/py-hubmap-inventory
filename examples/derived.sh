#!/bin/bash

#SBATCH -n 30
#SBATCH -p batch
#SBATCH --mem=128G

module load anaconda
. "/hive/users/hive/anaconda3/etc/profile.d/conda.sh"
conda activate /hive/users/hive/icaoberg/py-hubmapbags/env

python ./derived.py
