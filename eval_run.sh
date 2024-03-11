#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --mem=100gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=/shares/rra_sarkar-2135-1003-00/faces/face_verification/slurm_outs/output.%j
#SBATCH --partition=rra
#SBATCH --qos=rra
#SBATCH --time=2-00:00:00


module load apps/anaconda
source /apps/anaconda3/5.3.1/etc/profile.d/conda.sh
conda activate torch_faces

srun python ./tester/run_evals.py
# srun python ./tester/calc_score.py
# srun python ./tester/extract_bts_chips.py --config briar_6
