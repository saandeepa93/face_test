#!/bin/bash
#SBATCH --job-name=webds_writer
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=saandeepaath@usf.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task=16
#SBATCH --output=/shares/rra_sarkar-2135-1003-00/faces/face_verification/slurm_outs/output.%j
#SBATCH --partition=rra_con2020
#SBATCH --qos=rradl
#SBATCH --time=35-00:00:00


module load apps/anaconda
source /apps/anaconda3/5.3.1/etc/profile.d/conda.sh
conda activate torch_faces

srun python ./trainer/train_webds.py --config briar_4
# srun python ./trainer/extract_chips.py --config briar_2
# srun python ./process/extract_chips.py --config briar_3
# srun python ./trainer/read_chips.py
