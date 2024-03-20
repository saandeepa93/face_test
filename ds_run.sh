#!/bin/bash
#SBATCH --job-name=webds_writer
#SBATCH --mem=20gb
#SBATCH --cpus-per-task=16
#SBATCH --output=/shares/rra_sarkar-2135-1003-00/faces/face_verification/slurm_outs/output.%j
#SBATCH --partition=rra_con2020
#SBATCH --qos=rradl
#SBATCH --time=3-00:00:00


module load apps/anaconda
source /apps/anaconda3/5.3.1/etc/profile.d/conda.sh
conda activate torch_faces

srun python ./trainer/filter_tars.py --config briar_8
# srun python ./trainer/extract_chips.py --config briar_2
# srun python ./process/extract_chips.py --config briar_3
# srun python ./trainer/read_chips.py
# srun python playground.py
