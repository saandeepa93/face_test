#!/bin/bash
#SBATCH --job-name=train_base
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=saandeepaath@usf.edu
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120gb
#SBATCH --cpus-per-task=8
#SBATCH --output=/shares/rra_sarkar-2135-1003-00/faces/face_verification/slurm_outs/output.%j
#SBATCH --partition=rra
#SBATCH --qos=rra
#SBATCH --time=7-00:00:00

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=40

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR 


module load apps/anaconda
source /apps/anaconda3/5.3.1/etc/profile.d/conda.sh
conda activate torch_faces
CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=INFO srun python ./trainer/train_webds.py --config briar_7
