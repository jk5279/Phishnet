#!/bin/bash
#SBATCH --account=def-cglee
#SBATCH --job-name=DL_pipeline_script
#SBATCH --output=DL_pipeline%j.out
#SBATCH --nodes=1                        # Request 1 full nodes
#SBATCH --gpus-per-node=4                # 4 GPUs per node (full node)
#SBATCH --time=04:00:00

module load StdEnv/2023
module load cuda/12.6
module load openmpi/4.1.5

source /scratch/kimjong/MIE1517-Fall2025-Group11/ENV/bin/activate

srun python DL_pipeline_script.py