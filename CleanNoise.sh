#!/bin/bash

#SBATCH --job-name=CleanNoiseCPU
#SBATCH --account=STARTUP-PCHOUHAN
#SBATCH --partition=bdwall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=CleanNoiseCPU.out
#SBATCH --error=CleanNoiseCPU.error
#SBATCH --mail-user=pc19d@my.fsu.edu
#SBATCH --time=10:00:00


# Setup Environment
cd /home/ac.pchouhan/model_LSTM
module load anaconda/4.4.0
source activate tf


#` train model 

#python convlstm2.py --epochs 1000 --outf "logs1" --timestep 10 --lr 0.05 --mode 2 --initial_epoch 600

python convlstm2.py --epochs 500 --outf "log3" --timestep 10 --lr 0.01 --mode 1 