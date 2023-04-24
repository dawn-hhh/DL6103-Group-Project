#!/bin/sh
#SBATCH --partition=SCSEGPU_M2
#SBATCH --qos=q_dmsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=25G
#SBATCH --job-name=tr
#SBATCH --output=train_cutmix_bs32.out
#SBATCH --error=train_cutmix_bs32.err


python train.py -net resnest50