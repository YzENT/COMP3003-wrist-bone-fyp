#!/bin/bash
#SBATCH -c2 --mem=16g
#SBATCH -G 1

export nnUNet_compile=false
export nnUNet_n_proc_DA=0
export nnUNet_pin_memory=0

nnUNetv2_train 067 3d_fullres all