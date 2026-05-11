#!/bin/bash
#SBATCH -c10 --mem=32g
#SBATCH -G 2

export nnUNet_compile=false
export nnUNet_n_proc_DA=0
export nnUNet_pin_memory=0

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 067 3d_fullres 0 &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 067 3d_fullres 1 &
wait

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 067 3d_fullres 2 &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 067 3d_fullres 3 &
wait

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 067 3d_fullres 4