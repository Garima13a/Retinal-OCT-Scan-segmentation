#!/bin/bash
#PBS -q gpu 
#PBS -l gpus=1
#PBS -l pvmem=128gb
#PBS -l walltime=10:00:00
#PBS -l nodes=1
#PBS -l procs=8
# conda activate fastai
cd ~/garima_seg/seg
export CUDA_VISIBLE_DEVICES="0,1"
mkdir runs/new_run3
python main_test.py -lr 0.001 -ne 50 -n test2 -s 500 -fe deeplabv3_resnet50 -b 8 > runs/test2/log 
