#!/bin/bash -l

imodel=$1
id=$2
iml=$3
ins=$4

#$ -S /bin/bash


#$ -l h_rt=72:00:0
#$ -pe mpi 1
#$ -l mem=16G
#$ -N check
#$ -wd /home/ucacjm1/Scratch/


#source /home/ucacjm1/anaconda3/bin/activate /home/ucacjm1/anaconda3/envs/keras/

conda activate keras



python /home/ucacjm1/workspace/recsys2019/RSC2019/run.py --path /home/ucacjm1/workspace/recsys2019/RSC2019/ --model $1 --d $2 --ml $3 --ns $4 --small 0
#python /home/ucacjm1/workspace/recsys2019/RSC2019/test.py

