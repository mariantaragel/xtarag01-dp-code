#!/bin/bash
#PBS -N pc_ae_train_test_job
#PBS -q gpu
#PBS -l select=1:ncpus=6:mem=16gb:ngpus=1:scratch_local=16gb
#PBS -l walltime=2:00:00

CONTAINER="/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF"
HOME_DIR="/storage/brno2/home/xtarag01"
PROJEKT_DIR="$HOME_DIR/xtarag01-dp-code"
DATA_DIR="$HOME_DIR/removed-front-teeth"

SPLIT_FILE=../../removed-front-teeth/splits/unary-split.csv
PC_TOP_DIR=../../removed-front-teeth/point-clouds
LOG_DIR=../../log

random_seed=42
gpu_id=0
num_workers=6
encoder_net=pointnet
decoder_net=mlp
n_pc_points=4096
batch_size=32

echo "Creating env..."

cd $SCRATCHDIR
cp -r $PROJEKT_DIR .
cp -r $DATA_DIR .

trap 'clean_scratch' TERM EXIT

export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH

cd xtarag01-dp-code/src

echo "Starting training..."

singularity exec --nv \
    -B $SCRATCHDIR:/scratch \
    $CONTAINER \
    python3 -m scripts.train_pc_ae \
        -log_dir $LOG_DIR \
        -data_dir $PC_TOP_DIR \
        -split_file $SPLIT_FILE \
        --encoder_net $encoder_net \
        --decoder_net $decoder_net \
        --batch_size $batch_size \
        --n_pc_points $n_pc_points \
        --random_seed $random_seed \
        --gpu_id $gpu_id \
        --num_workers $num_workers

echo "Cloning resluts ..."

cp -r $LOG_DIR "$HOME_DIR/vysledky_trenovania"
