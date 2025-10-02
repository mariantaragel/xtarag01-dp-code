#!/bin/bash
#PBS -N train_pc_ae_job
#PBS -q gpu
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1:scratch_local=20gb
#PBS -l walltime=1:00:00

CONTAINER="/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF"
HOME_DIR="/storage/brno2/home/xtarag01"
PROJECT_DIR="$HOME_DIR/xtarag01-dp-code"
DATA_DIR="$HOME_DIR/removed-front-teeth-v3"

SPLIT_FILE=../../removed-front-teeth-v3/splits/unary-split.csv
PC_TOP_DIR=../../removed-front-teeth-v3/point-clouds
LOG_DIR=../../log_pc_ae

ENCODER_NET=pointnet
DECODER_NET=mlp
LOSS=chamfer
BATCH_SIZE=32
N_PC_POINTS=4096
RANDOM_SEED=42
SCALE=True
GPU_ID=0
NUM_WORKERS=4

export WANDB_API_KEY="d82cb78d19b6bb6e39d3f99f150c6bec08610567"

echo "Creating env..."

cd $SCRATCHDIR
cp -r $PROJECT_DIR .
cp -r $DATA_DIR .

trap 'clean_scratch' TERM EXIT

export SINGULARITYENV_PYTHONPATH="$HOME_DIR/.local/lib/python3.12/site-packages"

cd xtarag01-dp-code/src

echo "Starting training..."

singularity exec --nv \
    -B $SCRATCHDIR:/scratch \
    $CONTAINER \
    python3 -m scripts.train_pc_ae \
        -log_dir $LOG_DIR \
        -data_dir $PC_TOP_DIR \
        -split_file $SPLIT_FILE \
        --encoder_net $ENCODER_NET \
        --decoder_net $DECODER_NET \
        --batch_size $BATCH_SIZE \
        --n_pc_points $N_PC_POINTS \
        --random_seed $RANDOM_SEED \
        --scale_in_u_sphere $SCALE \
        --loss_function $LOSS \
        --gpu_id $GPU_ID \
        --num_workers $NUM_WORKERS

echo "Cloning resluts ..."

cp -r $LOG_DIR "$HOME_DIR/results"
