#!/bin/bash
#PBS -N train_latent_listener_job
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=8gb:ngpus=1:scratch_local=10gb
#PBS -l walltime=0:15:00

CONTAINER="/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF"
HOME_DIR="/storage/brno2/home/xtarag01"
PROJECT_DIR="$HOME_DIR/xtarag01-dp-code"
DATA_DIR="$HOME_DIR/removed-front-teeth-v2"

SPLIT_FILE=../../removed-front-teeth-v2/splits/removed-front-teeth-split-processed.csv
PC_TOP_DIR=../../removed-front-teeth-v2/point-clouds
VOCAB_FILE=../../removed-front-teeth-v2/vocabulary/vocabulary.pkl
LOG_DIR=../../log_listener
LATENTS=$HOME_DIR/pretrained/shape_latents/latent_codes.pkl

RANDOM_SEED=42
GPU_ID=0
BATCH_SIZE=64
NUM_WORKERS=1
MAX_EPOCHS=5

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
    python3 -m scripts.train_latent_listener \
        -latent_codes_file $LATENTS \
        -shape_talk_file $SPLIT_FILE \
        -vocab_file $VOCAB_FILE \
        --log_dir $LOG_DIR \
        --use_timestamp False \
        --random_seed $RANDOM_SEED \
        --gpu $GPU_ID \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --max_train_epochs $MAX_EPOCHS

echo "Cloning resluts ..."

cp -r $LOG_DIR "$HOME_DIR/results"
