#!/bin/bash
#PBS -N train_latent_listener_job
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=16gb:ngpus=1:scratch_local=20gb:gpu_cap=sm_75
#PBS -l walltime=2:00:00

DATASET_NAME=removed-upper-teeth-v5

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF
HOME_DIR=/storage/brno2/home/xtarag01
PROJECT_DIR=$HOME_DIR/xtarag01-dp-code
DATA_DIR=$HOME_DIR/datasets/$DATASET_NAME

SPLIT_FILE=$HOME_DIR/datasets/$DATASET_NAME/splits/processed-split.csv
PC_TOP_DIR=$HOME_DIR/datasets/$DATASET_NAME/point-clouds
VOCAB_FILE=$HOME_DIR/datasets/$DATASET_NAME/vocabulary/vocabulary.pkl
LOG_DIR=../../log_listener
LATENTS=$HOME_DIR/pretrained/$DATASET_NAME/pc_ae/latent_codes.pkl

RANDOM_SEED=42
GPU_ID=0
BATCH_SIZE=2048
LR=0.0005
NUM_WORKERS=1
LISTENING_MODEL=ablation_model_one # transformer
# LISTENING_MODEL=ablation_model_two # lstm

export WANDB_API_KEY="d82cb78d19b6bb6e39d3f99f150c6bec08610567"
export SINGULARITYENV_PYTHONPATH="$HOME_DIR/.local/lib/python3.12/site-packages"

echo "Creating env..."

cd $SCRATCHDIR
cp -r $PROJECT_DIR .
cp -r $DATA_DIR .
cd xtarag01-dp-code/src

trap 'clean_scratch' TERM EXIT

echo "Starting training..."

singularity exec --nv \
    -B $SCRATCHDIR:/scratch \
    $CONTAINER \
    python3 -m scripts.train_latent_listener \
        -latent_codes_file $LATENTS \
        -shape_talk_file $SPLIT_FILE \
        -vocab_file $VOCAB_FILE \
        -data_dir $PC_TOP_DIR \
        --log_dir $LOG_DIR \
        --use_timestamp False \
        --random_seed $RANDOM_SEED \
        --gpu $GPU_ID \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --listening_model $LISTENING_MODEL \
        --init_lr $LR

echo "Cloning resluts ..."

mkdir -p $HOME_DIR/pretrained/$DATASET_NAME/latent_listener/

cp $LOG_DIR/analysis_of_trained_listener.pkl $HOME_DIR/pretrained/$DATASET_NAME/latent_listener/
cp $LOG_DIR/best_model.pkl $HOME_DIR/pretrained/$DATASET_NAME/latent_listener/
cp $LOG_DIR/best_model.pt $HOME_DIR/pretrained/$DATASET_NAME/latent_listener/
cp $LOG_DIR/config.json.txt $HOME_DIR/pretrained/$DATASET_NAME/latent_listener/
cp $LOG_DIR/log.txt $HOME_DIR/pretrained/$DATASET_NAME/latent_listener/
