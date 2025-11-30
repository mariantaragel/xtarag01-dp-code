#!/bin/bash
#PBS -N train_change_it_3d_job
#PBS -q gpu
#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1:scratch_local=20gb:gpu_cap=sm_75
#PBS -l walltime=4:00:00

DATASET_NAME=removed-front-teeth-v11

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF
HOME_DIR=/storage/brno2/home/xtarag01
PROJECT_DIR=$HOME_DIR/xtarag01-dp-code
DATA_DIR=$HOME_DIR/datasets/$DATASET_NAME

SPLIT_FILE=$HOME_DIR/datasets/$DATASET_NAME/splits/processed-split.csv
PC_TOP_DIR=$HOME_DIR/datasets/$DATASET_NAME/point-clouds
VOCAB_FILE=$HOME_DIR/datasets/$DATASET_NAME/vocabulary/vocabulary.pkl
LOG_DIR=../../log_change_it_3d

LATENTS=$HOME_DIR/pretrained/$DATASET_NAME/pc_ae/latent_codes.pkl
PC_AE_FILE=$HOME_DIR/pretrained/$DATASET_NAME/pc_ae/best_model.pt
LISTENER_FILE=$HOME_DIR/pretrained/$DATASET_NAME/latent_listener/best_model.pkl

RANDOM_SEED=42
GPU_ID=0
LATENT_BACKBONE=pcae
SELF_CONTRAST=True
NET_ABLATION=coupled # decoupling_mag_direction
NUM_WORKERS=2

BATCH_SIZE=1024
LR=0.0005
IDENTITY_PENALTY=0
WEIGHT_DECAY=0

EPOCHS=300
TRAIN_PATIENCE=1000
LR_PATIENCE=10

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
    python3 -m scripts.train_change_it_3d \
        -shape_talk_file $SPLIT_FILE \
        -vocab_file $VOCAB_FILE \
        -latent_codes_file $LATENTS \
        -pretrained_listener_file $LISTENER_FILE \
        -pretrained_shape_generator $PC_AE_FILE \
        -data_dir $PC_TOP_DIR \
        --shape_generator_type $LATENT_BACKBONE \
        --log_dir $LOG_DIR \
        --random_seed $RANDOM_SEED \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --gpu $GPU_ID \
        --shape_editor_variant $NET_ABLATION \
        --identity_penalty $IDENTITY_PENALTY \
        --self_contrast $SELF_CONTRAST \
        --use_timestamp False \
        --init_lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --train_patience $TRAIN_PATIENCE \
        --lr_patience $LR_PATIENCE \
        --max_train_epochs $EPOCHS

echo "Cloning resluts ..."

mkdir -p $HOME_DIR/pretrained/$DATASET_NAME/change_it_3d/

cp $LOG_DIR/best_model.pt $HOME_DIR/pretrained/$DATASET_NAME/change_it_3d/
cp $LOG_DIR/config.json.txt $HOME_DIR/pretrained/$DATASET_NAME/change_it_3d/
cp $LOG_DIR/log.txt $HOME_DIR/pretrained/$DATASET_NAME/change_it_3d/
