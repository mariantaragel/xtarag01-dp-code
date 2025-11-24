#!/bin/bash
#PBS -N train_pc_ae_job
#PBS -q gpu
#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1:scratch_local=20gb:gpu_cap=sm_75
#PBS -l walltime=8:00:00

DATASET_NAME=removed-front-teeth-v11

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF
HOME_DIR=/storage/brno2/home/xtarag01
PROJECT_DIR=$HOME_DIR/xtarag01-dp-code
DATA_DIR=$HOME_DIR/datasets/$DATASET_NAME

SPLIT_FILE=$HOME_DIR/datasets/$DATASET_NAME/splits/unary-split.csv
PC_TOP_DIR=$HOME_DIR/datasets/$DATASET_NAME/point-clouds
LOG_DIR=../../log_pc_ae

TIMESTAMP=False
RANDOM_SEED=42
SCALE=True
GPU_ID=0
NUM_WORKERS=2

ENCODER_NET=pointnet # dgcnn
DECODER_NET=mlp
BATCH_SIZE=32 # 64
LR=0.0005 # 0.00075

LATENT_BACKBONE=pc_ae_cls
LOSS=hybrid # chamfer, emd
N_PC_POINTS=4096
EPOCHS=100
TRAIN_PATIENCE=1000
LR_PATIENCE=10

ENCODER_LAYERS="32 64 64 128 256" # 64 128 128 256 512
DECODER_LAYERS="256 256 512" # 512 512 1024

CLASSIFIER_LAYERS="128"
ALFA=2.0

BETA=1.0

LOAD_PRETRAINED=False
PRETRAINED_FILE=$HOME_DIR/pretrained/$DATASET_NAME/pc_ae/best_model.pt

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
        --num_workers $NUM_WORKERS \
        --use_timestamp $TIMESTAMP \
        --encoder_conv_layers $ENCODER_LAYERS \
        --decoder_fc_neurons $DECODER_LAYERS \
        --init_lr $LR \
        --max_train_epochs $EPOCHS \
        --latent_backbone $LATENT_BACKBONE \
        --beta $BETA \
        --train_patience $TRAIN_PATIENCE \
        --lr_patience $LR_PATIENCE \
        --classifier_fc_neurons $CLASSIFIER_LAYERS \
        --alfa $ALFA \
        --load_pretrained_model $LOAD_PRETRAINED \
        --pretrained_model_file $PRETRAINED_FILE

echo "Cloning resluts..."

mkdir -p $HOME_DIR/pretrained/$DATASET_NAME/pc_ae/

cp $LOG_DIR/best_model.pt $HOME_DIR/pretrained/$DATASET_NAME/pc_ae/
cp $LOG_DIR/config.json.txt $HOME_DIR/pretrained/$DATASET_NAME/pc_ae/
cp $LOG_DIR/latent_codes.pkl $HOME_DIR/pretrained/$DATASET_NAME/pc_ae/
