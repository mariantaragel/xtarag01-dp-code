#!/bin/bash
#PBS -N evaluate_change_it_3d_job
#PBS -q gpu
#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1:scratch_local=20gb:gpu_cap=sm_75
#PBS -l walltime=2:00:00

DATASET_NAME=removed-front-teeth-v9

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF
HOME_DIR=/storage/brno2/home/xtarag01
PROJECT_DIR=$HOME_DIR/xtarag01-dp-code
DATA_DIR=$HOME_DIR/datasets/$DATASET_NAME

SPLIT_FILE=$HOME_DIR/datasets/$DATASET_NAME/splits/processed-split.csv
PC_TOP_DIR=$HOME_DIR/datasets/$DATASET_NAME/point-clouds
VOCAB_FILE=$HOME_DIR/datasets/$DATASET_NAME/vocabulary/vocabulary.pkl
LOG_DIR=../../log_change_it_3d

CHANGEIT3D_MODEL=$HOME_DIR/pretrained/$DATASET_NAME/change_it_3d/best_model.pt
LATENTS=$HOME_DIR/pretrained/$DATASET_NAME/pc_ae/latent_codes.pkl
PC_AE_FILE=$HOME_DIR/pretrained/$DATASET_NAME/pc_ae/best_model.pt

GPU_ID=0
N_PC_POINTS=4096
SEED=42
BATCH=96
NUM_WORKERS=2

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
    python3 -m scripts.evaluate_change_it_3d \
        -pretrained_changeit3d $CHANGEIT3D_MODEL \
        -shape_talk_file $SPLIT_FILE \
        -latent_codes_file $LATENTS \
        -vocab_file $VOCAB_FILE \
        -top_pc_dir $PC_TOP_DIR \
        --pretrained_shape_generator $PC_AE_FILE \
        --log_dir $LOG_DIR \
        --gpu_id $GPU_ID \
        --n_sample_points $N_PC_POINTS \
        --batch_size $BATCH \
        --random_seed $SEED \
        --num_workers $NUM_WORKERS \
        --evaluate_retrieval_version True


echo "Cloning resluts ..."

mkdir -p $HOME_DIR/results/$DATASET_NAME/

cp $LOG_DIR/evaluation_metric_results.pkl $HOME_DIR/results/$DATASET_NAME/
cp $LOG_DIR/evaluation_metric_results_for_retrieval.pkl $HOME_DIR/results/$DATASET_NAME/
