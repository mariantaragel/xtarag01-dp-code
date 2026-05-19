#!/bin/bash
#PBS -N create_dataset
#PBS -q default
#PBS -l select=1:ncpus=1:mem=32gb:ngpus=0:scratch_local=20gb
#PBS -l walltime=8:00:00

DATASET_NAME=removed-upper-teeth-v5

HOME_DIR=/storage/brno2/home/xtarag01
PROJECT_DIR=$HOME_DIR/xtarag01-dp-code
DATA_DIR=$HOME_DIR/data/Orthodontic_dental_dataset
SAVE_DIR=$HOME_DIR/datasets
SPLIT_FILE=$HOME_DIR/data/train-test-split.json

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF
export SINGULARITYENV_PYTHONPATH="$HOME_DIR/.local/lib/python3.12/site-packages"

trap 'clean_scratch' TERM EXIT

cd $SCRATCHDIR
cp -r $PROJECT_DIR .
cd xtarag01-dp-code/src

TEETH_TO_REMOVE="11 12 13 14 15 16 17 18 21 22 23 24 25 26 27 28"
NOTATION=1

singularity exec \
    -B $SCRATCHDIR:/scratch \
    $CONTAINER \
    python3 -m dataset.create_dataset \
        --data_dir $DATA_DIR \
        --dataset_name $DATASET_NAME \
        --save_dir $SAVE_DIR \
        --split_file $SPLIT_FILE \
        --teeth_to_remove $TEETH_TO_REMOVE \
        --notation $NOTATION

singularity exec \
    -B $SCRATCHDIR:/scratch \
    $CONTAINER \
    python3 -m dataset.create_unary_split \
        --dataset_name $DATASET_NAME \
        --save_dir $SAVE_DIR

singularity exec \
    -B $SCRATCHDIR:/scratch \
    $CONTAINER \
    python3 -m dataset.process_dataset \
        --dataset_name $DATASET_NAME \
        --save_dir $SAVE_DIR
