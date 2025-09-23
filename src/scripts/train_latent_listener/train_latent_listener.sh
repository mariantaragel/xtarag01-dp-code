#!/bin/bash
#PBS -N train_latent_listener_job
#PBS -q gpu
#PBS -l select=1:ncpus=4:mem=8gb:ngpus=1:scratch_local=10gb
#PBS -l walltime=0:30:00

CONTAINER="/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF"
HOME_DIR="/storage/brno2/home/xtarag01"
PROJECT_DIR="$HOME_DIR/xtarag01-dp-code"
DATA_DIR="$HOME_DIR/removed-front-teeth-v2"

SPLIT_FILE=../../removed-front-teeth-v2/splits/removed-front-teeth-split.csv
PC_TOP_DIR=../../removed-front-teeth-v2/point-clouds
SPLIT_FILE=../../removed-front-teeth-v2/splits/unary-split.csv
VOCAB_FILE=../../removed-front-teeth-v2/vocabulary/vocabulary.pkl
LOG_DIR=../../log

RANDOM_SEED=42
GPU_ID=0

## 1. PC-AE, trained with pointclouds scaled to be aligned with rendering images
latents=$PC_TOP_DIR/pretrained/shape_latents/pcae_latent_codes.pkl
log_dir=$LOG_DIR/latent_pcae_based

if false
then
  python $script_file\
   -latent_codes_file $latents\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu $gpu_id
fi

## 2. SGF based (gradient) latents
latents=$top_data_dir/pretrained/shape_latents/sgf_latent_codes.pkl
log_dir=$top_log_dir/latent_sgf_based

if false
then
  python $script_file\
   -latent_codes_file $latents\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu $gpu_id
fi

## 3. ImNet based (implicit) latents
latents=$top_data_dir/pretrained/shape_latents/imnet_latent_codes.pkl
log_dir=$top_log_dir/latent_imnet_based_scaled

if false
then
  python $script_file\
   -latent_codes_file $latents\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu $gpu_id
fi


## 4. ResNet34/101 based (image) latents
# we increase the weight-decay for these, since these are 512D (or 2048D) instead of 256D
resnet=101
latents=$top_data_dir/pretrained/shape_latents/resnet"$resnet"_latent_codes.pkl
log_dir=$top_log_dir/latent_resnet"$resnet"_based

if false
then
  python $script_file\
   -latent_codes_file $latents\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu $gpu_id\
   --weight_decay 0.005
fi


## 5. OpenAI CLIP
latents=$top_data_dir/pretrained/shape_latents/openai_clip-vit-large-patch14_latent_codes.pkl
log_dir=$top_log_dir/latent_openai_clip-vit-large-patch14_based

if true
then
  python $script_file\
   -latent_codes_file $latents\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu $gpu_id\
   --weight_decay 0.003
fi


## 6. Open-CLIP
latents=$top_data_dir/pretrained/shape_latents/laion_CLIP-ViT-H-14-laion2B-s32B-b79K_latent_codes.pkl
log_dir=$top_log_dir/latent_laion_CLIP-ViT-H-14-laion2B-s32B-b79K_based


if false
then
  python $script_file\
   -latent_codes_file $latents\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu $gpu_id\
   --weight_decay 0.003
fi
