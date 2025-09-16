#!/bin/sh

python_script=/home/marian/DP/xtarag01-dp-code/src/scripts/train_pc_ae.py
split_file=/home/marian/DP/xtarag01-dp-code/data/removed-front-teeht/split.csv
pc_top_dir=/home/marian/DP/xtarag01-dp-code/data/removed-front-teeht/point-clouds
log_dir=/home/marian/DP/log-test-pc-ae
experiment_tag=$encoder_net-$decoder_net/points_$n_pc_points

random_seed=42
gpu_id=0
num_workers=6

encoder_net=pointnet
decoder_net=mlp

n_pc_points=4096
batch_size=32


python3 $python_script\
  -log_dir $log_dir\
  -data_dir $pc_top_dir\
  -split_file $split_file\
  --encoder_net $encoder_net\
  --decoder_net $decoder_net\
  --batch_size $batch_size\
  --n_pc_points $n_pc_points\
  --random_seed $random_seed\
  --gpu $gpu_id\
  --num_workers $num_workers\
  --experiment_tag "$experiment_tag"
