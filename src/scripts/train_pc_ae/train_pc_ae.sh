#!/bin/sh

split_file=../../removed-front-teeth/splits/unary-split.csv
pc_top_dir=../../removed-front-teeth/point-clouds
log_dir=../../log-test-pc-ae

random_seed=42
gpu_id=0
num_workers=6

encoder_net=pointnet
decoder_net=mlp

n_pc_points=4096
batch_size=32

cd ../..

python3 -m scripts.train_pc_ae \
  -log_dir $log_dir\
  -data_dir $pc_top_dir\
  -split_file $split_file\
  --encoder_net $encoder_net\
  --decoder_net $decoder_net\
  --batch_size $batch_size\
  --n_pc_points $n_pc_points\
  --random_seed $random_seed\
  --gpu_id $gpu_id\
  --num_workers $num_workers
