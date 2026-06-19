#!/bin/bash
#/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/conda_env/diffusion_planner_v10_mix
###################################
# User Configuration Section
###################################
cd /mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_tem_czj/Diffusion-Planner
RUN_PYTHON_PATH="/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/conda_env/diffusion_planner_mix/bin/python"
TRAIN_SET_PATH="/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-img-lidar-current-100w/npz_data" # preprocess data using data_process.sh
TRAIN_SET_LIST_PATH="/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-img-lidar-current-100w/npz_100w.json"


export MASTER_PORT=29500
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'


MASTER_IP="10.80.12.197"
echo "主节点IP: $MASTER_IP"

###################################

###################################
echo "Start Single Node Multi-GPU Training"

$RUN_PYTHON_PATH -m torch.distributed.run \
  --nnodes=11 \
  --nproc_per_node=8 \
  --node_rank=2 \
  --master_addr="$MASTER_IP" \
  --master_port=$MASTER_PORT \
  train_predictor.py \
  --train_set $TRAIN_SET_PATH \
  --train_set_list $TRAIN_SET_LIST_PATH \
