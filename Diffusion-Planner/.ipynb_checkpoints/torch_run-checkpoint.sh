export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_tem_czj/Diffusion-Planner
###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/conda_env/diffusion_planner_mix/bin/python" # python path (e.g., "/home/xxx/anaconda3/envs/diffusion_planner/bin/python")

# Set training data path
TRAIN_SET_PATH="/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-img-lidar-current-100w/npz_data" # preprocess data using data_process.sh
TRAIN_SET_LIST_PATH="/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-img-lidar-current-100w/npz_100w.json"
###################################

sudo -E $RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \
