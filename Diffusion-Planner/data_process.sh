###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/splits/mini" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_baseline/nuplan-dataset/npz_data" # preprocess training data
###################################

/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/conda_env/diffusion_planner_new/bin/python data_process_my.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \

