export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
# Set environment variables
# export NUPLAN_DEVKIT_ROOT="REPLACE_WITH_NUPLAN_DEVIKIT_DIR"  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
# export NUPLAN_DATA_ROOT="REPLACE_WITH_DATA_DIR"  # nuplan dataset absolute path (e.g. "/data")
# export NUPLAN_MAPS_ROOT="REPLACE_WITH_MAPS_DIR" # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
# export NUPLAN_EXP_ROOT="REPLACE_WITH_EXP_DIR" # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

export NUPLAN_DEVKIT_ROOT="/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan-devkit-master"  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
export NUPLAN_DATA_ROOT="/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset"  # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_MAPS_ROOT="/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/nuplan-maps-v1.0/nuplan-maps-v1.0/maps" # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
export NUPLAN_EXP_ROOT="/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/Diffusion-Planner/exp" # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")


# Dataset split to use
# Options: 
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
# SPLIT="val14-collision"  # e.g., "val14"
SPLIT="val14"
# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="closed_loop_nonreactive_agents"  # e.g., "closed_loop_nonreactive_agents"
###################################

BRANCH_NAME=collision
ARGS_FILE=/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/Diffusion-Planner/checkpoints/args.json
CKPT_FILE=/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/Diffusion-Planner/checkpoints/model.pth
SCENARIO_BUILDER="nuplan_mini"

echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner_guidance

sudo -E /lpai/volumes/base-3da-ali-sh-mix/zhangxc/miniconda3/envs/diffusion_planner/bin/python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=128 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"