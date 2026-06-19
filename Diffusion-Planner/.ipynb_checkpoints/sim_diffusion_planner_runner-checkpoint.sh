export CUDA_VISIBLE_DEVICES=2
#export CUDA_VISIBLE_DEVICES=7
export HYDRA_FULL_ERROR=1
# conda activate /lpai/volumes/base-3da-ali-sh-mix/chengzhijing/conda_env/diffusion_planner_v10_mix
cd /lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10
###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT="/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/nuplan-devkit-master"  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")

#export NUPLAN_DATA_ROOT="/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/"  # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_DATA_ROOT="/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-past-current-trainval/nuplan-v1.1/trainval"
export NUPLAN_MAPS_ROOT="/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/maps" # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
export NUPLAN_EXP_ROOT= "/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/Diffusion-Planner/exp" # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

# Dataset split to use
# Options: 
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="val14"  # e.g., "val14"

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
#CHALLENGE="closed_loop_nonreactive_agents"  # e.g., "closed_loop_nonreactive_agents"
###################################
CHALLENGE="closed_loop_nonreactive_agents" 

BRANCH_NAME=diffusion_planner_release
ARGS_FILE=/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/Diffusion-Planner/training_log/v33/2025-11-08-14:25:04/args.json
# CKPT_FILE=/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_baseline/Diffusion-Planner/training_log/diffusion-planner-training/2025-07-26-01:15:16/model_epoch_100_trainloss_0.0271.pth
CKPT_FILE=/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/Diffusion-Planner/training_log/v33/2025-11-08-14:25:04/model_epoch_180_trainloss_0.0122.pth
if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan_mini"   #SCENARIO_BUILDER="nuplan_mini"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner

#export PYTHONPATH=/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_baseline/pluto-main/src:$PYTHONPATH
export PYTHONPATH="/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/Diffusion-Planner:$PYTHONPATH"



#/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/conda_env/diffusion_planner/bin/python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/conda_env/diffusion_planner_v10_mix/bin/python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=4 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.5 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
