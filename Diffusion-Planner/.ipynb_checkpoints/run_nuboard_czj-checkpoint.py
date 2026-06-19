# Useful imports
import os
from pathlib import Path
import tempfile
import hydra
import sys
import sys
sys.path.append('/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/pluto-main/src')
# conda activate /lpai/volumes/base-3da-ali-sh-mix/chengzhijing/conda_env/diffusion_planner_v10_mix
#RESULT_FOLDER = "/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/Diffusion-Planner/exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14/diffusion_planner_release/model_epoch_50_trainloss_0.0347_2025-08-19-00-58-16" # simulation result absolute path
# RESULT_FOLDER = "/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v2v3/Diffusion-Planner/exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14/diffusion_planner_release/model_epoch_20_trainloss_0.0052_2025-08-08-13-34-21"
RESULT_FOLDER = "/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/Diffusion-Planner/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14/diffusion_planner_release/model_epoch_290_trainloss_0.0057_2025-11-09-11-52-41"
env_variables = {
    "NUPLAN_DEVKIT_ROOT": "/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/nuplan-devkit-master",  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
    "NUPLAN_DATA_ROOT": "/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset", # nuplan dataset absolute path (e.g. "/data")
    "NUPLAN_MAPS_ROOT": "/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/maps", # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
    "NUPLAN_EXP_ROOT": "/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/Diffusion-Planner/exp", # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")
    "NUPLAN_SIMULATION_ALLOW_ANY_BUILDER":"1"
}

for k, v in env_variables.items():
    os.environ[k] = v

# Location of path with all nuBoard configs
current_file_dir = os.getcwd()
print(f"Sss{current_file_dir}")
CONFIG_PATH = os.path.relpath(os.path.join(current_file_dir, '../nuplan-devkit-master/nuplan/planning/script/config/nuboard'), os.getcwd())
#CONFIG_PATH = '../../nuplan-devkit-master/nuplan/planning/script/config/nuboard' # relative path to nuplan-devkit

CONFIG_NAME = 'default_nuboard'

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

ml_planner_simulation_folder = RESULT_FOLDER
ml_planner_simulation_folder = [dp for dp, _, fn in os.walk(ml_planner_simulation_folder) if True in ['.nuboard' in x for x in fn]]

#print("Sensor root exists:", os.path.exists(os.path.join(os.environ['NUPLAN_DATA_ROOT'], 'sensor_blobs')))

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    'scenario_builder=nuplan',  # set the database (same as simulation) used to fetch data for visualization
    f'simulation_path={ml_planner_simulation_folder}',  # nuboard file path(s), if left empty the user can open the file inside nuBoard
    'hydra.searchpath=[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]',
    'port_number=1121',

    #f'+scenario_builder.data_root={os.environ["NUPLAN_DATA_ROOT"]}',
    #f'+scenario_builder.map_root={os.environ["NUPLAN_MAPS_ROOT"]}',
    #f'+scenario_builder.sensor_root=/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionDriveData/openscene-v1.1/meta_datas/mini'
])
#print(cfg.scenario_builder)

from nuplan.planning.script.run_nuboard import main as main_nuboard

# Run nuBoard
main_nuboard(cfg)