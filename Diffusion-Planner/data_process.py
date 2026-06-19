import os
import argparse
import json

from diffusion_planner.data_process.data_processor import DataProcessor

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):

    scenario_types = None

    scenario_tokens                      # List of scenario tokens to include
    log_names = log_names                # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type               # Number of scenarios per type
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None         # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = True              # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = False          # Whether to remove scenarios where the mission goal is invalid
    shuffle                              # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized Data Processing')
    #parser.add_argument('--data_path', default="/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-past-current-trainval/nuplan-v1.1/trainval", type=str)
    parser.add_argument('--data_path', default="/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-past-current-trainval/nuplan-v1.1/trainval", type=str)
    parser.add_argument('--map_path', default="/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/maps", type=str)
    parser.add_argument('--save_path', default="/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-past-current-trainval/npz_data", type=str)
    parser.add_argument('--scenarios_per_type', type=int, default=20)
    parser.add_argument('--total_scenarios', type=int, default=100)
    parser.add_argument('--shuffle_scenarios', type=bool, default=True)
    parser.add_argument('--agent_num', type=int, default=32)
    parser.add_argument('--static_objects_num', type=int, default=5)
    parser.add_argument('--lane_len', type=int, default=20)
    parser.add_argument('--lane_num', type=int, default=70)
    parser.add_argument('--route_len', type=int, default=20)
    parser.add_argument('--route_num', type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    db_files = None
    sensor_root = '/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-past-current-trainval/nuplan-v1.1/sensor_blobs'

    with open('/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-past-current-trainval/tem.json', "r", encoding="utf-8") as file:
        log_names = json.load(file)
        
    map_version = "nuplan-maps-v1.0"    
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, log_names=log_names))

    worker = SingleMachineParallelExecutor(use_process_pool=False)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    # process data
    del worker, builder, scenario_filter
    processor = DataProcessor(args)
    processor.work(scenarios)

    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]

    # Save the list to a JSON file
    with open('/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/try/npz_data/nuplan_mini_npz_training.json', 'w') as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"Saved {len(npz_files)} .npz file names")