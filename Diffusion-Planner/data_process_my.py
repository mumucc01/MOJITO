import os
import argparse
import json
import multiprocessing
from tqdm import tqdm
import numpy as np
from typing import List

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from diffusion_planner.data_process.utils import convert_to_model_inputs
from diffusion_planner.data_process.agent_process import agent_past_process, sampled_tracked_objects_to_array_list,sampled_static_objects_to_array_list,agent_future_process
from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario, CameraChannel, LidarChannel

from diffusion_planner.data_process.data_processor import DataProcessor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.common.geometry.compute import principal_value
from shapely.geometry import Point, LineString
from nuplan.common.actor_state.state_representation import Point2D
import os

def all_images_exist(image_path_list, root_dir):
    """检查所有 image_path_list 中的 jpg 是否存在"""
    for img_dict in image_path_list:
        for k, rel_path in img_dict.items():
            abs_path = os.path.join(root_dir, rel_path)
            if not os.path.exists(abs_path):
                return False
    return True

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


def process_scenario(scenario, num_past_poses, past_time_horizon, num_agents, num_static,
                    max_ped_bike, map_features, radius, max_elements, max_points,
                    num_future_poses, future_time_horizon, save_dir):
    """Process a single scenario in parallel."""
    try:
        map_name = scenario._map_name
        token = scenario.token
        map_api = scenario.map_api  

        '''
            current + past sensor_images
        '''   
        time_horizon = 2 
        iteration_index = 0
        num_samples = 4
        #past_image_path_list , _ = scenario.get_past_sensors(iteration_index, time_horizon , num_samples, [channel for channel in CameraChannel])
        #current_image_path_list , _ = scenario.get_sensors_at_iteration(iteration_index,[channel for channel in CameraChannel])
        #image_path_list = past_image_path_list + [current_image_path_list]
        #image_path_list = past_image_path_list + [current_image_path_list]
        #sensor_root = '/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-past-current-trainval/nuplan-v1.1/sensor_blobs'
        #if not all_images_exist(image_path_list, sensor_root):
        #    return None
        print(f"Get!")

        '''
            current sensor_images
        '''   
        #iteration_index = 0
        #current_image_path_list , _ = scenario.get_sensors_at_iteration(iteration_index,[channel for channel in CameraChannel])
        #image_path_list = [current_image_path_list]
       
        #sensor_root = "/lpai/volumes/base-3da-ali-sh/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/sensor_blobs"
        #if not all_images_exist(image_path_list, sensor_root):
        #    return None
        #print(f"Get!")

        #sensors = scenario.get_sensors_at_iteration(0, [channel for channel in CameraChannel])
        #images_list = []

        #for channel, img in sensors.images.items():
        #   print(channel)
        #    print(img.as_numpy.shape)
        #    images_list.append(img.as_numpy) 
        #stacked_images = np.stack(images_list, axis=0)
        
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, 
                                    ego_state.rear_axle.heading], dtype=np.float64)
        
        # Get past states
        ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(
            scenario, num_past_poses, past_time_horizon)

        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=0, time_horizon=past_time_horizon, num_samples=num_past_poses
            )
        ]
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        neighbor_agents_past, neighbor_agents_types = \
            sampled_tracked_objects_to_array_list(sampled_past_observations)

        static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

        # Process agents
        ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, 
                             num_agents, static_objects, static_objects_types, 
                             num_static, max_ped_bike, anchor_ego_state)

        # Process route and map
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))

        if route_roadblock_ids != ['']:
            route_roadblock_ids = route_roadblock_correction(
                ego_state, map_api, route_roadblock_ids
            )

        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, map_features, ego_coords, radius, traffic_light_data
        )

        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, 
                               traffic_light_data, speed_limit, lane_route, map_features, 
                               max_elements, max_points)

        # Get future states
        ego_agent_future = get_ego_future_array_from_scenario(
            scenario, ego_state, num_future_poses, future_time_horizon)

        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0, time_horizon=future_time_horizon, num_samples=num_future_poses
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
        neighbor_agents_future = agent_future_process(
            anchor_ego_state, future_tracked_objects_array_list, num_agents, neighbor_indices)

        # Calculate current state
        ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

        # Save data
        data = {
            "map_name": map_name, 
            "token": token, 
            "ego_current_state": ego_current_state, 
            "ego_agent_future": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past, 
            "neighbor_agents_future": neighbor_agents_future, 
            "static_objects": static_objects,
            "sensor_image_path":image_path_list
        }
        data.update(vector_map)

        output_path = f"{save_dir}/{data['map_name']}_{data['token']}.npz"
        np.savez(output_path, **data)
        return output_path
        
    except Exception as e:
        print(f"Error processing scenario {scenario.token}: {str(e)}")
        return None

class OptimizedDataProcessor(DataProcessor):
    def work(self, scenarios: List[NuPlanScenario]):
        """Process scenarios in parallel with progress tracking."""
        # Create a pool of workers
        
        num_processes = min(multiprocessing.cpu_count(), len(scenarios))
        
        # Prepare arguments for each scenario
        args = [(scenario, self.num_past_poses, self.past_time_horizon, self.num_agents, 
                self.num_static, self.max_ped_bike, self._map_features, self._radius, 
                self._max_elements, self._max_points, self.num_future_poses, 
                self.future_time_horizon, self._save_dir) for scenario in scenarios]
        #args = [single_scenario, self.num_past_poses, self.past_time_horizon, self.num_agents, 
        #        self.num_static, self.max_ped_bike, self._map_features, self._radius, 
        #        self._max_elements, self._max_points, self.num_future_poses, 
        #        self.future_time_horizon, self._save_dir]
        
        # Process scenarios in parallel with progress bar
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.starmap(process_scenario, args),
                total=len(scenarios),
                desc="Processing scenarios"
            ))
        
        # Return successful outputs
        return [r for r in results if r is not None]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized Data Processing')
    #parser.add_argument('--data_path', default="/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-past-current-trainval/nuplan-v1.1/trainval", type=str)
    parser.add_argument('--data_path', default="/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset_tem", type=str)
    parser.add_argument('--map_path', default="/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/maps", type=str)
    parser.add_argument('--save_path', default="/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-past-current-trainval/npz_data", type=str)
    parser.add_argument('--scenarios_per_type', type=int, default=2)
    parser.add_argument('--total_scenarios', type=int, default=10)
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
    sensor_root = '/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-img-lidar-current-100w/sensor_imgs'

    with open('/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-img-lidar-current-100w/jiaoshi_db.json', "r", encoding="utf-8") as file:
        log_names = json.load(file)
    '''
    parser = argparse.ArgumentParser(description='Optimized Data Processing')
    parser.add_argument('--data_path', default="/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/splits/mini", type=str)
    parser.add_argument('--map_path', default="/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/maps", type=str)
    parser.add_argument('--save_path', default="/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-current-20w/npz_data", type=str)
    parser.add_argument('--scenarios_per_type', type=int, default=20000)
    parser.add_argument('--total_scenarios', type=int, default=1000000)
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
    sensor_root = '/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/sensor_blobs'   #自动查找db文件

    with open('/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset/nuplan_mini_log_train.json', "r", encoding="utf-8") as file:
        log_names = json.load(file)  #读取log: db文件
    '''
    map_version = "nuplan-maps-v1.0"    
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, log_names=log_names))

    worker = SingleMachineParallelExecutor(use_process_pool=False)
    scenarios = builder.get_scenarios(scenario_filter, worker)  # Don't use worker here #P252 /nuplan-devkit-master/nuplan/planning/scenario_builder/nuplan_db/nuplan_scenario_builder.py
    print(f"Total number of scenarios: {len(scenarios)}")
    
    # Process data with optimized processor
    processor = OptimizedDataProcessor(args)
    processed_files = processor.work(scenarios)

    # Save the list to a JSON file
    with open('/lpai/volumes/base-3da-ali-sh/chengzhijing/nuplan-dataset-current-20w/nuplan_mini_npz_training.json', 'w') as json_file:
        json.dump(processed_files, json_file, indent=4)

    print(f"Successfully processed {len(processed_files)} scenarios")
    