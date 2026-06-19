import numpy as np
import numpy.typing as npt
from typing import List

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

def get_ego_past_array_from_scenario(scenario, num_past_poses, past_time_horizon):
    
    current_ego_state = scenario.initial_ego_state
    
    past_ego_states = scenario.get_ego_past_trajectory(
        iteration=0, num_samples=num_past_poses, time_horizon=past_time_horizon
    )

    sampled_past_ego_states = list(past_ego_states) + [current_ego_state]
    past_ego_states_array = sampled_past_ego_states_to_array(sampled_past_ego_states)

    
    past_time_stamps = list(
        scenario.get_past_timestamps(
            iteration=0, num_samples=num_past_poses, time_horizon=past_time_horizon
        )
    ) + [scenario.start_time]

    def sampled_past_timestamps_to_array(past_time_stamps: List[TimePoint]) -> npt.NDArray[np.float32]:
        flat = [t.time_us for t in past_time_stamps]
        return np.array(flat, dtype=np.int64)

    past_time_stamps_array = sampled_past_timestamps_to_array(past_time_stamps)

    return past_ego_states_array, past_time_stamps_array


def sampled_past_ego_states_to_array(past_ego_states: List[EgoState]) -> npt.NDArray[np.float32]:

    output = np.zeros((len(past_ego_states), 7), dtype=np.float64)
    for i in range(0, len(past_ego_states), 1):
        output[i, EgoInternalIndex.x()] = past_ego_states[i].rear_axle.x
        output[i, EgoInternalIndex.y()] = past_ego_states[i].rear_axle.y
        output[i, EgoInternalIndex.heading()] = past_ego_states[i].rear_axle.heading
        output[i, EgoInternalIndex.vx()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.x
        output[i, EgoInternalIndex.vy()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.y
        output[i, EgoInternalIndex.ax()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.x
        output[i, EgoInternalIndex.ay()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.y
        
    return output


def get_ego_future_array_from_scenario(scenario, current_ego_state, num_future_poses, future_time_horizon):

    future_trajectory_absolute_states = scenario.get_ego_future_trajectory(
        iteration=0, num_samples=num_future_poses, time_horizon=future_time_horizon
    )

    # Get all future poses of the ego relative to the ego coordinate system
    future_trajectory_relative_poses = convert_absolute_to_relative_poses(
        current_ego_state.rear_axle, [state.rear_axle for state in future_trajectory_absolute_states]
    )

    return future_trajectory_relative_poses


def calculate_additional_ego_states(ego_agent_past, time_stamp):
    # transform haeding to cos h, sin h and calculate the steering_angle and yaw_rate for current state

    current_state = ego_agent_past[-1]
    prev_state = ego_agent_past[-2]

    dt = (time_stamp[-1] - time_stamp[-2]) * 1e-6

    cur_velocity = current_state[3]
    angle_diff = current_state[2] - prev_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = angle_diff / dt

    if abs(cur_velocity) < 0.2:
        steering_angle = 0.0
        yaw_rate = 0.0  # if the car is almost stopped, the yaw rate is unreliable
    else:
        steering_angle = np.arctan(
            yaw_rate * get_pacifica_parameters().wheel_base / abs(cur_velocity)
        )
        steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
        yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

    current = np.zeros((ego_agent_past.shape[1] + 3), dtype=np.float32)
    current[:2] = current_state[:2]
    current[2] = np.cos(current_state[2])
    current[3] = np.sin(current_state[2])
    current[4:8] = current_state[3:7]
    current[8] = steering_angle
    current[9] = yaw_rate

    return current