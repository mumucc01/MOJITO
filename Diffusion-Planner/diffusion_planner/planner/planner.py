
import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config

def identity(ego_state, predictions):
    return predictions


def normalization(tensor , device='cuda'):
    B, V, C, H, W = tensor.shape
    
    x = tensor.float().to(device)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,1,C,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,1,C,1,1)
    

    
    x = x / 255.0
    x = (x - mean) / std
    return x


class DiffusionPlanner(AbstractPlanner):
    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

       

        self._planner = Diffusion_Planner(config)

        self.data_processor = DataProcessor(config)
        
        self.observation_normalizer = config.observation_normalizer

    def name(self) -> str:
        """
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            state_dict:Dict = torch.load(self._ckpt_path, map_location=self._device)
            
            if self._ema_enabled:
                state_dict = state_dict['ema_state_dict']
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict['model']
            # use for ddp
            model_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
            missing , unexpected = self._planner.load_state_dict(model_state_dict,strict=False)
            print(f"missing key is {missing}")
            print(f"unexpected key is {unexpected}")
        else:
            print("load random model")
        
        '''
        #  image_backbone
        if hasattr(self._planner, 'image_backbone'):
            image_backbone_ckpt = '/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/Diffusion-Planner/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS'
            print("Loading image_backbone weights from:", image_backbone_ckpt)
            backbone_state_dict = torch.load(image_backbone_ckpt, map_location=self._device)['state_dict']
            try:      
                missing_keys, unexpected_keys = self._planner.image_backbone.load_state_dict(backbone_state_dict, strict=False)
                #print(f"Missing keys: {missing_keys}")
                #print(f"Unexpected keys: {unexpected_keys}")
            except RuntimeError as e:
                print(f"Failed to load image_backbone weights: {e}")
        else:
            print("No separate ckpt for image_backbone, or image_backbone_ckpt not set.")
        
        '''
        #  image_backbone
        if hasattr(self._planner, 'image_backbone'):
            image_backbone_ckpt = '/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/Diffusion-Planner/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS'
            print("Loading image_backbone weights from:", image_backbone_ckpt)
            backbone_state_dict = torch.load(image_backbone_ckpt, map_location=self._device)['state_dict']
            try:      
                missing_keys, unexpected_keys = self._planner.image_backbone.load_state_dict(backbone_state_dict, strict=False)
                #print(f"Missing keys: {missing_keys}")
                #print(f"Unexpected keys: {unexpected_keys}")
            except RuntimeError as e:
                print(f"Failed to load image_backbone weights: {e}")
        else:
            print("No separate ckpt for image_backbone, or image_backbone_ckpt not set.")
        
        '''

        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        return model_inputs

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    
        
        #predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        xy = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        
        T = xy.shape[0]

        dirs = np.diff(xy, axis=0)  # (T-1, 2)


        '''
        #eps = 1e-6
        #norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        # ，： ego x  [1, 0]
        #default_dir = np.array([[1.0, 0.0]])
        #dirs_safe = np.where(norms < eps, default_dir, dirs)

        #  yaw_rel()
        #headings_rel = np.arctan2(dirs_safe[:, 1], dirs_safe[:, 0])  # (T-1,)
        ######
        '''
        #eps = 1e-6
        #norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        # ，： ego x  [1, 0]
        #default_dir = np.array([[1.0, 0.0]])
        #dirs_safe = np.where(norms < eps, default_dir, dirs)

        #  yaw_rel()
        #headings_rel = np.arctan2(dirs_safe[:, 1], dirs_safe[:, 0])  # (T-1,)
        ######
        '''

        headings_rel = np.arctan2(dirs[:, 1], dirs[:, 0])  # (T-1,)
        headings_rel = np.concatenate([[0.0], headings_rel], axis=0)  # (T,)

        headings_rel = headings_rel[..., None]  # (T, 1)

        predictions = np.concatenate([xy, headings_rel], axis=-1).astype(np.float32)  # (T, 3)

       
        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states
    
    def compute_planner_trajectory(self, current_input: PlannerInput, sensor_image, lidar_points_processed, pc_points) -> AbstractTrajectory:
        """
        """
        inputs = self.planner_input_to_model_inputs(current_input)
        
        inputs = self.observation_normalizer(inputs)  


        _, outputs = self._planner(inputs, sensor_image, lidar_points_processed, pc_points)
        trajectory = self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        trajectory = InterpolatedTrajectory(
            trajectory = trajectory
        )

        return trajectory