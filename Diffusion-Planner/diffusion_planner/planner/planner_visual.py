import warnings
import torch
import numpy as np
from pathlib import Path
import time
import imageio
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from typing import Deque, Dict, List, Type
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
warnings.filterwarnings("ignore")
from nuplan.planning.simulation.planner.planner_report import PlannerReport
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
from PIL import Image
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config
from diffusion_planner.feature_builders.nuplan_scenario_render import NuplanScenarioRender
from diffusion_planner.scenario_manager.scenario_manager import ScenarioManager

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
    requires_scenario: bool = True
    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",

            eval_dt: float = 0.1,
            eval_num_frames: int = 80,
            render: bool = True,
            save_dir=None,
            scenario: AbstractScenario = None,
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

        self._planner_feature_builder = self._planner.get_list_of_required_feature()[0]

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        # Add visualization components
        self._eval_dt = eval_dt
        self._eval_num_frames = eval_num_frames
   
        self._scenario_manager: Optional[ScenarioManager] = None
        self._render = render
        self._imgs = []
        self._scenario = scenario
    
        save_dir='/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_tem_czj/Diffusion-Planner/video_try'
        if self._render:
            self._scene_render = NuplanScenarioRender()
            if save_dir is not None:
                self.video_dir = Path(save_dir)
            else:
                self.video_dir = Path(os.getcwd())
            self.video_dir.mkdir(exist_ok=True, parents=True)

    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
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
            self._planner.load_state_dict(model_state_dict,strict=False)
        else:
            print("load random model")
          
        if hasattr(self._planner, 'image_backbone'):
            image_backbone_ckpt = '/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/Diffusion-Planner/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS'
            print("Loading image_backbone weights from:", image_backbone_ckpt)
            backbone_state_dict = torch.load(image_backbone_ckpt, map_location=self._device)['state_dict']
            backbone_state_dict = {
                k.replace("agent._transfuser_model._backbone.", ""): v
                for k, v in backbone_state_dict.items()
            }
            try:      
                missing_keys, unexpected_keys = self._planner.image_backbone.load_state_dict(backbone_state_dict, strict=False)
                #print(f"Missing keys: {missing_keys}")
                #print(f"Unexpected keys: {unexpected_keys}")
            except RuntimeError as e:
                print(f"Failed to load image_backbone weights: {e}")
        else:
            print("No separate ckpt for image_backbone, or image_backbone_ckpt not set.")

        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization
        self._scenario_manager = ScenarioManager(
            map_api=self._map_api,
            ego_state=None,
            route_roadblocks_ids=self._route_roadblock_ids,
            radius=self._eval_dt * self._eval_num_frames * 60 / 4.0,
        )
        self._planner_feature_builder.scenario_manager = self._scenario_manager
        if self._render:
            self._scene_render.scenario_manager = self._scenario_manager

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        return model_inputs

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    

        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states
    
    def compute_planner_trajectory(self, current_input: PlannerInput,sensor_image) -> AbstractTrajectory:
        start_time = time.perf_counter()
        self._feature_building_runtimes.append(time.perf_counter() - start_time)
        start_time = time.perf_counter()
        ego_state = current_input.history.ego_states[-1]
        self._scenario_manager.update_ego_state(ego_state)
        self._scenario_manager.update_drivable_area_map()

        trajectory = self._run_planning_once(current_input,sensor_image)
        self._inference_runtimes.append(time.perf_counter() - start_time)
        return trajectory

    def _run_planning_once(self, current_input: PlannerInput, sensor_image):
        inputs = self.planner_input_to_model_inputs(current_input)
        inputs = self.observation_normalizer(inputs)        
        #start = time.time()

        sensor_image = sensor_image.unsqueeze(0) 
        sensor_image = normalization(sensor_image)

        _, outputs = self._planner(inputs,sensor_image)
        #end = time.time()
        #print(f"diffusion cost time: {end - start}")

        ego_state = current_input.history.ego_states[-1]

        output_trajectories = outputs['prediction'][:, 0, :, :].detach().cpu().numpy().astype(np.float64)  # [B, P, T, 4] -> [B, T, 4]
        candidate_trajectories = self.output_trajectories_process(output_trajectories, ego_state)   # [B, T, 4] -> [B, T + 1, 3]

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        if self._render:
            self._imgs.append(
                self._scene_render.render_from_simulation(
                    current_input=current_input,
                    initialization=self._initialization,
                    route_roadblock_ids=self._scenario_manager.get_route_roadblock_ids(),
                    scenario=self._scenario,
                    iteration=current_input.iteration.index,
                    planning_trajectory=self._global_to_local(trajectory, ego_state),
                    candidate_trajectories=self._global_to_local(
                        candidate_trajectories[:], ego_state
                    ),
                    candidate_index=0,
                    predictions=None,
                    return_img=True,
                )
            )

        return trajectory

    '''
    def compute_planner_trajectory(self, current_input: PlannerInput, sensor_image) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)
        
        #可能要做一个normalize
        inputs = self.observation_normalizer(inputs)  
      
        sensor_image = sensor_image.to(torch.float32) 
        sensor_image = sensor_image / 255.   
        sensor_image = sensor_image.unsqueeze(0) 
        _, outputs = self._planner(inputs, sensor_image)
        
        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

         
        return trajectory
    '''
    
    def output_trajectories_process(self, output_trajectories, ego_state):
        """
        output_trajectories: [N, T, 4] local x, y, cos, sin

        results:
        candidate_trajectories: [N, T, 3] global x y heading
        """
        heading = np.arctan2(output_trajectories[..., 3], output_trajectories[..., 2])[..., None]
        output_trajectories = np.concatenate([output_trajectories[..., :2], heading], axis=-1)

        # to global
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        output_trajectories[..., :2] = (
            np.matmul(output_trajectories[..., :2], rot_mat) + origin
        )
        output_trajectories[..., 2] += angle

        output_trajectories = np.concatenate(
            [output_trajectories[..., 0:1, :], output_trajectories],
            axis=-2,
        )

        return output_trajectories
    
    def _global_to_local(self, global_trajectory: np.ndarray, ego_state: EgoState):
        if isinstance(global_trajectory, InterpolatedTrajectory):
            states: List[EgoState] = global_trajectory.get_sampled_trajectory()
            global_trajectory = np.stack(
                [
                    np.array(
                        [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
                    )
                    for state in states
                ],
                axis=0,
            )

        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(global_trajectory[..., :2] - origin, rot_mat)
        heading = global_trajectory[..., 2] - angle

        return np.concatenate([position, heading[..., None]], axis=-1)
    


    def merge_images_and_save(self, a):
        token_dir = Path(a) / str(self._scenario.token)
        if not token_dir.exists():
            raise ValueError(f"{token_dir} 不存在")

        img_paths = sorted(token_dir.glob("*"), key=lambda x: int(x.stem))
        
        if len(img_paths) != len(self._imgs):
            raise ValueError(f"帧数不匹配: {len(img_paths)} vs {len(self._imgs)}")
        
        new_imgs = []
        for i, img_path in enumerate(img_paths):
            img = Image.open(img_path).convert("RGB")
            
            target_h = self._imgs[i].shape[0]
            w, h = img.size
            new_w = int(w * (target_h / h))
            img_resized = img.resize((new_w, target_h))
            
            img_resized = np.array(img_resized)
            
            merged = np.hstack([self._imgs[i], img_resized])
            new_imgs.append(merged)
        
        self.new_imgs = new_imgs
        
        output_path = self.video_dir / f"{self._scenario.log_name}_{self._scenario.token}.mp4"
        imageio.mimsave(output_path, self.new_imgs, fps=10)
        print(f"视频保存到 {output_path}")


    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        if self._render:
            #mini_scenario_path='/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_tem_czj/Diffusion-Planner/gt_scenario_visualization'
            #self.merge_images_and_save(mini_scenario_path)
            imageio.mimsave(
                self.video_dir
                / f"{self._scenario.log_name}_{self._scenario.token}.mp4",
                self._imgs,
                fps=10,
            )

            print("\n video saved to ", self.video_dir / "video.mp4\n")
           

        return report
    
    