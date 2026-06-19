import sys

from navsim.mojito_paths import DIFFUSION_PLANNER_ROOT

diffusion_planner_path = str(DIFFUSION_PLANNER_ROOT)
if diffusion_planner_path not in sys.path:
    sys.path.insert(0, diffusion_planner_path)

from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer

class DP_Config:
    def __init__(self):
        self.future_len = 80
        self.time_len = 21
        self.agent_state_dim = 11
        self.agent_num = 32
        self.static_objects_state_dim = 10
        self.static_objects_num = 5
        self.lane_len = 20
        self.output_dim = 384*8
        #self.output_dim = 32*40
        self.lane_state_dim = 12
        self.lane_num = 70
        self.route_len = 20
        self.route_state_dim = 12
        self.route_num = 25
        self.query_len = 107
        self.encoder_drop_path_rate = 0.1
        self.decoder_drop_path_rate = 0.1
        self.encoder_depth = 3
        #self.decoder_depth = 12
        self.decoder_depth = 12
        
        self.num_heads = 8
        #self.hidden_dim = 256
        self.hidden_dim = 8*384
        self.diffusion_model_type = "x_start"
        self.predicted_neighbor_num = 0
        self.normalization_file_path = str(DIFFUSION_PLANNER_ROOT / "normalization.json")
        self.state_normalizer = StateNormalizer.from_json(self)
        self.observation_normalizer = ObservationNormalizer.from_json(self)
        self.tf_d_model = 256

