import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath
#from DiffusionDrive.navsim.agents.transfuser.transfuser_config_raster import TransfuserConfig_Raster

from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.model.module.mixer import MixerBlock
from diffusion_planner.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer
from navsim.agents.diffusiondrive.trimodal_fusion import Trimodal_Fusion
#from diffusion_planner.model.lidar_guidanced_noise import lidar_guidance_noise_sampling

import torchvision
from torchvision.transforms import v2
def make_transform_for_tensor():
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([normalize])


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()
        
        self.dit = DiT(
            sde=self._sde, 
            #route_encoder = RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth, 
            output_dim= config.output_dim, # x, y, cos, sin
            hidden_dim=config.hidden_dim, 
            heads=config.num_heads, 
            dropout=dpr,
            model_type=config.diffusion_model_type
        ).to("cuda")
        
        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        #self.raster_config = TransfuserConfig_Raster()
        #self._guidance_fn = config.guidance_fn
        self._guidance_fn = getattr(config, 'guidance_fn', None)
        #self.inference_noise_sampling = lidar_guidance_noise_sampling(self.raster_config)
        
      
        self._dbg_dit_out = None
        self._dbg_task_out = None
    @property
    def sde(self):
        return self._sde
    
    def forward(self, noise, image_feature, status_encoding, t):
    #def forward(self, noise, bev_feature, lidar_feature,status_encoding,diffusion_time):
        """
        Diffusion decoder process.

        Args:
            encoder_outputs: Dict
                {
                    ...
                    "encoding": agents, static objects and lanes context encoding
                    ...
                }
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,            
                    "neighbor_agent_past": past and current neighbor states,  

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
                    ...
                }

        """
      
        #noise:[B,8,3] bev_feature:[B,64,256]  lidar_feature:[B,1,256,256] status_encoding:[B,1,256]
        B = noise.shape[0]
        P = self._predicted_neighbor_num + 1
        if self.training:  #noise [B,8,32]
            noise = noise.reshape(noise.shape[0], 1, -1) 
          
            #sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1) # [B, 1 + predicted_neighbor_num, (1 + V_future) * 2]
          
            x0 = self.dit(
                        noise, 
                        t, 
                        image_feature, 
                        status_encoding
                    )
            
            return x0
        else:
            noise = noise.reshape(noise.shape[0], 1, -1).to("cuda") 
            #def initial_state_constraint(xt, t, step):
            #    xt = xt.reshape(B, P, -1, 3)
            #    xt[:, :, 0, :] = current_states
            #    return xt.reshape(B, P, -1)    y, attn_mask, ):  
            x  = dpm_sampler(  #/Diffusion-Planner/diffusion_planner/model/diffusion_utils/sampling.py  P18    
                        self.dit,
                        noise,
                        other_model_params={
                            "image_feature": image_feature,
                            "status_encoding": status_encoding,
                        },
                        dpm_solver_params={
                            "correcting_xt_fn": None,
                        },
                        model_wrapper_params={
                            "classifier_fn": self._guidance_fn,
                            "classifier_kwargs": {
                                "model": self.dit,
                                "model_condition": {
                                    "image_feature": image_feature,
                                    "status_encoding": status_encoding, 
                                },
                                "inputs": None,
                                "observation_normalizer": None,
                                "state_normalizer": None
                            },
                            "guidance_scale": 0.5,
                            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond"
                        },
                )
         
            return  x

        
class RouteEncoder(nn.Module):
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D
        '''
        # only x and x->x' vector, no boundary, no speed limit, no traffic light
        x = x[..., :4]

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)

        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, -1)


    



class DiT(nn.Module):
    def __init__(self, sde: SDE, depth, output_dim, hidden_dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, model_type="x_start"):
        super().__init__()
        
        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        #self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        #self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
        self.trimodal_fusion = Trimodal_Fusion()
    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, image_feature, status_encoding):
        """
        Forward pass of DiT.
        x: (B, P, output_dim)   -> Embedded out of DiT
        t: (B,)
        cross_c: (B, N, D)      -> Cross-Attention context
        """ #bev_feature:[B,64,256]
        B, P, _ = x.shape # P=1
        
        x = self.preproj(x) #[B,1,256]

        x_embedding = torch.cat([self.agent_embedding.weight[0][None, :], self.agent_embedding.weight[1][None, :].expand(P - 1, -1)], dim=0)  # (1agent,192)
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1) # (B,1, 192)
        x = x + x_embedding     
        
        y = status_encoding
        y = y + self.t_embedder(t)      #[6,256]
        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
       
        with torch.autocast(device_type="cuda", dtype=x.dtype):
            #image_feature = image_feature.repeat(1, 3, 1, 1)
            transform = make_transform_for_tensor() 
            image_feature = transform(image_feature)   
            #for layer_idx in range(11):
            for layer_idx in range(12):
                image_feature = self.trimodal_fusion.process_joint_attention( #[256,1024]
                    image_feature, #[B,3,H，W] [1,256,256]
                    layer_idx 
                )
                #for block in self.blocks: #x:[B,1,256] bev_feature:[B,107,256] y:[B,256]
                image_feature, x = self.blocks[layer_idx](x, y, attn_mask, image_feature)  
            #lidar_feature:[B,261,384] image_feature:[B,1029,384] x:[B,1,3072]
        
        x = self.final_layer(x, y) #x:[B,1,24]

        
        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return  x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
    
    

   