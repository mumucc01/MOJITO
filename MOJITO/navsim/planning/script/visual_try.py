from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import torchvision.utils as vutils
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from navsim.planning.training.dataset import Dataset
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.agents.diffusiondrive.transfuser_callback import TransfuserCallback
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def dict_to_device(dict: Dict[str, torch.Tensor], device: Union[torch.device, str]) -> Dict[str, torch.Tensor]:
    """
    Helper function to move tensors from dictionary to device.
    :param dict: dictionary of names and tensors
    :param device: torch device to move tensors to
    :return: dictionary with tensors on specified device
    """
    for key in dict.keys():
        dict[key] = dict[key].to(device)
    return dict

def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    token = np.random.choice(scene_loader.tokens)
    scene = scene_loader.get_scene_from_token(token)
    from navsim.visualization.plots import plot_bev_frame

    frame_idx = scene.scene_metadata.num_history_frames - 1 # current frame
    #fig, ax = plot_bev_frame(scene, frame_idx)
    #plt.savefig('/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/my_plot.png') 
    from navsim.visualization.plots import plot_bev_with_agent
    from navsim.agents.constant_velocity_agent import ConstantVelocityAgent

    agent = ConstantVelocityAgent()
    fig, ax = plot_bev_with_agent(scene, agent)
    plt.savefig('/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/my_agent.png') 

    return None
    # tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    # pdm_results: List[Dict[str, Any]] = []


    # with open('/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/dac0.json', 'r') as f:
    #     ep0_tokens = json.load(f)


    # for idx, (token) in enumerate(tokens_to_evaluate):
    #     # if token in ep0_tokens:
    #     # else: 
    #     #      continue
    #     # logger.info(
    #     #     f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
    #     # )
    #     score_row: Dict[str, Any] = {"token": token, "valid": True}
    #     try:
    #         metric_cache_path = metric_cache_loader.metric_cache_paths[token]
    #         with lzma.open(metric_cache_path, "rb") as f:
    #             metric_cache: MetricCache = pickle.load(f)

    #         agent_input = scene_loader.get_agent_input_from_token(token)
    #         if agent.requires_scene:
    #             scene = scene_loader.get_scene_from_token(token)
    #             trajectory = agent.compute_trajectory(agent_input, scene)
    #             #trajectory = None
    #             predictions, trajectory = agent.compute_trajectory(agent_input)

    #         dataset = Dataset(
    #             scene_loader=scene_loader,
    #             feature_builders=agent.get_feature_builders(),
    #             target_builders=agent.get_target_builders(),
    #             cache_path=cfg.metric_cache_path,
    #             force_cache_computation=False,
    #             visualize = True
    #         )
           
    #         features, targets  = dataset.visual_dataset(token)
    #         features, targets, predictions = (
    #             dict_to_device(features, "cpu"),
    #             dict_to_device(targets, "cpu"),
    #             dict_to_device(predictions, "cpu"),
    #         )
    #         img = TransfuserCallback(TransfuserConfig()).pdm_score_visualize_model(features, targets, predictions)
    #         vutils.save_image(img.float() / 255.0 if img.dtype != torch.float32 else img, os.path.join('/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/visualization_v69', f"{token}.png"))
            
    #         pdm_result = pdm_score(
    #             metric_cache=metric_cache,
    #             model_trajectory=trajectory,
    #             future_sampling=simulator.proposal_sampling,
    #             simulator=simulator,
    #             scorer=scorer,
    #         )
    #         score_row.update(asdict(pdm_result))
    #     except Exception as e:
    #         logger.warning(f"----------- Agent failed for token {token}:")
    #         traceback.print_exc()
    #         score_row["valid"] = False

    #     pdm_results.append(score_row)
    # return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
   
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    data_points = data_points[:1]
    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    main()
