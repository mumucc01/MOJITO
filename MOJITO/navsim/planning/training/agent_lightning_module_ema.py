import copy
import pytorch_lightning as pl

from torch import Tensor
from typing import Dict, Tuple
import torch
from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent, ema_decay: float = 0.9999):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        :param ema_decay: EMA decay coefficient, default 0.9999
        """
        super().__init__()
        self.agent = agent
        self.ema_decay = ema_decay

        self.ema_agent = copy.deepcopy(agent)
        for param in self.ema_agent.parameters():
            param.requires_grad_(False)

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str, use_ema: bool = False) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :param use_ema: whether to use EMA model for inference
        :return: scalar loss
        """
        features, targets = batch
        model = self.ema_agent if use_ema else self.agent
        prediction = model.forward(features, targets)
        loss_dict = model.compute_loss(features, targets, prediction)
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[0]))
        return loss_dict['loss']

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train", use_ema=False)

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples, using EMA model for inference.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val", use_ema=True)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Update EMA model parameters after each training batch."""
        decay = self.ema_decay
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_agent.parameters(),
                self.agent.parameters()
            ):
                ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Replace state_dict with EMA model parameters when saving checkpoint."""
        ema_state_dict = {}
        for k, v in self.ema_agent.state_dict().items():
            ema_state_dict[f"agent.{k}"] = v
        checkpoint['state_dict'] = ema_state_dict

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Restore EMA model from checkpoint state_dict."""
        ema_state = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('agent.'):
                ema_state[k[len('agent.'):]] = v
        missing, _ = self.ema_agent.load_state_dict(ema_state, strict=False)
        if missing:
            print(f"[EMA] on_load_checkpoint missing keys in ema_agent: {missing}")

    def on_after_backward(self):
        if not self.trainer.is_global_zero:
            return
