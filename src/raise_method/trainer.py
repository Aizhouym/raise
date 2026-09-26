"""Small veRL trainer extension for original-context RAISE updates."""
from __future__ import annotations

import copy
from functools import partial
from pathlib import Path

import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.utils.padding import left_right_2_no_padding

from .loss import raise_actor_loss


class RaiseTrainer(RayPPOTrainer):
    """Use guided behavior contexts while optimizing the original prompt."""

    def init_workers(self):
        config = self.config
        actor = config.actor_rollout_ref.actor
        if actor.ppo_epochs != 1 or actor.ppo_mini_batch_size != config.data.train_batch_size:
            raise ValueError("RAISE requires one optimizer batch and ppo_epochs=1")
        super().init_workers()
        actor_config = omega_conf_to_dataclass(actor)
        self.actor_rollout_wg.set_loss_fn(partial(raise_actor_loss, config=actor_config))
        manager = self.async_rollout_manager
        manager.tokenizer = self.tokenizer
        manager.run_dir = Path(config.trainer.rollout_data_dir).parent
        manager.run_dir.mkdir(parents=True, exist_ok=True)

    def _compute_old_log_prob(self, batch):
        behavior = copy.deepcopy(batch)
        for key in ("prompts", "input_ids", "attention_mask", "position_ids"):
            behavior.batch[key] = batch.batch["behavior_" + key]
        return super()._compute_old_log_prob(behavior)

    def _update_actor(self, batch):
        n = len(batch)
        response_mask = batch.batch["response_mask"]
        pg_tokens = max(1, int(response_mask.sum()))
        batch.batch["advantages"] = batch.batch["raise_advantage"][:, None] * response_mask
        batch.batch["returns"] = batch.batch["advantages"].clone()
        batch.batch["raise_replay"] = torch.zeros(n, dtype=torch.bool)
        batch.meta_info.pop("temperature", None)
        batch.meta_info["multi_turn"] = False

        td = left_right_2_no_padding(batch.to_tensordict())
        tu.assign_non_tensor(
            td,
            calculate_entropy=False,
            distillation_use_topk=False,
            global_batch_size=n,
            mini_batch_size=n,
            epochs=1,
            seed=self.config.actor_rollout_ref.actor.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            compute_loss=True,
            raise_pg_tokens=pg_tokens,
            raise_replay_count=0,
            raise_replay_weight=0.0,
        )
        output = tu.get(self.actor_rollout_wg.update_actor(td), "metrics")
        metrics = {"actor/" + key: value for key, value in output.items()}
        metrics.update({"raise/" + key: value for key, value in self.async_rollout_manager.last_raise_metrics.items()})
        return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})
