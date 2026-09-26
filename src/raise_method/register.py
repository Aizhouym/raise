"""Register RAISE objectives without editing the upstream veRL checkout."""
from collections import defaultdict

import torch
from verl.trainer.ppo.core_algos import agg_loss, register_adv_est, register_policy_loss

from .compat import install

install()


@register_adv_est("raise_grpo")
def raise_advantage(token_level_rewards, response_mask, index, **kwargs):
    scores = token_level_rewards.sum(-1)
    advantages = torch.zeros_like(scores)
    groups = defaultdict(list)
    for i, uid in enumerate(index):
        groups[uid].append(i)
    for indices in groups.values():
        if len(indices) < 2:
            raise ValueError("RAISE requires at least two samples in every group")
        values = scores[indices]
        advantages[indices] = (values - values.mean()) / (values.std(correction=1) + 1e-4)
    advantages = advantages[:, None] * response_mask
    return advantages, advantages


@register_policy_loss("raise_clip")
def raise_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                      loss_agg_mode="token-mean", config=None, rollout_is_weights=None, **kwargs):
    """Two-sided clipped GRPO, with no extra dual-clip or rollout correction."""
    if rollout_is_weights is not None:
        raise ValueError("Rollout correction is disabled for the initial parity port")
    ratio = (log_prob - old_log_prob.detach()).exp()
    clipped = ratio.clamp(1 - config.clip_ratio_low, 1 + config.clip_ratio_high)
    losses = -torch.minimum(ratio * advantages, clipped * advantages)
    loss = agg_loss(losses, response_mask, loss_agg_mode, **config.global_batch_info)
    with torch.no_grad():
        clip_fraction = (((ratio != clipped) * response_mask).sum() / response_mask.sum().clamp_min(1))
    return loss, {"actor/pg_clipfrac": clip_fraction.item()}
