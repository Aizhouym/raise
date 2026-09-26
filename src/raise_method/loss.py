"""Context-corrected policy loss used by RAISE."""
import torch
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.padding import no_padding_2_padding


def context_corrected_loss(logp, old, ref, advantages, mask, replay, *, pg_tokens, replay_count,
               replay_weight, dp_size, beta, clip_low=0.2, clip_high=0.2, eligible=None):
    pg_mask = mask * (~replay[:, None])
    if eligible is not None:
        pg_mask = pg_mask * eligible[:, None]
    ratio = torch.exp(torch.where(replay[:, None], 0.0, logp - old.detach()))
    pg = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip_low, 1 + clip_high) * advantages)
    pg_loss = (pg * pg_mask).sum() * dp_size / pg_tokens
    delta = torch.where(replay[:, None], 0.0, ref.detach() - logp)
    kl_loss = ((delta.exp() - delta - 1) * pg_mask).sum() * dp_size / pg_tokens
    # Each answer has equal CE weight, regardless of completion length.
    per_answer = -(logp * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    ce = (per_answer * replay).sum() * dp_size / max(replay_count, 1)
    total = pg_loss + beta * kl_loss + replay_weight * ce
    return total, {"pg_loss": pg_loss, "kl_loss": kl_loss, "replay_nll": ce,
                   "replay_weighted_loss": replay_weight * ce}


def raise_actor_loss(config, model_output, data, dp_group=None):
    logp = no_padding_2_padding(model_output["log_probs"], data)
    counts = {k: tu.get_non_tensor_data(data, k, None) for k in
              ("raise_pg_tokens", "raise_replay_count", "raise_replay_weight")}
    dp_size = data["dp_size"]
    keys = ["old_log_probs", "ref_log_prob", "advantages", "response_mask", "raise_replay"]
    if "raise_verifier_valid" in data.keys():
        keys.append("raise_verifier_valid")
    fields = data.select(*keys).to_padded_tensor()
    loss, values = context_corrected_loss(logp, fields["old_log_probs"], fields["ref_log_prob"],
        fields["advantages"], fields["response_mask"], fields["raise_replay"].bool(),
        pg_tokens=counts["raise_pg_tokens"], replay_count=counts["raise_replay_count"],
        replay_weight=counts["raise_replay_weight"], dp_size=dp_size,
        beta=config.kl_loss_coef if config.use_kl_loss else 0,
        clip_low=config.clip_ratio_low, clip_high=config.clip_ratio_high,
        eligible=fields.get("raise_verifier_valid"))
    return loss, {k: Metric(value=v, aggregation=AggregationType.SUM) for k, v in values.items()}
