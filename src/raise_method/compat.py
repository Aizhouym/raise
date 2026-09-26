"""Narrow compatibility fixes for the pinned veRL FSDP2 implementation."""


def snapshot_merged_params(params):
    """Detach merged weights from state_dict's aliases before base restoration.

    v0.8.0 leaves merged_lora_context before its lazy weight iterator runs.
    state_dict tensors share storage with the model, and the context restores
    base weights with copy_(). Without this snapshot, rollout receives the
    restored base weights instead of the trained, merged policy.
    """
    from verl.utils.fsdp_utils import normalize_peft_param_name

    return {name: value.detach().clone()
            for name, value in normalize_peft_param_name(params).items()}


def install():
    from verl.workers.engine.fsdp import transformer_impl

    # This imported helper is used only in the merged-LoRA export branch.
    transformer_impl.normalize_peft_param_name = snapshot_merged_params
    print("RAISE_COMPAT merged LoRA export uses independent weight snapshots", flush=True)
