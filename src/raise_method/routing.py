"""Backend-independent RAISE rollout coordinator.

This module does not yet bind the coordinator to veRL's distributed workers.
The callback interface makes the token/context contract executable in CPU tests.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Trajectory:
    prompt_ids: tuple[int, ...]
    behavior_prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    behavior_log_probs: tuple[float, ...]
    exact: bool
    policy_version: str
    checks: dict
    syntax_valid: bool = True
    source: str = "unguided"
    subgroup: str = "unguided"
    advantage: float = 0.0

    def __post_init__(self):
        if not self.prompt_ids or not self.behavior_prompt_ids or not self.response_ids:
            raise ValueError("Trajectory tokens must be nonempty and unpadded")
        if any(type(token) is not int or token < 0 for ids in
               (self.prompt_ids, self.behavior_prompt_ids, self.response_ids) for token in ids):
            raise ValueError("Trajectory token IDs must be nonnegative integers")
        if len(self.response_ids) != len(self.behavior_log_probs):
            raise ValueError("Behavior log probabilities must align with response token IDs")
        if not all(math.isfinite(x) for x in self.behavior_log_probs):
            raise ValueError("Non-finite behavior log probability")
        if self.source not in ("unguided", "guided"):
            raise ValueError("Invalid trajectory source")
        if self.source == "unguided" and self.prompt_ids != self.behavior_prompt_ids:
            raise ValueError("Unguided trajectory has a different behavior context")
        expected = bool(self.syntax_valid and self.checks and all(c["passed"] for c in self.checks.values()))
        if type(self.exact) is not bool or self.exact != expected:
            raise ValueError("Exact label disagrees with full-verifier result")


def normalize_group_advantages(rewards, epsilon=1e-4):
    """Match TRL OC's per-subgroup sample standard deviation (ddof=1)."""
    if len(rewards) < 2:
        raise ValueError("OC advantage subgroup requires at least two samples")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1))
    return [(r - mean) / (std + epsilon) for r in rewards]


def select_failed_checks(rows, max_checks=4):
    if max_checks < 1:
        raise ValueError("max_checks must be positive")
    frequency = Counter()
    details = {}
    for row in rows:
        if row.syntax_valid:
            for cid, check in row.checks.items():
                if not check["passed"]:
                    frequency[cid] += 1
                    details.setdefault(cid, check)
    selected = frequency.most_common(min(max_checks, len(rows) // 2))
    if not selected:
        return []
    base, remainder = divmod(len(rows), len(selected))
    return [(cid, details[cid], base + (i < remainder)) for i, (cid, _) in enumerate(selected)]


def build_feedback_messages(messages, check, scenario_path):
    """Reproduce the fresh-restart TRL hint; never insert an earlier policy.

    `check` is one check dict, or a list of them for per-rollout self-feedback
    (guidance_mode=self), where a single rollout is told about every check IT
    violated. Multiple descriptions are joined with the same "\n- " separator the
    TRL implementation uses, so the two backends build identical hint text.
    """
    import copy
    from pathlib import Path
    from raise_method.reward import get_verifier
    get_verifier()
    from raise_method.feedback import (
        attr_names_in,
        canonicalize_counterexample,
        enum_types_from_schema,
        reduce_counterexample,
    )
    # Set RAISE_RAW_COUNTEREXAMPLE=1 to bypass deterministic reduction.
    import os as _os
    if _os.environ.get("RAISE_RAW_COUNTEREXAMPLE") == "1":
        reduce_counterexample = attr_names_in = None
    checks = list(check) if isinstance(check, (list, tuple)) else [check]
    if not checks:
        raise ValueError("Guided hint requires at least one failed check")
    enums = None
    blocks = []
    for item in checks:
        description = item["description"]
        if item.get("counterexample"):
            if enums is None:
                enums = enum_types_from_schema((Path(scenario_path) / "schema.cedarschema").read_text())
            canon = canonicalize_counterexample(item["counterexample"], enums)
            case = (reduce_counterexample(canon, attr_names_in(description))
                    if reduce_counterexample
                    else canon.replace("\n", " ").strip()[:600])
            description += "\n  concrete case your earlier attempt decided wrongly: " + case
        blocks.append(description)
    hint = ("\n\nA verifier found your earlier attempts violated this requirement; "
            "make sure your policy satisfies it:\n- " + "\n- ".join(blocks))
    messages = copy.deepcopy(list(messages))
    for message in reversed(messages):
        if message["role"] == "user":
            message["content"] += hint
            break
    else:
        raise ValueError("RAISE prompt has no user message")
    return messages


def build_feedback_prompt(messages, check, scenario_path, tokenizer):
    messages = build_feedback_messages(messages, check, scenario_path)
    return tuple(tokenizer.apply_chat_template(messages, tokenize=True,
                 return_dict=False, add_generation_prompt=True, enable_thinking=False))


async def route_rollout_group(rows, generate_guided, *, max_checks=4, capture_unguided=None):
    """Route a whole K-sibling group; callbacks receive check metadata and count.

    generate_guided(cid, check, count) must return fully verified Trajectory
    objects with original prompt_ids and actual behavior_prompt_ids/log_probs.
    Generated response IDs must be retained verbatim, never decode/re-encoded.
    """
    from dataclasses import replace
    rows = list(rows)
    if len(rows) < 2 or any(r.source != "unguided" for r in rows):
        raise ValueError("Expected at least two fresh unguided sibling trajectories")
    first = rows[0]
    if any((r.prompt_ids, r.policy_version) != (first.prompt_ids, first.policy_version) for r in rows):
        raise ValueError("Mixed prompt identities or policy versions")
    if capture_unguided is not None:
        capture_unguided([r for r in rows if r.exact])

    def with_adv(group, subgroup):
        return [replace(r, subgroup=subgroup, advantage=a)
                for r, a in zip(group, normalize_group_advantages([int(r.exact) for r in group]))]

    if any(r.exact for r in rows):
        return with_adv(rows, "unguided")
    plan = select_failed_checks(rows, max_checks)
    if not plan:
        return with_adv(rows, "unguided")
    outputs = await asyncio.gather(*(generate_guided(cid, check, count) for cid, check, count in plan))
    guided = []
    for (cid, _, count), group in zip(plan, outputs):
        if len(group) != count:
            raise ValueError("Guided sampler returned the wrong subgroup size")
        for row in group:
            if (row.source != "guided" or row.prompt_ids != first.prompt_ids or
                    row.policy_version != first.policy_version):
                raise ValueError("Guided trajectory lost its original context or policy version")
        if len({row.behavior_prompt_ids for row in group}) != 1:
            raise ValueError("An advantage subgroup must share one behavior context")
        guided.extend(with_adv(group, cid))
    return guided if any(r.exact for r in guided) else with_adv(rows, "unguided")
