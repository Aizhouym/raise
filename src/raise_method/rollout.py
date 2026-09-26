"""veRL rollout adapter for RAISE feedback-guided exploration."""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
import copy
import json

import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.utils.ray_utils import auto_await

from .reward import score_candidate
from .routing import build_feedback_messages, normalize_group_advantages


def token_rows(output):
    width = output.batch["prompts"].shape[1]
    return [
        (
            output.batch["prompts"][i][output.batch["attention_mask"][i, :width].bool()].tolist(),
            output.batch["responses"][i][output.batch["response_mask"][i].bool()].tolist(),
        )
        for i in range(len(output))
    ]


def classify_guided_outcome(subgroup_rewards):
    """Classify a guided group using only subgroup-local binary rewards."""
    if not subgroup_rewards or any(not values for values in subgroup_rewards):
        raise ValueError("Guided outcome requires non-empty subgroups")
    if any(any(value not in (0, 1, False, True) for value in values)
           for values in subgroup_rewards):
        raise ValueError("Guided outcomes require binary rewards")
    if any(0 < sum(values) < len(values) for values in subgroup_rewards):
        return "advantage_recovered"
    if any(any(values) for values in subgroup_rewards):
        return "success_without_advantage"
    return "unrecovered"


def set_context(batch, index, prompt_ids, response_ids, pad_id):
    """Keep the original prompt while retaining response tokens verbatim."""
    prompt_width = batch["prompts"].shape[1]
    response_width = batch["responses"].shape[1]
    if not prompt_ids or not response_ids or len(prompt_ids) > prompt_width or len(response_ids) > response_width:
        raise ValueError("Context does not fit configured prompt/response widths")
    batch["prompts"][index].fill_(pad_id)
    batch["prompts"][index, -len(prompt_ids):] = torch.tensor(prompt_ids)
    batch["responses"][index].fill_(pad_id)
    batch["responses"][index, :len(response_ids)] = torch.tensor(response_ids)
    batch["input_ids"][index] = torch.cat((batch["prompts"][index], batch["responses"][index]))
    mask = batch["attention_mask"][index]
    mask.zero_()
    mask[prompt_width - len(prompt_ids):prompt_width + len(response_ids)] = 1
    batch["response_mask"][index] = mask[prompt_width:]
    batch["position_ids"][index] = (mask.cumsum(-1) - 1).clamp_min(0)


class RaiseAgentLoopManager(AgentLoopManager):
    """Generate from the original prompt, then explore selected failure regions."""

    async def _base(self, prompts):
        workers = len(self.agent_loop_workers)
        count = len(prompts)
        indices = list(range(count)) + [count - 1] * ((-count) % workers)
        output = await super().generate_sequences(prompts.select_idxs(indices))
        return output.select_idxs(list(range(count)))

    async def _grades(self, output, prompts):
        semaphore = asyncio.Semaphore(16)
        rows = token_rows(output)

        async def one(index):
            async with semaphore:
                completion = self.tokenizer.decode(rows[index][1], skip_special_tokens=True)
                scenario = prompts.non_tensor_batch["extra_info"][index]["scenario_path"]
                return await asyncio.to_thread(score_candidate, completion, scenario)

        return await asyncio.gather(*(one(i) for i in range(len(output))))

    @auto_await
    async def generate_sequences(self, prompts):
        if prompts.meta_info.get("validate", False):
            return await self._base(prompts)

        original = await self._base(prompts)
        grades = await self._grades(original, prompts)
        original_rows = token_rows(original)
        groups = defaultdict(list)
        for index, uid in enumerate(prompts.non_tensor_batch["uid"]):
            groups[uid].append(index)

        result = copy.deepcopy(original)
        result.batch["raise_advantage"] = torch.zeros(len(result), dtype=torch.float32)
        result.batch["raise_guided"] = torch.zeros(len(result), dtype=torch.bool)
        for key in ("prompts", "input_ids", "attention_mask", "position_ids"):
            result.batch["behavior_" + key] = original.batch[key].clone()

        mode = self.config.raise_method.get("guidance_mode", "top_freq")
        max_checks = self.config.raise_method.get("max_guidance_checks", 4)
        requested, messages, subgroups, candidate_groups = [], [], [], []
        metrics = {"groups": len(groups), "all_zero_groups": 0, "guided_groups": 0,
                   "guided_rescued_groups": 0, "guided_responses": 0,
                   "informative_guided_subgroups": 0, "unguidable_all_zero_groups": 0}

        for indices in groups.values():
            expected = self.config.actor_rollout_ref.rollout.n
            if len(indices) != expected:
                raise ValueError("RAISE received an incomplete rollout group")
            result.batch["raise_advantage"][indices] = torch.tensor(
                normalize_group_advantages([int(grades[i]["exact"]) for i in indices]))
            if any(grades[i]["exact"] for i in indices):
                continue

            metrics["all_zero_groups"] += 1
            counts, details = Counter(), {}
            for index in indices:
                if not grades[index]["syntax_valid"]:
                    continue
                for check_id, check in grades[index]["checks"].items():
                    if not check["passed"]:
                        counts[check_id] += 1
                        details.setdefault(check_id, check)
            selected = counts.most_common(min(max_checks, len(indices) // 2))
            if not selected:
                metrics["unguidable_all_zero_groups"] += 1
                continue

            raw_messages = prompts.non_tensor_batch["raw_prompt"][indices[0]]
            start = len(requested)
            if mode == "self":
                ranked = [check_id for check_id, _ in selected]
                for index in indices:
                    failed = [check for check in grades[index]["checks"].values() if not check["passed"]]
                    messages.append(build_feedback_messages(
                        raw_messages, failed or [details[ranked[0]]],
                        prompts.non_tensor_batch["extra_info"][index]["scenario_path"],
                    ))
                    requested.append(indices[0])
                subgroups.append(list(range(start, len(requested))))
            else:
                base, remainder = divmod(len(indices), len(selected))
                for offset, (check_id, _) in enumerate(selected):
                    count = base + int(offset < remainder)
                    message = build_feedback_messages(
                        raw_messages, details[check_id],
                        prompts.non_tensor_batch["extra_info"][indices[0]]["scenario_path"],
                    )
                    low = len(requested)
                    requested.extend([indices[0]] * count)
                    messages.extend([message] * count)
                    subgroups.append(list(range(low, low + count)))
            candidate_groups.append((indices, list(range(start, len(requested)))))

        if requested:
            guided_in = copy.deepcopy(prompts.select_idxs(requested))
            guided_in.non_tensor_batch["raw_prompt"] = messages
            guided = await self._base(guided_in)
            guided_grades = await self._grades(guided, guided_in)
            guided_rows = token_rows(guided)
            guided_advantages = torch.zeros(len(guided), dtype=torch.float32)
            for subgroup in subgroups:
                values = [int(guided_grades[i]["exact"]) for i in subgroup]
                guided_advantages[subgroup] = torch.tensor(normalize_group_advantages(values))
                metrics["informative_guided_subgroups"] += int(0 < sum(values) < len(values))

            outcomes = {}
            for destinations, sources in candidate_groups:
                source_set = set(sources)
                subgroup_values = []
                for subgroup in subgroups:
                    overlap = source_set.intersection(subgroup)
                    if overlap:
                        if overlap != set(subgroup):
                            raise RuntimeError("Feedback subgroup crossed a rollout group")
                        subgroup_values.append([int(guided_grades[i]["exact"]) for i in subgroup])
                outcome = classify_guided_outcome(subgroup_values)
                outcomes[tuple(sources)] = outcome
                metrics["guided_groups"] += 1
                if outcome == "advantage_recovered":
                    metrics["guided_rescued_groups"] += 1

            for destinations, sources in candidate_groups:
                if outcomes[tuple(sources)] == "unrecovered":
                    continue
                for destination, source in zip(destinations, sources):
                    for key in original.batch.keys():
                        result.batch[key][destination] = guided.batch[key][source]
                    for key in original.non_tensor_batch:
                        result.non_tensor_batch[key][destination] = guided.non_tensor_batch[key][source]
                    for key in ("prompts", "input_ids", "attention_mask", "position_ids"):
                        result.batch["behavior_" + key][destination] = guided.batch[key][source]
                    set_context(result.batch, destination, original_rows[destination][0],
                                guided_rows[source][1], self.tokenizer.pad_token_id)
                    result.batch["raise_advantage"][destination] = guided_advantages[source]
                    result.batch["raise_guided"][destination] = True
            metrics["guided_responses"] = len(guided)

        metrics["selected_guided_responses"] = int(result.batch["raise_guided"].sum())
        self.last_raise_metrics = metrics
        if getattr(self, "run_dir", None):
            with (self.run_dir / "raise_routing.jsonl").open("a") as output:
                output.write(json.dumps(metrics) + "\n")
        return result
