from dataclasses import replace
import asyncio

import pytest

from raise_method.feedback import canonicalize_counterexample, reduce_counterexample
from raise_method.policy import extract_policy
from raise_method.routing import (
    Trajectory,
    normalize_group_advantages,
    route_rollout_group,
    select_failed_checks,
)


def _trajectory(*, exact=False, source="unguided", behavior_prompt_ids=(1, 2)):
    checks = {"floor:owner": {"passed": exact, "description": "owner must read"}}
    return Trajectory(
        prompt_ids=(1, 2),
        behavior_prompt_ids=behavior_prompt_ids,
        response_ids=(3, 4),
        behavior_log_probs=(-1.0, -1.0),
        exact=exact,
        policy_version="v1",
        checks=checks,
        source=source,
    )


def test_policy_extraction_accepts_tag_and_plain_text():
    assert extract_policy("<cedar_policy>permit (principal, action, resource);</cedar_policy>")
    assert extract_policy("```cedar\npermit (principal, action, resource);\n```")
    assert extract_policy("not a policy") is None


def test_counterexample_reduction_preserves_request_facts():
    raw = (
        'principal: User::"opaque-user", action: Action::"read", '
        'resource: File::"opaque-file"\n'
        'entities: User::"opaque-user" {role: "analyst", unused: false}; '
        'File::"opaque-file" {owner: User::"opaque-user", unused: false}'
    )
    canonical = canonicalize_counterexample(raw)
    reduced = reduce_counterexample(canonical, {"owner"})
    assert 'Action::"read"' in reduced
    assert "unused" not in reduced
    assert "owner" in reduced


def test_failed_check_selection_is_frequency_ordered():
    rows = [_trajectory(), _trajectory(), _trajectory()]
    rows[2] = replace(rows[2], checks={"floor:other": {"passed": False}})
    selected = select_failed_checks(rows, max_checks=2)
    assert [item[0] for item in selected] == ["floor:owner"]
    assert sum(item[2] for item in selected) == len(rows)


def test_normalization_requires_a_real_group():
    with pytest.raises(ValueError):
        normalize_group_advantages([1])
    values = normalize_group_advantages([0, 1])
    assert values[0] < 0 < values[1]


def test_route_rollout_group_preserves_original_context():
    rows = [_trajectory(), _trajectory()]

    async def generate_guided(check_id, check, count):
        assert check_id == "floor:owner"
        assert count == 2
        return [
            _trajectory(exact=True, source="guided", behavior_prompt_ids=(9, 9)),
            _trajectory(exact=False, source="guided", behavior_prompt_ids=(9, 9)),
        ]

    routed = asyncio.run(route_rollout_group(rows, generate_guided, max_checks=1))
    assert {row.source for row in routed} == {"guided"}
    assert all(row.prompt_ids == (1, 2) for row in routed)
    assert all(row.policy_version == "v1" for row in routed)
