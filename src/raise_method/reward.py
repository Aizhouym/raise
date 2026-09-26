"""Exact Cedar reward used by the RAISE training loop.

The formal verifier is shared with TRL. Partial check counts are diagnostics,
never the optimization reward. No candidate.cedar is read by this module.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
import os
from pathlib import Path
import shutil
import weakref

_LIMITERS = weakref.WeakKeyDictionary()


@lru_cache(maxsize=1)
def get_verifier():
    # Match the existing verifier defaults, but fail before scoring if binaries
    # are unavailable instead of silently treating infrastructure failures as 0.
    default = Path.home() / "anaconda3/envs/vllm/bin"
    for key, name in (("CEDAR", "cedar"), ("CVC5", "cvc5")):
        executable = os.environ.get(key) or (str(default / name) if (default / name).is_file() else shutil.which(name))
        if not executable or not Path(executable).exists() and not shutil.which(executable):
            raise RuntimeError(f"Missing {key} executable: {executable}")
        os.environ[key] = executable
    return verify_candidate


def verification_info(result):
    """Keep duplicate semantic names as separate obligations."""
    gates = {"syntax", "validation"}
    if any(r.check_type in gates and not r.passed for r in result.results):
        return False, {}
    info = {f"{i}:{r.check_type}:{r.check_name}": {
        "name": r.check_name, "type": r.check_type, "description": r.description,
        "passed": bool(r.passed), "counterexample": r.counterexample or ""}
        for i, r in enumerate(result.results) if r.check_type not in gates}
    return True, info


def verify_candidate(solution, scenario_path):
    from raise_method.policy import extract_policy, make_workspace
    from raise_method.verifier.orchestrator import run_verification
    policy = extract_policy(solution)
    if not policy:
        return False, {}
    with make_workspace(Path(scenario_path), policy) as workspace:
        result = run_verification(str(workspace.path))
    return verification_info(result)


def score_candidate(solution, scenario_path):
    path = Path(scenario_path).resolve()
    if not (path / "verification_plan.py").is_file():
        raise ValueError(f"Invalid verifier scenario: {path}")
    syntax_valid, info = get_verifier()(solution, str(path))
    exact = bool(syntax_valid and info and all(c["passed"] for c in info.values()))
    return {"score": float(exact), "exact": exact, "syntax_valid": bool(syntax_valid),
            "n_checks": len(info), "n_passed": sum(c["passed"] for c in info.values()),
            "checks": info}


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Official veRL custom-reward signature; keep subprocess work off the event loop."""
    if data_source != "raise_exact_v1":
        raise ValueError(f"Unexpected reward data source: {data_source}")
    if extra_info and Path(extra_info["scenario_path"]).resolve() != Path(ground_truth).resolve():
        raise ValueError("Reward ground truth and scenario metadata disagree")
    loop = asyncio.get_running_loop()
    if loop not in _LIMITERS:
        concurrency = int(os.environ.get("CEDAR_VERIFIER_CONCURRENCY_PER_WORKER", "2"))
        if concurrency < 1:
            raise ValueError("Verifier concurrency must be positive")
        _LIMITERS[loop] = asyncio.Semaphore(concurrency)
    async with _LIMITERS[loop]:
        result = await asyncio.to_thread(score_candidate, solution_str, ground_truth)
    # Avoid nested objects in veRL's scalar reward metrics. RAISE consumes score_candidate()
    # directly, keeping failed-check descriptions out of validation aggregates.
    return {key: result[key] for key in ("score", "exact", "syntax_valid", "n_checks", "n_passed")}
