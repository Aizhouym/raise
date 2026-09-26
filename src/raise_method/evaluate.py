"""Evaluate a model on an external RAISE held-out JSONL partition.

Each row must contain a chat ``prompt`` and a ``scenario_path`` pointing to a
scenario directory with ``schema.cedarschema``, ``verification_plan.py``, and
``references/``.  Generation uses vLLM; grading uses the same exact verifier
as the RAISE reward.  Per-example output is optional and is never written to
the repository by default.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _conversation(prompt):
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, dict):
        return [prompt]
    return [{"role": "user", "content": str(prompt)}]


def _scenario_path(row: dict, eval_file: Path, scenario_root: Path | None) -> Path:
    path = Path(row["scenario_path"]).expanduser()
    if not path.is_absolute():
        path = (scenario_root / path) if scenario_root else eval_file.parent / path
    return path.resolve()


def run_evaluation(args) -> list[dict]:
    records = _read_jsonl(args.eval_file)
    if not records:
        raise ValueError(f"No evaluation records found: {args.eval_file}")

    from vllm import LLM, SamplingParams

    model_kwargs = {
        "model": args.base_model,
        "tensor_parallel_size": args.tp,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem_frac,
        "enforce_eager": args.enforce_eager,
    }
    lora_request = None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        model_kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)
        lora_request = LoRARequest("raise", 1, str(args.adapter.resolve()))

    llm = LLM(**model_kwargs)
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
    conversations = [_conversation(row["prompt"]) for row in records]
    outputs = llm.chat(
        conversations,
        sampling,
        chat_template_kwargs={"enable_thinking": False},
        lora_request=lora_request,
    )
    completions = [output.outputs[0].text for output in outputs]

    from .reward import score_candidate

    def grade(item):
        index, row, completion = item
        scenario = _scenario_path(row, args.eval_file, args.scenario_root)
        score = score_candidate(completion, scenario)
        return {"index": index, "id": row.get("id", str(index)),
                "scenario_path": str(scenario), "score": score["score"],
                "passed": score["exact"], "n_checks": score["n_checks"],
                "n_passed": score["n_passed"]}

    with ThreadPoolExecutor(max_workers=args.grade_workers) as executor:
        results = list(executor.map(grade, zip(range(len(records)), records, completions)))
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--scenario-root", type=Path)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--gpu-mem-frac", type=float, default=0.90)
    parser.add_argument("--grade-workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--out", type=Path,
                        help="Optional external JSONL file for per-example scores")
    args = parser.parse_args(argv)

    results = run_evaluation(args)
    total = len(results)
    passed = sum(row["passed"] for row in results)
    print(f"RAISE held-out: exact={passed}/{total} ({100 * passed / total:.1f}%)")
    by_domain = defaultdict(list)
    for row in results:
        by_domain[Path(row["scenario_path"]).parent.name].append(row)
    for domain, rows in sorted(by_domain.items()):
        print(f"  {domain}: {sum(row['passed'] for row in rows)}/{len(rows)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(row) + "\n" for row in results))
        print(f"per-example scores -> {args.out}")


if __name__ == "__main__":
    main()
