#!/usr/bin/env python3
"""Build leak-free SFT, RL, and held-out splits from Cedar scenarios.

Design (confirmed 2026-08-26):
  - Content-level dedup: cluster scenarios by a normalized (schema+gold) signature
    that blanks string/numeric literals, so template near-duplicates share a cluster.
  - Held-out (sacred eval): whole small clusters (size <= HELDOUT_MAX_CLUSTER),
    area-stratified, family-capped -> structurally distinctive, zero train near-twins.
  - RL: tier heuristic (total_depth in RL_TIERS) over the remaining pool, family- and
    cluster-capped for diversity -> one-shottable "donor" scenarios that keep GRPO
    groups gradient-bearing (0 < pass_rate < 1).
  - SFT: everything else non-held-out (all tiers) -> broad gold-imitation coverage.
  - Splits are scenario-disjoint; held-out signatures are disjoint from all train.
  - Outputs go to datagen/artifacts/datasets/v2/ (old files are left untouched so
    previously reported numbers stay reproducible).

The release contains no scenarios. Supply a manifest and scenario directory
from CedarForge (or another compatible corpus), for example:

    python -m raise_method.build_splits --manifest /data/manifest.jsonl \
        --scenario-root /data/scenarios --output /tmp/raise-splits
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---- paths ----
ROOT = Path(__file__).resolve().parents[2]
SCEN = Path(os.environ.get("RAISE_SCENARIO_ROOT", ROOT / "data" / "scenarios"))
MANIFEST = Path(os.environ.get("RAISE_MANIFEST", ROOT / "data" / "manifest.jsonl"))
OUT = Path(os.environ.get("RAISE_SPLITS_DIR", ROOT / "data" / "splits" / "v2"))

# ---- params ----
SEED = 0
HELDOUT_TARGET = 400
HELDOUT_MAX_CLUSTER = 2      # only unique/paired structures are eligible for held-out
HELDOUT_FAMILY_CAP = 3
RL_TARGET = 1800
RL_TIERS = {2, 3}
RL_FAMILY_CAP = 8
RL_CLUSTER_CAP = 2           # <= 2 identical-structure templates per RL family cluster

# ---- prompt format (inlined verbatim from datagen/eval/run_eval.py to avoid the
#      heavy cedar_reward import; must stay byte-identical to keep eval/finetune
#      consumers working) ----
SFT_SYSTEM = """\
You are an expert Cedar access-control policy synthesizer. Given a Cedar schema \
and a natural-language access-control requirement, write the Cedar policies that \
exactly implement it.

Rules:
- Cedar denies by default; write only permit and forbid rules.
- forbid always overrides permit.
- Use unless { ... } for exceptions to forbid rules.
- Optional attributes: always has-guard before reading.

Output Format:
Only output the final Cedar policy inside <cedar_policy> tags.

<cedar_policy>
Cedar policy here
</cedar_policy>\
"""


def build_user_message(schema: str, spec: str) -> str:
    return (
        "## Cedar Schema\n```cedar\n"
        f"{schema.strip()}\n```\n\n"
        "## Access-Control Requirement\n"
        f"{spec.strip()}\n\n"
        "Write the Cedar policies now."
    )


def chat_prompt(schema: str, spec: str) -> list[dict]:
    return [
        {"role": "system", "content": SFT_SYSTEM},
        {"role": "user", "content": build_user_message(schema, spec)},
    ]


def sft_text(schema: str, spec: str, gold: str) -> str:
    user = build_user_message(schema, spec)
    gold = gold.strip("\n")
    return (
        f"<|im_start|>system\n{SFT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<cedar_policy>\n{gold}\n</cedar_policy><|im_end|>\n"
    )


# ---- helpers ----
def tier_of(id_: str) -> int:
    m = re.search(r"-T(\d+)-\d+$", id_)
    return int(m.group(1)) if m else -1


def family_of(id_: str) -> str:
    return re.sub(r"-T\d+-\d+$", "", id_)


def extract_spec(intent_text: str) -> str:
    """The NL requirement = body after '## Natural-Language Scenario'; fallback strips
    leading markdown headers."""
    m = re.search(r"##\s*Natural-Language Scenario\s*\n(.*)", intent_text, re.S)
    body = m.group(1) if m else re.sub(r"^\s*(#[^\n]*\n)+", "", intent_text)
    return body.strip()


_STR_RE = re.compile(r'"[^"]*"')
_NUM_RE = re.compile(r"\b\d+\b")
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_ANNOT_RE = re.compile(r"@\w+\([^)]*\)")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Structural normalization: drop comments/annotations, blank string & numeric
    literals, collapse whitespace. Two policies that differ only in literal values or
    formatting map to the same string."""
    text = _LINE_COMMENT_RE.sub("", text)
    text = _ANNOT_RE.sub("", text)
    text = _STR_RE.sub('""', text)
    text = _NUM_RE.sub("0", text)
    return _WS_RE.sub(" ", text).strip()


def signature(schema: str, gold: str) -> str:
    return hashlib.md5((normalize(schema) + "||" + normalize(gold)).encode()).hexdigest()


# ---- load ----
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--scenario-root", type=Path, default=SCEN)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--heldout-target", type=int, default=HELDOUT_TARGET)
    parser.add_argument("--rl-target", type=int, default=RL_TARGET)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    scenario_root = args.scenario_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output.expanduser()
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    print(f"loaded {len(rows)} manifest rows")

    # read files once; compute derived fields
    recs = []
    for i, r in enumerate(rows):
        sid = r["id"]
        d = scenario_root / sid
        schema = (d / "schema.cedarschema").read_text(errors="replace")
        gold = (d / "candidate.cedar").read_text(errors="replace")
        intent = (d / "intent.md").read_text(errors="replace")
        recs.append({
            "id": sid,
            "area": r.get("area", "?"),
            "domain": r.get("domain", "?"),
            "origin": r.get("origin", "?"),
            "tier": tier_of(sid),
            "family": family_of(sid),
            "schema": schema,
            "spec": extract_spec(intent),
            "gold": gold,
            "sig": signature(schema, gold),
        })
        if (i + 1) % 1000 == 0:
            print(f"  read {i + 1}/{len(rows)}")
    by_id = {r["id"]: r for r in recs}

    # cluster by signature
    clusters: dict[str, list[str]] = defaultdict(list)
    for r in recs:
        clusters[r["sig"]].append(r["id"])
    csize = {sig: len(ids) for sig, ids in clusters.items()}
    n_singleton = sum(1 for s in csize.values() if s == 1)
    print(f"signature clusters: {len(clusters)} (singletons={n_singleton}, "
          f"max cluster={max(csize.values())})")

    # ---- HELD-OUT: whole small clusters, area-stratified, family-capped ----
    total = len(recs)
    area_count = Counter(r["area"] for r in recs)
    # proportional quota per area (min 5), summing ~= HELDOUT_TARGET
    area_quota = {a: max(5, round(args.heldout_target * area_count[a] / total)) for a in area_count}

    # eligible held-out scenarios = members of clusters with size <= HELDOUT_MAX_CLUSTER
    eligible = [r for r in recs if csize[r["sig"]] <= HELDOUT_MAX_CLUSTER]
    rng.shuffle(eligible)

    heldout_ids: set[str] = set()
    heldout_sigs: set[str] = set()
    area_taken: Counter = Counter()
    fam_taken: Counter = Counter()
    for r in eligible:
        if len(heldout_ids) >= args.heldout_target:
            break
        a, fam, sig = r["area"], r["family"], r["sig"]
        if sig in heldout_sigs:
            continue
        if area_taken[a] >= area_quota[a]:
            continue
        if fam_taken[fam] >= HELDOUT_FAMILY_CAP:
            continue
        # take the WHOLE cluster (<= HELDOUT_MAX_CLUSTER members) so no twin leaks to train
        members = clusters[sig]
        heldout_sigs.add(sig)
        for m in members:
            heldout_ids.add(m)
            area_taken[by_id[m]["area"]] += 1
            fam_taken[by_id[m]["family"]] += 1

    # ---- RL: tier heuristic over remaining pool, family/cluster-capped ----
    remaining = [r for r in recs if r["sig"] not in heldout_sigs and r["id"] not in heldout_ids]
    rl_candidates = [r for r in remaining if r["tier"] in RL_TIERS]
    rng.shuffle(rl_candidates)
    rl_ids: set[str] = set()
    rl_fam: Counter = Counter()
    rl_clu: Counter = Counter()
    for r in rl_candidates:
        if len(rl_ids) >= args.rl_target:
            break
        if rl_fam[r["family"]] >= RL_FAMILY_CAP:
            continue
        if rl_clu[r["sig"]] >= RL_CLUSTER_CAP:
            continue
        rl_ids.add(r["id"])
        rl_fam[r["family"]] += 1
        rl_clu[r["sig"]] += 1

    # ---- SFT: everything else non-held-out ----
    sft_ids = {r["id"] for r in remaining if r["id"] not in rl_ids}

    # ---- assertions ----
    assert heldout_ids.isdisjoint(rl_ids), "held-out overlaps RL"
    assert heldout_ids.isdisjoint(sft_ids), "held-out overlaps SFT"
    assert rl_ids.isdisjoint(sft_ids), "RL overlaps SFT"
    train_sigs = {by_id[i]["sig"] for i in rl_ids | sft_ids}
    leak = heldout_sigs & train_sigs
    assert not leak, f"CONTENT LEAK: {len(leak)} held-out signatures also in train"
    assert heldout_ids | rl_ids | sft_ids == set(by_id), "partition does not cover all scenarios"
    print(f"\nsplit sizes: heldout={len(heldout_ids)}  RL={len(rl_ids)}  SFT={len(sft_ids)}  "
          f"(sum={len(heldout_ids) + len(rl_ids) + len(sft_ids)}/{total})")
    print("assertions passed: scenario-disjoint + content-disjoint held-out + full coverage")

    # ---- write outputs ----
    (output_dir / "heldout").mkdir(parents=True, exist_ok=True)
    (output_dir / "sft").mkdir(parents=True, exist_ok=True)
    (output_dir / "rl").mkdir(parents=True, exist_ok=True)

    def spath(sid: str) -> str:
        return str(scenario_root / sid)

    with (output_dir / "heldout" / "heldout.jsonl").open("w") as fh:
        for sid in sorted(heldout_ids):
            r = by_id[sid]
            fh.write(json.dumps({
                "id": sid, "source": "heldout_v2", "domain": r["domain"],
                "scenario_path": spath(sid), "prompt": chat_prompt(r["schema"], r["spec"]),
            }) + "\n")

    with (output_dir / "rl" / "grpo_train.jsonl").open("w") as fh:
        for sid in sorted(rl_ids):
            r = by_id[sid]
            fh.write(json.dumps({
                "id": sid, "prompt": chat_prompt(r["schema"], r["spec"]),
                "scenario_path": spath(sid), "src": r["origin"],
            }) + "\n")

    with (output_dir / "sft" / "train.jsonl").open("w") as ft, (output_dir / "sft" / "meta.jsonl").open("w") as fm:
        for sid in sorted(sft_ids):
            r = by_id[sid]
            ft.write(json.dumps({"text": sft_text(r["schema"], r["spec"], r["gold"])}) + "\n")
            fm.write(json.dumps({"id": sid, "area": r["area"], "domain": r["domain"],
                                 "family": r["family"], "tier": r["tier"], "origin": r["origin"]}) + "\n")

    with (output_dir / "splits.jsonl").open("w") as fh:
        for r in recs:
            split = ("heldout" if r["id"] in heldout_ids else
                     "rl" if r["id"] in rl_ids else "sft")
            fh.write(json.dumps({
                "id": r["id"], "split": split, "area": r["area"], "domain": r["domain"],
                "family": r["family"], "tier": r["tier"], "origin": r["origin"],
                "sig": r["sig"], "cluster_size": csize[r["sig"]],
            }) + "\n")

    # ---- report ----
    def dist(name, ids, key):
        c = Counter(by_id[i][key] for i in ids)
        line = "  ".join(f"{k}:{v}" for k, v in sorted(c.items()))
        print(f"  {name} by {key}: {line}")

    lines = []
    def report(s):
        print(s); lines.append(s)

    report("\n=== SPLIT REPORT ===")
    report(f"heldout={len(heldout_ids)}  RL={len(rl_ids)}  SFT={len(sft_ids)}")
    for name, ids in [("heldout", heldout_ids), ("RL", rl_ids), ("SFT", sft_ids)]:
        report(f"\n[{name}]")
        report(f"  origin: {dict(Counter(by_id[i]['origin'] for i in ids))}")
        report(f"  tier:   {dict(sorted(Counter(by_id[i]['tier'] for i in ids).items()))}")
        report(f"  #areas: {len(set(by_id[i]['area'] for i in ids))}/19   "
               f"#families: {len(set(by_id[i]['family'] for i in ids))}")
    # RL diversity: family-size histogram
    rl_famsize = Counter(Counter(by_id[i]['family'] for i in rl_ids).values())
    report(f"\nRL family-size histogram (scenarios per family): {dict(sorted(rl_famsize.items()))}")
    report(f"held-out area coverage: {len(set(by_id[i]['area'] for i in heldout_ids))}/19 areas")
    report(f"\noutputs -> {output_dir}")
    (output_dir / "build_report.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
