"""Convert RAISE JSONL partitions to veRL-compatible Parquet files.

The input rows are produced by :mod:`raise_method.build_splits` and contain
``id``, ``prompt`` and ``scenario_path``.  Scenario files are intentionally
external to this repository; this command only prepares a user-provided
corpus for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenario_dir(row: dict, scenario_root: Path | None) -> Path:
    path = Path(row["scenario_path"]).expanduser()
    if not path.is_absolute() and scenario_root is not None:
        path = scenario_root / path
    path = path.resolve()
    for filename in ("schema.cedarschema", "verification_plan.py"):
        if not (path / filename).is_file():
            raise ValueError(f"Missing {filename}: {path}")
    if not (path / "references").is_dir():
        raise ValueError(f"Missing references directory: {path}")
    return path


def convert(rows: list[dict], split: str, scenario_root: Path | None = None) -> list[dict]:
    if not rows:
        raise ValueError(f"Empty {split} partition")
    ids = [row.get("id") for row in rows]
    if any(not ident for ident in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate or missing IDs in {split}")

    output = []
    for index, row in enumerate(rows):
        path = scenario_dir(row, scenario_root)
        extra = dict(row.get("extra_info") or {})
        extra.update({"id": row["id"], "index": index, "split": split,
                      "scenario_path": str(path)})
        for key in ("family", "sig"):
            if key in row:
                extra[key] = row[key]
        output.append({
            "data_source": "raise_exact_v1",
            "prompt": row["prompt"],
            "ability": "cedar_policy_synthesis",
            "reward_model": {"style": "rule", "ground_truth": str(path)},
            "extra_info": extra,
        })
    return output


def _check_disjoint(train: list[dict], dev: list[dict]) -> None:
    for key in ("id", "family", "sig"):
        left = {row["extra_info"].get(key) for row in train}
        right = {row["extra_info"].get(key) for row in dev}
        overlap = (left & right) - {None}
        if overlap:
            raise ValueError(f"Train/dev leakage by {key}: {sorted(overlap)[:3]}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-root", type=Path, default=None,
                        help="Base directory for relative scenario_path values")
    args = parser.parse_args(argv)

    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)
    converted = {
        "train": convert(train_rows, "train", args.scenario_root),
        "dev": convert(dev_rows, "dev", args.scenario_root),
    }
    _check_disjoint(converted["train"], converted["dev"])

    from datasets import Dataset

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("train.parquet", "dev.parquet", "manifest.json")):
        raise FileExistsError(f"Output already exists: {output}")
    for split, rows in converted.items():
        Dataset.from_list(rows).to_parquet(str(output / f"{split}.parquet"))

    manifest = {
        "counts": {key: len(rows) for key, rows in converted.items()},
        "sources": {str(path): sha256(path) for path in (args.train, args.dev)},
        "parquet_sha256": {key: sha256(output / f"{key}.parquet") for key in converted},
        "data_source": "raise_exact_v1",
        "disjoint_by": ["id", "family", "sig"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
