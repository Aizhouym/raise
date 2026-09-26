"""Launch feedback-guided RAISE GRPO with external veRL."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory containing train.parquet and dev.parquet")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*",
                        help="Additional Hydra overrides")
    args = parser.parse_args(argv)

    data = args.data.expanduser().resolve()
    for filename in ("train.parquet", "dev.parquet", "manifest.json"):
        if not (data / filename).is_file():
            parser.error(f"Missing {data / filename}; run raise-prepare-data first")
    model = Path(args.model_path).expanduser().resolve()
    if not (model / "config.json").is_file():
        parser.error(f"Missing model config: {model / 'config.json'}")
    if args.gpus < 1:
        parser.error("--gpus must be positive")

    run_name = args.run_name or f"raise-grpo-{datetime.now():%Y%m%d-%H%M%S}"
    if Path(run_name).name != run_name or run_name in ("", ".", ".."):
        parser.error("--run-name must be a simple directory name")

    env = os.environ.copy()
    env.update({
        "RAISE_ROOT": str(ROOT),
        "RAISE_MODEL": str(model),
        "RAISE_DATA": str(data),
        "RAISE_RUN": run_name,
        "VERL_USE_EXTERNAL_MODULES": "raise_method.register",
        "TOKENIZERS_PARALLELISM": "false",
        "HYDRA_FULL_ERROR": "1",
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
    })
    overrides = [
        "hydra.searchpath=[pkg://verl.trainer.config]",
        f"trainer.n_gpus_per_node={args.gpus}",
        f"actor_rollout_ref.actor.fsdp_config.fsdp_size={args.gpus}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.gpus}",
        *args.overrides,
    ]
    command = [sys.executable, "-m", "raise_method.main",
               "--config-path", str(ROOT / "src/raise_method/configs"),
               "--config-name", "raise", *overrides]
    print("RAISE command:", " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
