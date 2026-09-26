#!/usr/bin/env python3
"""Merge a RAISE LoRA adapter into its base model.

Usage:
    conda activate vllm
    python -m raise_method.merge_adapter --adapter-path /path/to/adapter \
        --base-model /path/to/base --output /path/to/merged
"""
import argparse
import json
import os
import struct
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
DEFAULT_BASE = os.environ.get("RAISE_MODEL")
DEFAULT_CKPTS = Path(os.environ.get("RAISE_CHECKPOINT_ROOT", _ROOT / "checkpoints"))


def _patch_safe_open_no_mmap() -> None:
    """Replace safetensors mmap loader with heap read()-based loader.
    Avoids ENOMEM on HPC nodes where vm.overcommit_memory=2 blocks mmap.
    """
    class _Slice:
        _DTYPE = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
                  "I32": torch.int32, "I64": torch.int64, "I8": torch.int8, "U8": torch.uint8}
        def __init__(self, path, offset, length, shape, dtype_str):
            self._path, self._offset, self._length = path, offset, length
            self._shape, self._dtype_str = shape, dtype_str
        def get_shape(self): return self._shape
        def get_dtype(self): return self._dtype_str
        def __getitem__(self, idx):
            dtype = self._DTYPE.get(self._dtype_str, torch.float32)
            with open(self._path, "rb") as fh:
                fh.seek(self._offset)
                buf = bytearray(fh.read(self._length))
            return torch.frombuffer(buf, dtype=dtype).reshape(self._shape).clone()[idx]

    class _SafeOpen:
        def __init__(self, path, framework="pt", device="cpu"):
            self._path = path
            with open(path, "rb") as fh:
                hdr_len = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(hdr_len))
                self._data_start = 8 + hdr_len
            self._meta = hdr.pop("__metadata__", {})
            self._info = {k: {"dtype": v["dtype"], "shape": v["shape"],
                               "offset": self._data_start + v["data_offsets"][0],
                               "length": v["data_offsets"][1] - v["data_offsets"][0]}
                          for k, v in hdr.items()}
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def keys(self): return list(self._info.keys())
        def metadata(self): return self._meta
        def offset_keys(self): return []
        def get_tensor(self, key): return self.get_slice(key)[...]
        def get_slice(self, key):
            i = self._info[key]
            return _Slice(self._path, i["offset"], i["length"], i["shape"], i["dtype"])

    import transformers.modeling_utils as _mu
    _mu.safe_open = _SafeOpen


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="Run ID, e.g. F -> checkpoints/run_F/final_adapter")
    source.add_argument("--adapter-path", help="Explicit adapter or checkpoint directory")
    parser.add_argument("--base-model", default=DEFAULT_BASE,
                        help="Base model path (or set RAISE_MODEL)")
    parser.add_argument("--output",  default=None,
                        help="Destination directory (default: RAISE checkpoint root/merged-run<run>)")
    args = parser.parse_args()

    if not args.base_model:
        parser.error("--base-model is required unless RAISE_MODEL is set")

    adapter_dir = (Path(args.adapter_path) if args.adapter_path else
                   DEFAULT_CKPTS / f"run_{args.run}" / "final_adapter")
    if not adapter_dir.exists():
        raise SystemExit(f"Adapter not found: {adapter_dir}")

    if args.adapter_path and not args.output:
        parser.error("--adapter-path requires an explicit --output directory")
    output_dir = Path(args.output) if args.output else DEFAULT_CKPTS / f"merged-run{args.run}"

    print(f"Base model : {args.base_model}")
    print(f"Adapter    : {adapter_dir}")
    print(f"Output     : {output_dir}")
    print()

    print("Loading base model to GPU (device_map=auto)...")
    _patch_safe_open_no_mmap()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print("Loading adapter and merging...")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save directly from GPU — safetensors serialises shard-by-shard,
    # so full CPU RAM is never needed.
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")

    print("Saving tokenizer...")
    AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    ).save_pretrained(output_dir)

    # Copy auxiliary files that save_pretrained does NOT write for Qwen3.5 MoE:
    # - preprocessor_config.json / video_preprocessor_config.json: multimodal preprocessors,
    #   required by vLLM even for text-only fine-tuning.
    # - vocab.json: visual-token vocab not covered by AutoTokenizer.save_pretrained.
    # - config.json: re-copied to preserve VL-MoE specific fields that may be stripped.
    aux_files = [
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
        "config.json",
    ]
    base_dir = Path(args.base_model).expanduser()
    print("Copying auxiliary files from base model...")
    for fname in aux_files:
        src = base_dir / fname
        if src.exists():
            import shutil
            shutil.copy2(src, output_dir / fname)
            print(f"  {fname}")
        else:
            print(f"  (skip) {fname} not in base model dir")

    print(f"\nDone. Merged model saved → {output_dir}")
    print(f"\nServe with:")
    print(f"  vllm serve {output_dir} --served-model-name raise --port 8002 --max-model-len 20480")


if __name__ == "__main__":
    main()
