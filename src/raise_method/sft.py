#!/usr/bin/env python3
"""Fine-tune a causal language model for RAISE with LoRA + TRL SFTTrainer.

Requires: transformers >= 5.5.0 (adds qwen3_5_moe support).
          Run from conda activate vllm (upgraded to transformers 5.5.4).

Architecture notes for Qwen3.5-35B-A3B (affects LoRA target selection):
  - 40 layers, hidden 2048, 256 experts, 8 active per token.
  - Layer pattern: every 4th is full attention; the rest are linear attention.

  Full attention (Qwen3_5MoeAttention, 10 layers):
    q_proj, k_proj, v_proj, o_proj  ← nn.Linear ✓

  Linear attention (Qwen3_5MoeGatedDeltaNet, 30 layers):
    in_proj_qkv, out_proj           ← nn.Linear ✓ (in_proj_z/b/a are tiny scalar projections)

  Shared MLP (Qwen3_5MoeMLP, all 40 layers):
    gate_proj, up_proj, down_proj   ← nn.Linear ✓

  Expert params (Qwen3_5MoeExperts, all 40 layers):
    gate_up_proj, down_proj         ← nn.Parameter (fused) ✗ NOT targetable by LoRA

The script intentionally only writes checkpoints to the user-supplied output
directory.  It does not ship a model, checkpoint, dataset, or training log.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
# Defaults are deliberately local and overridable.  The release contains no
# data or model weights, so users must provide --model-path and --train-file
# (or set the corresponding environment variables).
SFT_DATASET_DIR = Path(os.environ.get("RAISE_SFT_DIR", _ROOT / "data" / "sft"))
OUTPUT_DIR = Path(os.environ.get("RAISE_SFT_OUTPUT", _ROOT / "checkpoints" / "sft"))
DEFAULT_MODEL = os.environ.get("RAISE_MODEL")

# ---------------------------------------------------------------------------
# LoRA target modules for Qwen3.5-35B-A3B
#
# Full attention (10 layers):  q_proj, k_proj, v_proj, o_proj
# Linear attention (30 layers): in_proj_qkv, out_proj
# Shared MLP (all 40 layers):   gate_proj, up_proj, down_proj
# Expert params: nn.Parameter (fused) — CANNOT be targeted by standard LoRA
# ---------------------------------------------------------------------------
LORA_TARGET_MODULES = [
    # Full attention projections
    "q_proj", "k_proj", "v_proj", "o_proj",
    # Linear (GatedDeltaNet) attention projections
    "in_proj_qkv", "out_proj",
    # Shared MLP
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# No-mmap safetensors patch (lazy, per-tensor seek-and-read)
# ---------------------------------------------------------------------------

class _LazyHeapSlice:
    """Mimics safetensors.PySafeSlice.

    Reads only this tensor's bytes from disk on __getitem__ — no mmap, no
    full-shard load. Supports get_shape(), get_dtype() for transformers'
    dtype/shape checks, and tensor[...] for materialisation.
    """

    _DTYPE_MAP: dict = {
        "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
        "I32": torch.int32,   "I64": torch.int64,   "I16": torch.int16,
        "I8":  torch.int8,    "U8":  torch.uint8,   "BOOL": torch.bool,
    }

    def __init__(self, path: str, offset: int, length: int,
                 shape: list, dtype_str: str) -> None:
        self._path = path
        self._offset = offset
        self._length = length
        self._shape = shape
        self._dtype_str = dtype_str

    def get_shape(self) -> list:
        return self._shape

    def get_dtype(self) -> str:
        return self._dtype_str

    def __getitem__(self, idx) -> torch.Tensor:
        dtype = self._DTYPE_MAP.get(self._dtype_str, torch.float32)
        with open(self._path, "rb") as fh:
            fh.seek(self._offset)
            raw = fh.read(self._length)   # read() = heap alloc, not mmap()
        buf = bytearray(raw)              # mutable buffer for torch.frombuffer
        del raw
        t = torch.frombuffer(buf, dtype=dtype).reshape(self._shape).clone()
        del buf
        return t[idx]


class _LazyHeapSafeOpen:
    """Drop-in for safetensors.safe_open.

    Parses only the JSON header in __init__ (~KB) and returns _LazyHeapSlice
    objects that seek-and-read individual tensors on demand. No mmap() is ever
    called; peak CPU RAM is one tensor at a time (~MB to ~100 MB) instead of
    one full shard (~5 GB).
    """

    def __init__(self, path: str, framework: str = "pt", device: str = "cpu") -> None:
        import json
        import struct
        self._path = path
        with open(path, "rb") as fh:
            hdr_len = struct.unpack("<Q", fh.read(8))[0]
            hdr_raw = fh.read(hdr_len)
            self._data_start = 8 + hdr_len
        hdr = json.loads(hdr_raw)
        self._meta = hdr.pop("__metadata__", {})
        self._info: dict = {}
        for name, v in hdr.items():
            s, e = v["data_offsets"]
            self._info[name] = {
                "dtype":  v["dtype"],
                "shape":  v["shape"],
                "offset": self._data_start + s,
                "length": e - s,
            }
        print(f"    [lazy-no-mmap] indexed {Path(path).name} "
              f"({len(self._info)} tensors)", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def keys(self) -> list:
        return list(self._info.keys())

    def metadata(self) -> dict:
        return self._meta

    def offset_keys(self) -> list:
        return []

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.get_slice(key)[...]

    def get_slice(self, key: str) -> _LazyHeapSlice:
        i = self._info[key]
        return _LazyHeapSlice(self._path, i["offset"], i["length"],
                              i["shape"], i["dtype"])


def _patch_safe_open_no_mmap() -> None:
    """Monkey-patch transformers to use _LazyHeapSafeOpen instead of safe_open.

    Must be called before AutoModelForCausalLM.from_pretrained().
    """
    import transformers.modeling_utils as _mu
    _mu.safe_open = _LazyHeapSafeOpen


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",  default=DEFAULT_MODEL)
    p.add_argument("--train-file",  default=str(SFT_DATASET_DIR / "train.jsonl"))
    p.add_argument("--output-dir",  default=str(OUTPUT_DIR))
    p.add_argument("--lora-rank",   type=int,   default=16)
    p.add_argument("--lora-alpha",  type=int,   default=32)
    p.add_argument("--lora-dropout",type=float, default=0.05)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--per-device-batch", type=int, default=2,
                   help="Per-GPU batch size. 67GB model leaves ~74GB on H200 — batch=2 is safe.")
    p.add_argument("--grad-accum",  type=int,   default=4,
                   help="Gradient accumulation steps. Effective batch = 4GPUs × 2 × 4 = 32.")
    p.add_argument("--max-seq-len", type=int,   default=4096)
    p.add_argument("--use-qlora",   action="store_true",
                   help="4-bit QLoRA (only needed if running with less than 4×H200).")
    p.add_argument(
        "--disable-lazy-no-mmap",
        action="store_true",
        help="Use the default safetensors mmap loader instead of the heap-read loader.",
    )
    return p.parse_args()


def run_sft() -> None:
    args = parse_args()
    if not args.model_path:
        raise SystemExit("Provide --model-path or set RAISE_MODEL")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if is_main:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        eff_batch = world_size * args.per_device_batch * args.grad_accum
        print(f"Model:        {args.model_path}")
        print(f"Train:        {args.train_file}")
        print(f"Output:       {args.output_dir}")
        print(f"LoRA:         rank={args.lora_rank}  alpha={args.lora_alpha}")
        print(f"LoRA targets: {LORA_TARGET_MODULES}")
        print(f"LR:           {args.lr}  epochs={args.epochs}")
        print(f"Eff. batch:   {eff_batch}  ({world_size}GPUs × {args.per_device_batch} × {args.grad_accum})")
        print(f"QLoRA:        {args.use_qlora}")
        print(f"Lazy no-mmap: {not args.disable_lazy_no_mmap}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Distributed init — needed for staggered loading barrier below
    # ------------------------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    # ------------------------------------------------------------------
    # Model — patch safe_open to use heap I/O (no mmap), then load normally.
    # HPC clusters set vm.overcommit_memory=2; large mmap() calls fail with
    # ENOMEM under memory pressure. _HeapSafeOpen reads via read() instead.
    # Ranks still load serially (staggered) to avoid concurrent RAM spikes.
    # ------------------------------------------------------------------
    bnb_config = None
    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if not args.disable_lazy_no_mmap:
        _patch_safe_open_no_mmap()  # replace mmap loader with heap-read loader

    for rank_turn in range(world_size):
        if rank_turn != local_rank:
            if world_size > 1:
                dist.barrier()   # wait for the rank ahead to finish
            continue

        if is_main:
            print(f"[rank {local_rank}] Loading model to cuda:{local_rank} ...")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            attn_implementation="sdpa",
            device_map={"": local_rank},
            low_cpu_mem_usage=True,
        )

        if world_size > 1:
            dist.barrier()   # signal next rank it can start

    model.config.use_cache = False  # required for gradient checkpointing
    if is_main:
        print("All ranks loaded model.")

    # Freeze the vision encoder — this is a VL model; we only train the LLM.
    # The visual tower is under model.visual (Qwen3.5 VL naming convention).
    frozen_params = 0
    for name, param in model.named_parameters():
        if "visual" in name or "vision" in name:
            param.requires_grad_(False)
            frozen_params += param.numel()
    if is_main and frozen_params > 0:
        print(f"Frozen vision encoder: {frozen_params/1e9:.2f}B params")

    # ------------------------------------------------------------------
    # QLoRA prep (only when bitsandbytes actually quantized the model)
    # ------------------------------------------------------------------
    if args.use_qlora and getattr(model, "is_quantized", False):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        # Only apply LoRA to language model layers, not vision
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    # Assistant response marker. The chat template (enable_thinking=False) injects an
    # empty "<think>\n\n</think>\n\n" block right after the assistant tag, and that same
    # block is prefilled into the generation prompt at inference (vLLM llm.chat), so the
    # model only ever generates <cedar_policy>...  Masking THROUGH the empty think block
    # keeps loss on the policy alone and makes the train/inference boundary byte-identical.
    # Fall back to the bare tag for any think-free (hand-rolled) text.
    ASSISTANT_MARKER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ASSISTANT_MARKER_FALLBACK = "<|im_start|>assistant\n"

    def format_record(example: dict) -> dict:
        """Tokenize one example with prompt tokens masked (-100 in labels).

        Training text is pre-built by datagen Stage E without any <think> tokens.
        Loss is computed only on the assistant response (<cedar_policy>...</cedar_policy>).
        """
        full_text = example["text"]
        # Split at the last assistant marker to find the response boundary
        marker = ASSISTANT_MARKER
        split_idx = full_text.rfind(marker)
        if split_idx == -1:
            marker = ASSISTANT_MARKER_FALLBACK
            split_idx = full_text.rfind(marker)
        prompt_text = full_text[:split_idx + len(marker)]

        full_enc   = tokenizer(full_text,   add_special_tokens=False,
                               truncation=True, max_length=args.max_seq_len)
        prompt_enc = tokenizer(prompt_text, add_special_tokens=False)

        input_ids  = full_enc["input_ids"]
        prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))

        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids":      input_ids,
            "attention_mask": full_enc["attention_mask"],
            "labels":         labels,
        }

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    train_ds = train_ds.map(format_record, remove_columns=train_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Optimiser
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        # Sequence length — truncation is handled in format_record; this caps
        # any residual length after collation padding.
        max_length=args.max_seq_len,
        dataset_text_field=None,
        packing=False,
        # No eval — CedarBench v2 is the held-out eval, run separately.
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=3,
        # Logging
        logging_steps=5,
        report_to="none",
        # DDP
        ddp_find_unused_parameters=True,   # True needed with frozen vision encoder
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    if is_main:
        # No eval -> no best-by-eval_loss selection; save the final adapter.
        adapter_dir = Path(args.output_dir) / "final_adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"\nFinal adapter saved → {adapter_dir}")


if __name__ == "__main__":
    run_sft()


main = run_sft
