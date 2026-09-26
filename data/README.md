# Bundled Cedar training data

`cedarforge-scenarios.tar.gz` contains the CedarForge scenario corpus used by
the RAISE SFT and feedback-guided GRPO recipes. It includes the manifest and
the verifier inputs for each scenario: schema, natural-language requirement,
gold policy, verification plan, and references.

Unpack it once from the repository root:

```bash
python -m raise_method.dataset
```

Then build the reproducible SFT/RL/held-out JSONL partitions:

```bash
python -m raise_method.build_splits \
  --manifest data/manifest.jsonl \
  --scenario-root data/scenarios \
  --output data/splits/v2
```

The archive contains training inputs only. It does not contain model weights,
checkpoints, generated samples, logs, or evaluation results.
