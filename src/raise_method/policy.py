"""Small helpers for extracting and verifying model-produced Cedar policies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import tempfile

_POLICY_TAG = re.compile(r"<cedar_policy>\s*(.*?)\s*</cedar_policy>", re.DOTALL)


def extract_policy(completion: str) -> str | None:
    """Extract a Cedar policy from the model's tagged or raw completion."""
    match = _POLICY_TAG.search(completion)
    if match:
        return match.group(1).strip()
    text = completion.split("</think>", 1)[-1].strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text if "permit" in text or "forbid" in text else None


@dataclass
class VerificationWorkspace:
    path: Path

    def __enter__(self) -> "VerificationWorkspace":
        return self

    def __exit__(self, *_: object) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def make_workspace(scenario_dir: Path, policy_text: str) -> VerificationWorkspace:
    """Create the verifier's temporary candidate workspace."""
    path = Path(tempfile.mkdtemp(prefix="raise_verify_"))
    (path / "candidate.cedar").write_text(policy_text)
    for name in ("schema.cedarschema", "verification_plan.py", "references"):
        source = scenario_dir / name
        if source.exists():
            (path / name).symlink_to(source.resolve())
    return VerificationWorkspace(path)
