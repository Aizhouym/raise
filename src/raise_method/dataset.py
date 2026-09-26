"""Install the bundled Cedar training corpus into a working data directory."""
from __future__ import annotations

import argparse
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = ROOT / "data" / "cedarforge-scenarios.tar.gz"


def unpack_dataset(archive: Path = DEFAULT_ARCHIVE, output: Path = ROOT / "data") -> Path:
    """Safely unpack the bundled manifest and scenario directories."""
    archive = archive.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        root = output
        for member in bundle.getmembers():
            destination = (root / member.name).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"Unsafe dataset member: {member.name}")
        bundle.extractall(root)
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args(argv)
    print(f"dataset unpacked to {unpack_dataset(args.archive, args.output)}")


if __name__ == "__main__":
    main()
