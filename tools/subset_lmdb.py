#!/usr/bin/env python3
"""Copy a small read-only subset of an ED-VAE LMDB dataset.

Used by the training example: a few dozen entries are enough to exercise the
data loading, split, forward/backward and checkpoint code paths.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import lmdb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--entries", type=int, default=40)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if args.entries < 4:
        parser.error("--entries must be at least 4")
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        parser.error("output must not be inside the source LMDB")
    if (output / "data.mdb").is_file():
        print(f"[SKIP] subset LMDB already exists: {output}")
        return

    # Build in a temporary directory and publish atomically, so an interrupted
    # or failed run can never masquerade as a complete subset.
    building = output.with_name(f"{output.name}.building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True, exist_ok=True)
    if output.exists() and not any(output.iterdir()):
        output.rmdir()

    input_env = lmdb.open(
        str(source), readonly=True, lock=False, readahead=False, meminit=False
    )
    output_env = None
    try:
        output_env = lmdb.open(str(building), map_size=2 * 1024**3)
        base_ids: set[str] = set()
        copied = 0
        with input_env.begin(write=False) as source_txn, output_env.begin(write=True) as target_txn:
            available = source_txn.stat()["entries"]
            for index in range(min(args.entries, available)):
                raw = source_txn.get(str(index).encode("ascii"))
                if raw is None:
                    raise RuntimeError(f"Missing source key {index}")
                record = pickle.loads(raw)
                base_ids.add(str(record["base_id"]))
                target_txn.put(str(copied).encode("ascii"), raw)
                copied += 1
        output_env.sync()
        output_env.close()
        output_env = None
        manifest = {
            "source": str(source),
            "output": str(output),
            "entries": copied,
            "unique_base_ids": len(base_ids),
        }
        (building / "subset_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(building, output)
    except BaseException:
        if output_env is not None:
            output_env.close()
        shutil.rmtree(building, ignore_errors=True)
        raise
    finally:
        input_env.close()
    print(f"[DONE] copied {copied} entries across {len(base_ids)} base IDs")


if __name__ == "__main__":
    main()
