#!/usr/bin/env python3
"""Read-only inspection of an ED-VAE LMDB dataset."""

from __future__ import annotations

import argparse
import json
import pickle
import pickletools
from pathlib import Path

import lmdb
import numpy as np


def base_id_from_pickle_prefix(raw: bytes) -> str:
    """Read the leading base_id without materializing the two 48^3 arrays."""

    prefix = bytes(raw[:512])
    expect_value = False
    for opcode, argument, _ in pickletools.genops(prefix):
        if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
            if expect_value:
                return str(argument)
            expect_value = argument == "base_id"
    raise RuntimeError("base_id was not found in the first 512 pickle bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lmdb_dir", type=Path)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument(
        "--scan-base-ids",
        action="store_true",
        help="Scan every entry for its base_id (prefix parse, no full unpickle)",
    )
    parser.add_argument("--expected-entries", type=int)
    parser.add_argument(
        "--expected-shape",
        default=None,
        help="Comma-separated array shape required for both channels, e.g. 48,48,48",
    )
    parser.add_argument("--expected-dtype", default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected_shape = (
        tuple(int(value) for value in args.expected_shape.split(","))
        if args.expected_shape
        else None
    )
    env = lmdb.open(
        str(args.lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    report: dict[str, object] = {"path": str(args.lmdb_dir.resolve())}
    base_ids: set[str] = set()
    entries: list[dict[str, object]] = []
    with env.begin(write=False, buffers=True) as txn:
        count = int(txn.stat()["entries"])
        report["entries"] = count
        limit = count if args.scan_base_ids else min(args.samples, count)
        for index in range(limit):
            raw = txn.get(str(index).encode("ascii"))
            if raw is None:
                raise RuntimeError(f"Missing integer key {index}")
            if args.scan_base_ids:
                base_ids.add(base_id_from_pickle_prefix(bytes(raw)))
            if index < args.samples:
                record = pickle.loads(bytes(raw))
                base_ids.add(str(record.get("base_id")))
                item: dict[str, object] = {
                    "index": index,
                    "base_id": record.get("base_id"),
                    "aug_idx": record.get("aug_idx"),
                }
                for key in ("host_ed", "guest_ed"):
                    array = np.asarray(record[key])
                    item[key] = {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "min": float(array.min()),
                        "max": float(array.max()),
                        "mean": float(array.mean()),
                    }
                entries.append(item)
    env.close()
    report["sampled_entries"] = entries
    report["scanned_unique_base_ids"] = len(base_ids)
    report["base_id_scan_complete"] = bool(args.scan_base_ids)
    errors: list[str] = []
    if args.expected_entries is not None and report["entries"] != args.expected_entries:
        errors.append(f"entries={report['entries']} (expected {args.expected_entries})")
    for item in entries:
        for channel in ("host_ed", "guest_ed"):
            details = item[channel]
            if expected_shape is not None and tuple(details["shape"]) != expected_shape:
                errors.append(
                    f"key {item['index']} {channel} shape={details['shape']} "
                    f"(expected {list(expected_shape)})"
                )
            if args.expected_dtype is not None and details["dtype"] != args.expected_dtype:
                errors.append(
                    f"key {item['index']} {channel} dtype={details['dtype']} "
                    f"(expected {args.expected_dtype})"
                )
    report["validation"] = {
        "expected_entries": args.expected_entries,
        "expected_shape": list(expected_shape) if expected_shape else None,
        "expected_dtype": args.expected_dtype,
        "pass": not errors,
        "errors": errors,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("LMDB validation failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
