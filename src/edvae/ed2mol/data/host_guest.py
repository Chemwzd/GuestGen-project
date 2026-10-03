"""LMDB dataset for host/guest electron-density pairs.

Each LMDB record is a pickled dict with the keys ``base_id`` (str), ``aug_idx``
(int), ``host_ed`` and ``guest_ed`` (float32 arrays of shape 48^3), as written
by ``preprocessing.lmdb_builder``. ``base_id`` identifies the underlying
host/guest complex; all augmented copies of one complex share the same
``base_id`` and are kept inside a single train/validation split.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import lmdb
import torch
from loguru import logger
from torch.utils.data import Dataset


def read_base_id_prefix(raw, limit: int = 512) -> str:
    """Return the ``base_id`` at the start of one pickled LMDB record.

    Only the first ``limit`` bytes are parsed with ``pickletools``, which
    avoids materializing the two 48^3 arrays when, for example, the training
    split is computed for a full-size dataset. ``raw`` may be any buffer
    (e.g. a ``memoryview`` returned by an LMDB transaction opened with
    ``buffers=True``); only the sliced prefix is copied. Falls back to a full
    ``pickle.loads`` if the prefix cannot be parsed.
    """

    import pickletools

    prefix = bytes(raw[:limit])
    expect_value = False
    try:
        for opcode, argument, _ in pickletools.genops(prefix):
            if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
                if expect_value:
                    return str(argument)
                expect_value = argument == "base_id"
    except Exception:
        pass

    record = pickle.loads(bytes(raw))
    return str(record["base_id"])


class HGLMDBDataset(Dataset):
    """Map-style dataset over an ED-VAE LMDB.

    LMDB environments cannot be shared across DataLoader worker processes, so
    the environment is opened lazily by the process that first reads it.
    """

    def __init__(self, data_dir: Path):
        super(HGLMDBDataset, self).__init__()

        self.data_dir = Path(data_dir)

        self.env = None
        self._env_pid = None

        meta_env = lmdb.open(
            str(self.data_dir),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with meta_env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        meta_env.close()

        logger.info(
            f"LMDB dataset initialized. Path: {self.data_dir}, Size: {self.length}"
        )

    def _init_env(self) -> None:
        self.env = lmdb.open(
            str(self.data_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._env_pid = os.getpid()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        # LMDB environments must not be shared across `fork`ed DataLoader
        # workers; reopen whenever the reading process differs.
        if self.env is None or self._env_pid != os.getpid():
            if self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass
                self.env = None
            self._init_env()

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(f"{index}".encode("ascii"))

        if byteflow is None:
            raise RuntimeError(
                f"Key {index} not found in LMDB dataset {self.data_dir}; "
                "the dataset may be truncated or built with non-contiguous keys."
            )

        data = pickle.loads(byteflow)

        host_ed = torch.from_numpy(data["host_ed"].copy())
        guest_ed = torch.from_numpy(data["guest_ed"].copy())

        return host_ed, guest_ed
