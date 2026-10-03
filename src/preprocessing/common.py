"""Small safety and provenance helpers shared by the preprocessing CLIs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


class PreprocessingError(RuntimeError):
    """A user-actionable preprocessing failure."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Atomically write UTF-8 JSON into an already-authorized output tree."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def path_is_within(candidate: Path, parent: Path) -> bool:
    candidate = candidate.resolve(strict=False)
    parent = parent.resolve(strict=False)
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def reject_output_inside_sources(output: Path, source_roots: Iterable[Path]) -> None:
    """Reject an output/work tree that could add files to a source directory.

    Input structures are treated as read-only: every run writes into its own
    output tree and never next to the inputs.
    """

    output = output.resolve(strict=False)
    for root in source_roots:
        root = root.resolve(strict=False)
        if path_is_within(output, root):
            raise PreprocessingError(
                f"Refusing output inside the input tree: output={output}, "
                f"input={root}. Choose an output directory outside the inputs."
            )


def ensure_workdir_is_safe(workdir: Path, source_roots: Iterable[Path]) -> None:
    reject_output_inside_sources(workdir, source_roots)


def copy_readonly_input(source: Path, destination: Path) -> Dict[str, Any]:
    """Copy one input before processing and return source/copy provenance."""

    if not source.is_file():
        raise PreprocessingError(f"Input file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))
    source_hash = sha256_file(source)
    copied_hash = sha256_file(destination)
    if source_hash != copied_hash:
        raise PreprocessingError(
            f"Copied input failed SHA256 verification: {source} -> {destination}"
        )
    return {
        "source": str(source.resolve()),
        "source_sha256": source_hash,
        "work_copy": str(destination.resolve()),
        "work_copy_sha256": copied_hash,
    }


def resolve_executable(
    value: Optional[str], env_names: Sequence[str], label: str
) -> Path:
    candidates: List[str] = []
    if value:
        candidates.append(value)
    for env_name in env_names:
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(env_value)
    for candidate in candidates:
        expanded = Path(candidate).expanduser()
        if expanded.is_file() and os.access(str(expanded), os.X_OK):
            return expanded.resolve()
        found = shutil.which(candidate)
        if found:
            return Path(found).resolve()
    joined = ", ".join(env_names)
    raise PreprocessingError(
        f"Cannot find executable for {label}. Pass the corresponding CLI option "
        f"or set one of: {joined}."
    )


def run_logged(
    command: Sequence[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    stdin_text: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    allowed_returncodes: Sequence[int] = (0,),
) -> int:
    """Run an external tool without a shell and retain complete logs."""

    cwd.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            input=stdin_text,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=env,
            check=False,
        )
    if completed.returncode not in allowed_returncodes:
        stderr_tail = ""
        try:
            lines = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines:
                stderr_tail = " Last stderr lines: " + " | ".join(lines[-5:])
        except OSError:
            pass
        raise PreprocessingError(
            f"External command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}. See {stdout_path} and {stderr_path}.{stderr_tail}"
        )
    return completed.returncode
