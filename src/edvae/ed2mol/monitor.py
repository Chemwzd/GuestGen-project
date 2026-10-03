"""Optional SwanLab logging backend for ``train.py``.

SwanLab is imported lazily so that the default ``--monitor_backend shell``
training run does not require a SwanLab installation or network access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union


def login_swanlab(api_key: str) -> None:
    import swanlab

    swanlab.login(api_key=api_key)


def init_swanlab_env(
    project_name: str,
    log_dir: Union[str, Path] = "./logs/swanlog",
    experiment_name: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None,
):
    """Initialize a SwanLab run and return the run object."""

    import swanlab

    if not isinstance(log_dir, Path):
        log_dir = Path(log_dir)

    return swanlab.init(
        project=project_name,
        experiment_name=experiment_name,
        config=run_config,
        logdir=log_dir,
        mode="cloud",
        resume="allow",
    )
