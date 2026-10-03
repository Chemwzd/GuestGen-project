"""Training code for the ED-VAE (host electron density -> guest electron density).

The model is a 3D convolutional VAE over 48^3 electron-density grids. Training
uses MSE reconstruction plus a small KL term, a linear warm-up followed by
cosine annealing, and a base-id-grouped train/validation split (all augmented
copies of one host/guest complex stay inside a single split).

Data loading expects an LMDB dataset as produced by
``preprocessing.lmdb_builder`` (see that module for the record schema).
"""

from __future__ import annotations

import argparse
import datetime
import random
from pathlib import Path
from typing import Any, Optional

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from ed2mol.data.host_guest import HGLMDBDataset, read_base_id_prefix
from ed2mol.vae import VAE


def set_monitor_backend(args: argparse.Namespace):
    """Set up the training monitor.

    Returns a SwanLab run object when ``--monitor_backend swanlab`` is
    requested and ``None`` otherwise (the default ``shell`` backend only
    prints to standard output).
    """

    if args.monitor_backend != "swanlab":
        return None

    from ed2mol.monitor import init_swanlab_env, login_swanlab

    logger.info("Monitor backend: SwanLab")
    login_swanlab(api_key=args.swanlab_api_key)

    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    experiment_name = args.experiment_name or f"VAE_{time_stamp}"
    run_config = {
        "dataset": "host_guest_lmdb",
        "model": "VAE",
        "optimizer": args.optimizer,
        "lr": args.lr,
        "start_lr": args.start_lr,
        "batch_size": args.batch_size,
        "T-max": args.t_max,
        "max_channels": args.max_channels,
        "latent_dim": args.latent_dim,
        "loss_beta": args.loss_beta,
        "max_steps": args.max_steps,
        "num_warmup_steps": args.num_warmup_steps,
        "dropout_ratio": args.dropout_ratio,
        "val_interval": args.val_interval,
        "ckpt_dir": args.ckpt_dir,
        "num_cpus": args.num_cpus,
    }
    return init_swanlab_env(
        project_name=args.project_name,
        log_dir=args.log_dir,
        experiment_name=experiment_name,
        run_config=run_config,
    )


def log_metrics(
    monitor, payload: dict[str, Any], step: int
) -> None:
    if monitor is not None:
        monitor.log(payload, step=step)


def load_base_ids_from_lmdb(data_dir: Path) -> np.ndarray:
    """Read the ``base_id`` of every LMDB entry without loading the arrays.

    Only the first bytes of each pickled record are parsed (see
    :func:`read_base_id_prefix`), which keeps the split computation fast on a
    full-size dataset.
    """

    env = lmdb.open(
        str(data_dir),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    base_ids = []
    # `buffers=True` returns a memoryview, so only the parsed prefix of each
    # record is copied instead of the full 48^3 arrays.
    with env.begin(write=False, buffers=True) as txn:
        length = txn.stat()["entries"]
        for idx in range(length):
            raw = txn.get(f"{idx}".encode("ascii"))
            if raw is None:
                raise RuntimeError(f"Missing key {idx} in LMDB dataset {data_dir}")
            base_ids.append(read_base_id_prefix(raw))
    env.close()
    return np.array(base_ids)


def split_by_base_id(data_dir: Path, dataset, train_ratio: float = 0.90, seed: int = 1000):
    """Split a dataset into train/validation subsets grouped by ``base_id``."""

    base_ids = load_base_ids_from_lmdb(data_dir)
    assert len(base_ids) == len(dataset)

    unique_base_ids = np.unique(base_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_base_ids)

    n_train_groups = int(len(unique_base_ids) * train_ratio)
    train_group_set = set(unique_base_ids[:n_train_groups])

    train_indices = [i for i, bid in enumerate(base_ids) if bid in train_group_set]
    val_indices = [i for i, bid in enumerate(base_ids) if bid not in train_group_set]

    print(f"train samples: {len(train_indices)}, val samples: {len(val_indices)}")
    print(f"train groups: {n_train_groups}, val groups: {len(unique_base_ids) - n_train_groups}")

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def seed_everything(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = True,
) -> None:
    """Seed Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def reconstruction_loss(x: Tensor, gt: Tensor) -> Tensor:
    """Mean squared error between generated and ground-truth guest ED."""

    return F.mse_loss(x.squeeze(1), gt, reduction="mean")


def kl_divergence_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL divergence to a standard normal prior."""

    return -0.5 * torch.mean(1 + logvar - mu**2 - torch.exp(logvar))


def criterion(
    x: Tensor,
    gt: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
) -> tuple[Tensor, float, float]:
    recon_loss = reconstruction_loss(x, gt)
    kl_div_loss = kl_divergence_loss(mu, logvar)

    return (
        recon_loss + beta * kl_div_loss,
        recon_loss.detach().item(),
        kl_div_loss.detach().item(),
    )


def save_checkpoint(model: nn.Module, ckpt_dir: Path, step_idx: int = 0) -> None:
    """Save a model checkpoint named after the number of completed steps."""

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"guestgen_step_{step_idx + 1}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model checkpoint has been saved to {checkpoint_path}.")


def validate(
    model: nn.Module,
    val_dataloader: DataLoader,
    step_idx: int = 0,
    loss_beta: float = 1.0,
    monitor=None,
) -> None:
    """Run one validation pass and report the mean losses."""

    model.eval()

    val_loss_acc = 0.0
    recon_loss_acc = 0.0
    ed_kl_loss_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for data in val_dataloader:
            host_ed, guest_ed = data[0], data[1]
            host_ed, guest_ed = host_ed.cuda().float(), guest_ed.cuda().float()

            generated_guest_ed, mu, logvar = model(host_ed)
            loss, recon, ed_kl = criterion(
                generated_guest_ed, guest_ed, mu, logvar, beta=loss_beta
            )

            val_loss_acc += loss
            recon_loss_acc += recon
            ed_kl_loss_acc += ed_kl
            num_batches += 1

    val_loss = val_loss_acc / num_batches
    recon_loss = recon_loss_acc / num_batches
    ed_kl_loss = ed_kl_loss_acc / num_batches

    print(
        f"Validation [Step {step_idx}] | "
        f"Loss: {val_loss:.4f} | "
        f"Recon: {recon_loss:.4f}"
    )
    log_metrics(
        monitor,
        {
            "val/loss": val_loss,
            "val/recon_loss": recon_loss,
            "val/ed_kl_div": ed_kl_loss,
        },
        step_idx,
    )

    model.train()


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ckpt_dir: Path,
    val_dataloader: Optional[DataLoader] = None,
    lr: float = 1e-3,
    start_lr: float = 1e-5,
    num_steps: int = 1000,
    num_warmup_steps: int = 0,
    loss_beta: float = 1.0,
    val_interval: int = 100,
    log_interval: int = 5,
    monitor=None,
) -> None:
    data_iterator = iter(train_dataloader)

    model.train()
    for step_idx in range(1, num_steps + 1):
        log_metrics(monitor, {"train/step": step_idx}, step=step_idx)

        optimizer.zero_grad(set_to_none=False)

        # Warm-up stage (skipped when num_warmup_steps == 0).
        if step_idx <= num_warmup_steps:
            current_lr = start_lr + (lr - start_lr) * (step_idx / num_warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        try:
            data = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_dataloader)
            data = next(data_iterator)

        host_ed, guest_ed = data[0], data[1]
        host_ed, guest_ed = host_ed.cuda().float(), guest_ed.cuda().float()
        del data

        generated_guest_ed, mu, logvar = model(host_ed)
        train_loss, recon_loss, ed_kl_div_loss = criterion(
            generated_guest_ed, guest_ed, mu, logvar, beta=loss_beta
        )

        train_loss.backward()
        optimizer.step()

        if step_idx > num_warmup_steps:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

        log_metrics(monitor, {"train/lr": current_lr}, step=step_idx)

        train_loss_value = train_loss.detach().item()

        if (step_idx + 1) % log_interval == 0:
            log_metrics(
                monitor,
                {
                    "train/train_loss": train_loss_value,
                    "train/recon_loss": recon_loss,
                    "train/ed_kl_div_loss": ed_kl_div_loss,
                },
                step=step_idx,
            )
            print(
                f"Step [{min(step_idx + 1, num_steps)} / {num_steps}] | "
                f"Training loss: {train_loss_value:.4f} | "
                f"Recon loss: {recon_loss:.4f} | "
                f"ED KL-div loss: {ed_kl_div_loss:.4f}"
            )

        if (step_idx + 1) % val_interval == 0 and val_dataloader is not None:
            # step_idx counts completed optimizer steps (1..num_steps).
            # Validation and checkpointing fire every `interval` completed
            # steps, and the checkpoint name records that count
            # (guestgen_step_<completed steps>.pt).
            validate(
                model=model,
                val_dataloader=val_dataloader,
                step_idx=step_idx,
                loss_beta=loss_beta,
                monitor=monitor,
            )
            save_checkpoint(model=model, ckpt_dir=ckpt_dir, step_idx=step_idx)


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed=args.random_seed, deterministic=False, benchmark=True)

    monitor = set_monitor_backend(args)

    dataset = HGLMDBDataset(data_dir=data_dir)
    train_dataset, val_dataset = split_by_base_id(
        data_dir, dataset, train_ratio=0.90, seed=args.random_seed
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise SystemExit(
            "The dataset must provide non-empty train and validation splits; got "
            f"{len(train_dataset)} train / {len(val_dataset)} validation samples."
        )
    if len(train_dataset) < args.batch_size:
        raise SystemExit(
            f"Training split has {len(train_dataset)} samples but the batch size "
            f"is {args.batch_size} (drop_last=True would leave no batches). Use "
            "more data or a smaller --batch_size."
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_cpus,
        prefetch_factor=None,
        persistent_workers=args.num_cpus > 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_cpus,
        prefetch_factor=None,
        persistent_workers=args.num_cpus > 0,
    )

    print("INFO: Use ED-VAE to train")
    model = VAE(
        max_channels=args.max_channels,
        encoder_kernel_sizes=[5, 5, 4, 4, 1],
        encoder_strides=[2, 2, 2, 2, 1],
        encoder_paddings=[2, 2, 1, 1, 0],
        decoder_kernel_sizes=[1, 4, 4, 5, 5],
        decoder_strides=[1, 2, 2, 2, 2],
        decoder_paddings=[0, 1, 1, 2, 2],
        decoder_output_paddings=[0, 0, 0, 1, 1],
        latent_dim=args.latent_dim,
        dropout_ratio=args.dropout_ratio,
    )
    model = model.cuda().float()

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.t_max,
        eta_min=0.01 * args.lr,
    )

    train(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        ckpt_dir=ckpt_dir,
        val_dataloader=val_dataloader,
        lr=args.lr,
        start_lr=args.start_lr,
        num_steps=args.max_steps,
        num_warmup_steps=args.num_warmup_steps,
        loss_beta=args.loss_beta,
        val_interval=args.val_interval,
        log_interval=args.log_interval,
        monitor=monitor,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # data
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "examples" / "training_lmdb_subset"),
        help="LMDB dataset directory",
    )
    parser.add_argument("--num_cpus", type=int, default=4)

    # model architecture
    parser.add_argument("--max_channels", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--dropout_ratio", type=float, default=0.2)

    # training
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "runs" / "training" / "checkpoints"),
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--start_lr", type=float, default=1e-5)
    parser.add_argument("--t_max", type=int, default=1)
    parser.add_argument(
        "--optimizer", type=str, default="Adam", choices=["Adam", "AdamW"]
    )
    parser.add_argument("--loss_beta", type=float, default=1e-4)
    parser.add_argument("--val_interval", type=int, default=100)

    # logging
    parser.add_argument(
        "--monitor_backend",
        type=str,
        default="shell",
        choices=["swanlab", "shell"],
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "runs" / "training" / "logs"),
    )
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--project_name", type=str, default="VAE")
    parser.add_argument("--swanlab_api_key", type=str, default="")

    args = parser.parse_args()

    if args.val_interval <= 0 or args.log_interval <= 0:
        parser.error("--val_interval and --log_interval must be positive")
    if args.max_steps <= 0 or args.batch_size <= 0 or args.t_max <= 0:
        parser.error("--max_steps, --batch_size and --t_max must be positive")
    if args.num_cpus < 0 or args.num_warmup_steps < 0:
        parser.error("--num_cpus and --num_warmup_steps must be non-negative")

    main(args)
