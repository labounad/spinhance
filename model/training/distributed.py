"""
model.training.distributed
==========================
Thin DDP helpers so the trainer can run single-process (CPU/1-GPU, unchanged) or
data-parallel across GPUs/nodes. Initialization reads either torchrun's env
(``RANK``/``WORLD_SIZE``/``LOCAL_RANK``) or Slurm's ``srun`` env
(``SLURM_PROCID``/``SLURM_NTASKS``/``SLURM_LOCALID`` + ``MASTER_ADDR``/``PORT``).

Everything is a no-op when ``WORLD_SIZE`` is 1, so non-distributed runs and the
test suite are completely unaffected. The model uses GroupNorm (no batch-stat
buffers), so no SyncBatchNorm is needed.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


class DistInfo:
    def __init__(self, enabled, rank, world_size, local_rank):
        self.enabled = enabled
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _slurm_master_addr() -> str:
    import subprocess
    nodelist = os.environ["SLURM_NODELIST"]
    return subprocess.check_output(
        ["scontrol", "show", "hostnames", nodelist]).decode().split()[0]


def init_distributed() -> DistInfo:
    """Init the process group from torchrun or Slurm env. Returns DistInfo
    (enabled=False / world_size=1 when not launched distributed)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:          # torchrun
        rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
    elif int(os.environ.get("SLURM_NTASKS", "1")) > 1:               # srun
        rank = int(os.environ["SLURM_PROCID"]); world = int(os.environ["SLURM_NTASKS"])
        local = int(os.environ.get("SLURM_LOCALID", 0))
        os.environ["RANK"] = str(rank); os.environ["WORLD_SIZE"] = str(world)
        os.environ["LOCAL_RANK"] = str(local)
        os.environ.setdefault("MASTER_ADDR", _slurm_master_addr())
        os.environ.setdefault("MASTER_PORT", "29500")
    else:
        return DistInfo(False, 0, 1, 0)

    if world <= 1:
        return DistInfo(False, 0, 1, 0)
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistInfo(True, rank, world, local)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_stop(stop: bool) -> bool:
    """Broadcast rank-0's early-stop decision so all ranks break together."""
    if not (dist.is_available() and dist.is_initialized()):
        return stop
    obj = [stop]
    dist.broadcast_object_list(obj, src=0)
    return bool(obj[0])


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
