"""TP 进程组初始化 + MCCL AllReduce 封装。

只支持 TP=8，固定 8 卡，不做通用抽象。
"""

import os
import pickle
import torch
import torch.distributed as dist
import torch_musa

_TP_GROUP: dist.ProcessGroup | None = None
_CPU_GROUP: dist.ProcessGroup | None = None
_TP_RANK: int = 0
_TP_SIZE: int = 8


def destroy_distributed() -> None:
    """清理进程组，避免 MCCL async error on exit。"""
    if dist.is_initialized():
        dist.destroy_process_group()


def init_distributed() -> None:
    """初始化 torch.distributed，使用 MCCL backend + Gloo CPU group。"""
    global _TP_GROUP, _CPU_GROUP, _TP_RANK, _TP_SIZE

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    assert world_size == _TP_SIZE, f"muserve requires exactly {_TP_SIZE} GPUs, got {world_size}"

    torch.musa.set_device(local_rank)

    dist.init_process_group(
        backend="mccl",
        rank=rank,
        world_size=world_size,
    )

    _TP_GROUP = dist.group.WORLD
    _TP_RANK = rank

    # Gloo CPU group for broadcasting Python objects (requests, control signals)
    _CPU_GROUP = dist.new_group(ranks=list(range(world_size)), backend="gloo")


def get_tp_rank() -> int:
    return _TP_RANK


def get_tp_size() -> int:
    return _TP_SIZE


def get_tp_group() -> dist.ProcessGroup:
    assert _TP_GROUP is not None, "call init_distributed() first"
    return _TP_GROUP


def all_reduce(tensor: torch.Tensor) -> None:
    """In-place AllReduce (sum) across all TP ranks."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=_TP_GROUP)


def all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor) -> None:
    """AllGather across TP ranks into a pre-allocated output tensor."""
    dist.all_gather_into_tensor(output, input, group=_TP_GROUP)


def barrier() -> None:
    dist.barrier(group=_TP_GROUP)


def get_cpu_group() -> dist.ProcessGroup:
    assert _CPU_GROUP is not None, "call init_distributed() first"
    return _CPU_GROUP


def broadcast_pyobj(obj, src: int = 0):
    """Broadcast a Python object from src rank to all ranks via Gloo CPU group.

    Used for distributing inference requests from rank 0 to all TP workers.
    """
    if _TP_RANK == src:
        data = pickle.dumps(obj)
        size_tensor = torch.tensor([len(data)], dtype=torch.long)
        dist.broadcast(size_tensor, src=src, group=_CPU_GROUP)
        data_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
        dist.broadcast(data_tensor, src=src, group=_CPU_GROUP)
        return obj
    else:
        size_tensor = torch.tensor([0], dtype=torch.long)
        dist.broadcast(size_tensor, src=src, group=_CPU_GROUP)
        data_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8)
        dist.broadcast(data_tensor, src=src, group=_CPU_GROUP)
        return pickle.loads(data_tensor.numpy().tobytes())
