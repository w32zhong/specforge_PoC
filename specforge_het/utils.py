import os
import gc
import torch


def master_print(*args, **kwargs):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        print(*args, **kwargs)


def recycle_vram():
    gc.collect()
    torch.cuda.empty_cache()


def dp_breakpoint():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        breakpoint()


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
