# %%
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
MAIN = __name__ == "__main__"
# %%
from plotly_utils import bar, imshow, line
from part3_optimization.utils import plot_fn, plot_fn_with_points
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
import part3_optimization.tests as tests
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch import Tensor, optim
from jaxtyping import Float, Int
from IPython.display import display
from IPython.core.display import HTML
import wandb
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch as t
import numpy as np
from typing import Callable, Iterable, Literal
from dataclasses import dataclass
import time
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter0_fundamentals"
repo = "ARENA_3.0"
branch = "main"

# Install dependencies
import jaxtyping

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

assert Path(f"{root}/{chapter}/exercises").exists(), (
    "Unexpected error: please manually clone ARENA repo into `root`"
)

if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")
# %%


# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


device = t.device("mps" if t.backends.mps.is_available()
                  else "cuda" if t.cuda.is_available() else "cpu")
# %%
IN_COLAB = "google.colab" in sys.modules
assert not IN_COLAB, "Should be doing these exercises in VS Code"
# %%
WORLD_SIZE = min(t.cuda.device_count(), 3)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"


def send_receive(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Send tensor to rank 1
        sending_tensor = t.zeros(1)
        print(f"{rank=}, sending {sending_tensor=}")
        dist.send(tensor=sending_tensor, dst=1)
    elif rank == 1:
        # Receive tensor from rank 0
        received_tensor = t.ones(1)
        print(f"{rank=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, received {received_tensor=}")

    dist.destroy_process_group()


if __name__=="__main__":
    world_size = 2  # simulate 2 processes
    mp.spawn(
        send_receive,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
# %%
assert t.cuda.is_available()
assert t.cuda.device_count() > 1, "This example requires at least 2 GPUs per machine"

# %%
def send_receive_nccl(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")

    if rank == 0:
        # Create a tensor, send it to rank 1
        sending_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, sending {sending_tensor=}")
        dist.send(sending_tensor, dst=1)  # Send tensor to CPU before sending
    elif rank == 1:
        # Receive tensor from rank 0 (it needs to be on the CPU before receiving)
        received_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, {device=}, received {received_tensor=}")

    dist.destroy_process_group()


if MAIN:
    world_size = 2  # simulate 2 processes
    mp.spawn(
        send_receive_nccl,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
# %%
def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank 0 to all other ranks.
    """
    if rank==src:
        for r in range(world_size):
            if r==rank:
                continue
            dist.send(tensor, dst=r)
    else:
        dist.recv(tensor, src=src)



if MAIN:
    tests.test_broadcast(broadcast, WORLD_SIZE)
# %%
def reduce(tensor, rank, world_size, dst=0, op: Literal["sum", "mean"] = "sum"):
    """
    Reduces gradients to rank `dst`, so this process contains the sum or mean of all tensors across
    processes.
    """
    if rank != dst:
        dist.send(tensor, dst=dst)
    else:
        for other_rank in range(world_size):
            if other_rank != dst:
                received_tensor = t.zeros_like(tensor)
                dist.recv(received_tensor, src=other_rank)
                tensor += received_tensor
    if op == "mean":
        tensor /= world_size


def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """
    reduce(tensor, rank, world_size, dst=0, op=op)
    broadcast(tensor, rank, world_size, src=0)


if MAIN:
    tests.test_reduce(reduce, WORLD_SIZE)
    tests.test_all_reduce(all_reduce, WORLD_SIZE)
# %%
