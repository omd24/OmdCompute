"""Times training steps of the denoiser and marks a capture range for Nsight Systems."""
import time

import torch
import torch.nn.functional as F

from model import Denoiser

WARMUP = 20
STEPS = 10


def step(model, optimiser, x):
    loss = F.mse_loss(model(x + 0.1 * torch.randn_like(x)), x)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()


def main():
    torch.manual_seed(0)
    x = torch.rand(32, 1, 64, 64, device="cuda")  # one batch; content does not change the cost
    model = Denoiser().cuda()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(WARMUP):  # cuDNN algorithm search and allocator warm-up
        step(model, optimiser, x)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()                  # nsys records only from here
    start = time.perf_counter()
    for i in range(STEPS):
        torch.cuda.nvtx.range_push(f"step {i}")  # named bar per step on the timeline
        step(model, optimiser, x)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / STEPS * 1e3
    torch.cuda.profiler.stop()
    print(f"{ms:.2f} ms per step")


if __name__ == "__main__":
    main()


## how to run (from experiment folder):
## "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.1\target-windows-x64\nsys.exe" profile --capture-range=cudaProfilerApi --trace=cuda,nvtx,cuDNN -o runs\profile_step .venv\Scripts\python.exe profile_step.py