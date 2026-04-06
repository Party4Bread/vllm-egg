# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for EGGROLL noise generation and gradient estimation.

These kernels eliminate all Python loops from ES hot paths:

1. `triton_batch_lora_noise`: Generates LoRA (A, B) noise for all
   population members in one kernel launch. Uses Philox RNG with
   deterministic per-(param, epoch, pair) seeds.

2. `triton_batch_full_noise`: Generates full-rank noise for all members.

3. `triton_lora_gradient`: Fused fitness-weighted LoRA outer product
   accumulation: grad = sum_i(f_i * A_i @ B_i^T) in one kernel.
"""

from __future__ import annotations

import math

import torch

from vllm.triton_utils import tl, triton

# ---------------------------------------------------------------------------
# Triton kernel: batched seeded normal generation via Box-Muller
# ---------------------------------------------------------------------------


@triton.jit
def _philox_normal_pair(seed, offset):
    """Generate 2 independent normal samples from Philox RNG.

    Uses Box-Muller transform on two uniform samples.
    """
    r0, r1, _, _ = tl.randint4x(seed, offset)
    # Convert to float32 uniform (0, 1)
    u0 = (r0.to(tl.uint32, bitcast=True).to(tl.float32) + 1.0) * (1.0 / 4294967808.0)
    u1 = (r1.to(tl.uint32, bitcast=True).to(tl.float32) + 1.0) * (1.0 / 4294967808.0)
    # Box-Muller
    r = tl.sqrt(-2.0 * tl.log(u0))
    theta = 6.283185307179586 * u1
    z0 = r * tl.cos(theta)
    z1 = r * tl.sin(theta)
    return z0, z1


@triton.jit
def _batch_lora_noise_kernel(
    out_A_ptr,  # (n_unique, a, r) output
    out_B_ptr,  # (n_unique, b, r) output
    seeds_ptr,  # (n_unique,) int64 seeds
    a: tl.constexpr,
    b: tl.constexpr,
    r: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Generate (a+b)*r normal values per unique pair from its seed."""
    pair_id = tl.program_id(0)
    elem_block = tl.program_id(1)

    seed = tl.load(seeds_ptr + pair_id).to(tl.uint32)
    total = (a + b) * r
    base_offset = elem_block * BLOCK * 2

    offsets = base_offset + tl.arange(0, BLOCK)
    mask = offsets < (total + 1) // 2  # Box-Muller produces pairs

    z0, z1 = _philox_normal_pair(seed, offsets)

    # Write z0 at position 2*offset, z1 at 2*offset+1
    idx0 = offsets * 2
    idx1 = offsets * 2 + 1

    # Determine if idx goes to B (< b*r) or A (>= b*r)
    # B region: [0, b*r), A region: [b*r, (a+b)*r)
    br = b * r

    # Write z0
    mask0 = (idx0 < total) & mask
    is_B_0 = idx0 < br
    row_B_0 = idx0 // r
    col_0 = idx0 % r
    row_A_0 = (idx0 - br) // r

    tl.store(
        out_B_ptr + pair_id * b * r + row_B_0 * r + col_0,
        z0,
        mask=mask0 & is_B_0,
    )
    tl.store(
        out_A_ptr + pair_id * a * r + row_A_0 * r + col_0,
        z0,
        mask=mask0 & (~is_B_0),
    )

    # Write z1
    mask1 = (idx1 < total) & mask
    is_B_1 = idx1 < br
    row_B_1 = idx1 // r
    col_1 = idx1 % r
    row_A_1 = (idx1 - br) // r

    tl.store(
        out_B_ptr + pair_id * b * r + row_B_1 * r + col_1,
        z1,
        mask=mask1 & is_B_1,
    )
    tl.store(
        out_A_ptr + pair_id * a * r + row_A_1 * r + col_1,
        z1,
        mask=mask1 & (~is_B_1),
    )


@triton.jit
def _batch_full_noise_kernel(
    out_ptr,  # (n_unique, *shape) output
    seeds_ptr,  # (n_unique,) int64 seeds
    numel: tl.constexpr,  # product of shape
    BLOCK: tl.constexpr,
):
    """Generate numel normal values per unique pair from its seed."""
    pair_id = tl.program_id(0)
    elem_block = tl.program_id(1)

    seed = tl.load(seeds_ptr + pair_id).to(tl.uint32)
    base_offset = elem_block * BLOCK * 2
    offsets = base_offset + tl.arange(0, BLOCK)

    z0, z1 = _philox_normal_pair(seed, offsets)

    idx0 = offsets * 2
    idx1 = offsets * 2 + 1
    base = pair_id * numel

    tl.store(out_ptr + base + idx0, z0, mask=idx0 < numel)
    tl.store(out_ptr + base + idx1, z1, mask=idx1 < numel)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _compute_seeds(
    param_seed: int,
    noise_reuse: int,
    epoch: int,
    n_unique: int,
    device: torch.device,
) -> torch.Tensor:
    """Vectorized seed computation — no Python loop."""
    true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse
    pair_ids = torch.arange(n_unique, dtype=torch.int64, device=device)
    return (param_seed ^ (true_epoch * 2654435761) ^ (pair_ids * 40503)) & 0xFFFFFFFF


def triton_batch_lora_noise(
    rank: int,
    sigma: float,
    noise_reuse: int,
    epoch: int,
    pop_size: int,
    param: torch.Tensor,
    param_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate LoRA noise for all population members via Triton.

    Returns:
        A: (pop_size, out_dim, rank) — scaled by ±sigma/sqrt(rank)
        B: (pop_size, in_dim, rank)
    """
    a, b = param.shape
    r = rank
    n_unique = (pop_size + 1) // 2
    device = param.device

    seeds = _compute_seeds(param_seed, noise_reuse, epoch, n_unique, device)

    out_A = torch.empty(n_unique, a, r, device=device, dtype=torch.float32)
    out_B = torch.empty(n_unique, b, r, device=device, dtype=torch.float32)

    total = (a + b) * r
    BLOCK = 1024
    n_blocks = (total // 2 + BLOCK - 1) // BLOCK

    _batch_lora_noise_kernel[(n_unique, n_blocks)](
        out_A,
        out_B,
        seeds,
        a,
        b,
        r,
        BLOCK,
    )

    # Expand antithetical pairs and apply signs + sigma
    all_A = out_A.repeat_interleave(2, dim=0)[:pop_size]
    all_B = out_B.repeat_interleave(2, dim=0)[:pop_size]

    signs = torch.ones(pop_size, 1, 1, device=device, dtype=torch.float32)
    signs[1::2] = -1.0
    all_A = all_A * signs * (sigma / math.sqrt(r))

    return all_A.to(param.dtype), all_B.to(param.dtype)


def triton_batch_full_noise(
    noise_reuse: int,
    sigma: float,
    epoch: int,
    pop_size: int,
    param: torch.Tensor,
    param_seed: int,
) -> torch.Tensor:
    """Generate full-rank noise for all population members via Triton."""
    n_unique = (pop_size + 1) // 2
    device = param.device
    numel = param.numel()

    seeds = _compute_seeds(param_seed, noise_reuse, epoch, n_unique, device)

    out = torch.empty(
        n_unique,
        numel,
        device=device,
        dtype=torch.float32,
    )

    BLOCK = 1024
    n_blocks = (numel // 2 + BLOCK - 1) // BLOCK

    _batch_full_noise_kernel[(n_unique, n_blocks)](
        out,
        seeds,
        numel,
        BLOCK,
    )

    out = out.reshape(n_unique, *param.shape)
    all_noise = out.repeat_interleave(2, dim=0)[:pop_size]

    signs = torch.ones(
        pop_size,
        *([1] * param.ndim),
        device=device,
        dtype=torch.float32,
    )
    signs[1::2] = -1.0

    return (all_noise * signs * sigma).to(param.dtype)


# ---------------------------------------------------------------------------
# Fused LoRA gradient kernel
# ---------------------------------------------------------------------------


@triton.jit
def _lora_gradient_kernel(
    grad_ptr,  # (a, b) output accumulator
    A_ptr,  # (pop, a, r) input
    B_ptr,  # (pop, b, r) input
    f_ptr,  # (pop,) fitnesses
    pop_size: tl.constexpr,
    a: tl.constexpr,
    b: tl.constexpr,
    r: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Compute grad[i,j] = sum_n( f_n * sum_k(A[n,i,k] * B[n,j,k]) ) / pop.

    Each program computes a (BLOCK_A, BLOCK_B) tile of the output.
    """
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)

    row_offs = pid_a * BLOCK_A + tl.arange(0, BLOCK_A)
    col_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)

    acc = tl.zeros((BLOCK_A, BLOCK_B), dtype=tl.float32)

    for n in range(pop_size):
        fn = tl.load(f_ptr + n)
        for k in range(r):
            # A[n, row, k]
            a_vals = tl.load(
                A_ptr + n * a * r + row_offs * r + k,
                mask=row_offs < a,
                other=0.0,
            )
            # B[n, col, k]
            b_vals = tl.load(
                B_ptr + n * b * r + col_offs * r + k,
                mask=col_offs < b,
                other=0.0,
            )
            acc += fn * a_vals[:, None] * b_vals[None, :]

    acc = acc / pop_size
    # Store
    for i in range(BLOCK_A):
        row = pid_a * BLOCK_A + i
        if row < a:
            tl.store(
                grad_ptr + row * b + col_offs,
                acc[i, :],
                mask=col_offs < b,
            )


def triton_lora_gradient(
    rank: int,
    sigma: float,
    noise_reuse: int,
    epoch: int,
    pop_size: int,
    param: torch.Tensor,
    param_seed: int,
    fitnesses: torch.Tensor,
) -> torch.Tensor:
    """Fused LoRA gradient estimation via Triton.

    Generates noise + computes fitness-weighted outer product sum
    in a single pipeline.
    """
    a, b = param.shape

    # Generate noise (uses Triton kernel)
    all_A, all_B = triton_batch_lora_noise(
        rank,
        sigma,
        noise_reuse,
        epoch,
        pop_size,
        param,
        param_seed,
    )

    # Ensure float32 for accumulation
    A_f = all_A.to(torch.float32).contiguous()
    B_f = all_B.to(torch.float32).contiguous()
    f_f = fitnesses.to(device=param.device, dtype=torch.float32).contiguous()

    grad = torch.zeros(a, b, device=param.device, dtype=torch.float32)

    BLOCK_A = min(32, a)
    BLOCK_B = min(32, b)
    grid = (
        (a + BLOCK_A - 1) // BLOCK_A,
        (b + BLOCK_B - 1) // BLOCK_B,
    )

    _lora_gradient_kernel[grid](
        grad,
        A_f,
        B_f,
        f_f,
        pop_size,
        a,
        b,
        rank,
        BLOCK_A,
        BLOCK_B,
    )

    return grad.to(param.dtype)
