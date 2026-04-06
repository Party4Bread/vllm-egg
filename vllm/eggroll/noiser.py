# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EGGROLL and OpenES noiser implementations.

Two backends for noise generation and gradient estimation:
1. Triton kernels (default on CUDA): all noise generated in GPU kernels
   via Philox RNG — zero Python loops, zero CPU↔GPU sync.
2. PyTorch fallback (CPU or non-CUDA): uses torch.Generator per pair.

The backend is selected automatically based on device type.

Reference: https://github.com/ESHyperscale/HyperscaleES
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

# Parameter classification (matching HyperscaleES)
PARAM = 0  # Bias/scalar → full perturbation
MM_PARAM = 1  # Weight matrix → LoRA perturbation
EMB_PARAM = 2  # Embedding → excluded
EXCLUDED = 3  # Frozen


@dataclass
class NoisedParams:
    """Mutable state for the noiser during evolution."""

    sigma: float
    optimizer: torch.optim.Optimizer
    step: int = 0


@dataclass
class FrozenNoiserParams:
    """Immutable noiser configuration."""

    group_size: int = 0
    freeze_nonlora: bool = False
    noise_reuse: int = 0
    rank: int = 1


# ---------------------------------------------------------------------------
# Seed arithmetic (no Python loops, no Generator objects)
# ---------------------------------------------------------------------------


def _pair_seeds(
    param_seed: int,
    noise_reuse: int,
    epoch: int,
    n_unique: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute deterministic seeds for each unique antithetical pair.

    Returns int64 tensor of shape (n_unique,) on `device`.
    All arithmetic is vectorized — no Python loop.
    """
    true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse
    pair_ids = torch.arange(n_unique, dtype=torch.int64, device=device)
    return (param_seed ^ (true_epoch * 2654435761) ^ (pair_ids * 40503)) & 0xFFFFFFFF


def _seeded_randn(
    seeds: torch.Tensor,
    shape_per_seed: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate random normal tensors from a batch of integer seeds.

    This uses a simple but deterministic approach: generate one large
    randn block from a master seed derived from the seed tensor, then
    use per-seed manual_seed for exact reproducibility with the original
    per-pair Generator approach.

    Returns: (n_seeds, *shape_per_seed)
    """
    n = seeds.shape[0]
    out = torch.empty(n, *shape_per_seed, dtype=dtype, device=device)
    # CPU generator loop is unavoidable for exact reproducibility with
    # torch.Generator, but we minimize overhead: no Generator allocation
    # per call, reuse one generator.
    gen = torch.Generator(device=device) if device.type == "cpu" else torch.Generator()
    for i in range(n):
        gen.manual_seed(seeds[i].item())
        out[i] = torch.randn(*shape_per_seed, generator=gen, dtype=dtype, device=device)
    return out


# ---------------------------------------------------------------------------
# Batched noise generation
# ---------------------------------------------------------------------------


def batch_lora_noise(
    frozen: FrozenNoiserParams,
    sigma: float,
    epoch: int,
    pop_size: int,
    param: torch.Tensor,
    param_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate LoRA noise for ALL population members.

    Uses Triton kernels on CUDA (zero Python loops), falls back to
    PyTorch Generator on CPU.

    Returns:
        A: (pop_size, out_dim, rank) — scaled by ±sigma/sqrt(rank)
        B: (pop_size, in_dim, rank) — raw directions
    """
    if param.is_cuda:
        from vllm.eggroll.ops.noise_gen import triton_batch_lora_noise

        return triton_batch_lora_noise(
            frozen.rank,
            sigma,
            frozen.noise_reuse,
            epoch,
            pop_size,
            param,
            param_seed,
        )

    # PyTorch fallback for CPU
    a, b = param.shape
    r = frozen.rank
    n_unique = (pop_size + 1) // 2

    seeds = _pair_seeds(param_seed, frozen.noise_reuse, epoch, n_unique, param.device)
    raw = _seeded_randn(seeds, (a + b, r), param.device, param.dtype)
    unique_B = raw[:, :b, :]
    unique_A = raw[:, b:, :]

    all_A = unique_A.repeat_interleave(2, dim=0)[:pop_size]
    all_B = unique_B.repeat_interleave(2, dim=0)[:pop_size]

    signs = torch.ones(pop_size, 1, 1, device=param.device, dtype=param.dtype)
    signs[1::2] = -1.0

    all_A = all_A * signs * (sigma / math.sqrt(r))
    return all_A, all_B


def batch_full_noise(
    frozen: FrozenNoiserParams,
    sigma: float,
    epoch: int,
    pop_size: int,
    param: torch.Tensor,
    param_seed: int,
) -> torch.Tensor:
    """Generate full-rank noise for ALL population members."""
    if param.is_cuda:
        from vllm.eggroll.ops.noise_gen import triton_batch_full_noise

        return triton_batch_full_noise(
            frozen.noise_reuse,
            sigma,
            epoch,
            pop_size,
            param,
            param_seed,
        )

    # PyTorch fallback
    n_unique = (pop_size + 1) // 2
    seeds = _pair_seeds(param_seed, frozen.noise_reuse, epoch, n_unique, param.device)
    raw = _seeded_randn(seeds, param.shape, param.device, param.dtype)

    all_noise = raw.repeat_interleave(2, dim=0)[:pop_size]
    signs = torch.ones(
        pop_size,
        *([1] * param.ndim),
        device=param.device,
        dtype=param.dtype,
    )
    signs[1::2] = -1.0
    return all_noise * signs * sigma


# ---------------------------------------------------------------------------
# Vectorized gradient estimation
# ---------------------------------------------------------------------------


def compute_lora_gradient_batched(
    frozen: FrozenNoiserParams,
    sigma: float,
    epoch: int,
    pop_size: int,
    param: torch.Tensor,
    param_seed: int,
    fitnesses: torch.Tensor,
) -> torch.Tensor:
    """grad = einsum('nir,njr->ij', fitness*A, B) / pop_size

    On CUDA, uses fused Triton kernel that does noise gen + gradient
    accumulation without materializing intermediate tensors.
    """
    if param.is_cuda:
        from vllm.eggroll.ops.noise_gen import triton_lora_gradient

        return triton_lora_gradient(
            frozen.rank,
            sigma,
            frozen.noise_reuse,
            epoch,
            pop_size,
            param,
            param_seed,
            fitnesses,
        )

    # PyTorch fallback
    all_A, all_B = batch_lora_noise(frozen, sigma, epoch, pop_size, param, param_seed)
    f = fitnesses.to(device=param.device, dtype=param.dtype).reshape(pop_size, 1, 1)
    return torch.einsum("nir,njr->ij", f * all_A, all_B) / pop_size


def compute_full_gradient_batched(
    frozen: FrozenNoiserParams,
    sigma: float,
    epoch: int,
    pop_size: int,
    param: torch.Tensor,
    param_seed: int,
    fitnesses: torch.Tensor,
) -> torch.Tensor:
    if frozen.freeze_nonlora:
        return torch.zeros_like(param)
    all_noise = batch_full_noise(frozen, sigma, epoch, pop_size, param, param_seed)
    f = fitnesses.to(device=param.device, dtype=param.dtype)
    f = f.reshape(pop_size, *([1] * param.ndim))
    return (f * all_noise).mean(dim=0)


# ---------------------------------------------------------------------------
# Fitness normalization
# ---------------------------------------------------------------------------


def normalize_fitnesses(raw_scores: torch.Tensor, group_size: int = 0) -> torch.Tensor:
    """Z-score normalize fitness. Group-aware if group_size > 0."""
    global_std = torch.sqrt(torch.var(raw_scores) + 1e-5)
    if group_size == 0:
        return (raw_scores - raw_scores.mean()) / global_std
    grouped = raw_scores.reshape(-1, group_size)
    group_means = grouped.mean(dim=-1, keepdim=True)
    return ((grouped - group_means) / global_std).reshape(-1)


# ---------------------------------------------------------------------------
# Param classification cache
# ---------------------------------------------------------------------------


def build_param_plan(
    model: nn.Module,
    classify_fn,
    base_seed: int = 42,
) -> list[tuple[str, int, int]]:
    """Pre-compute (name, classification, seed) for every parameter.

    Called once at init. Eliminates repeated classify_param() + dict
    lookups in the hot path.
    """
    plan = []
    for i, (name, param) in enumerate(model.named_parameters()):
        cls_id = classify_fn(name, param)
        seed = base_seed + i * 7919
        plan.append((name, cls_id, seed))
    return plan


# ---------------------------------------------------------------------------
# Noiser classes
# ---------------------------------------------------------------------------


class EggRoll:
    """EGGROLL evolutionary strategy noiser.

    Hot paths (compute_gradients, export_lora_adapters) iterate the
    pre-computed param_plan instead of calling named_parameters() +
    classify_param() each time.
    """

    @classmethod
    def init(
        cls,
        model: nn.Module,
        sigma: float,
        lr: float,
        optimizer_cls: type = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        group_size: int = 0,
        freeze_nonlora: bool = True,
        noise_reuse: int = 0,
        rank: int = 1,
        **kwargs,
    ) -> tuple[FrozenNoiserParams, NoisedParams]:
        if optimizer_cls is None:
            optimizer_cls = torch.optim.Adam
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)
        frozen = FrozenNoiserParams(
            group_size=group_size,
            freeze_nonlora=freeze_nonlora,
            noise_reuse=noise_reuse,
            rank=rank,
        )
        return frozen, NoisedParams(sigma=sigma, optimizer=optimizer)

    @classmethod
    def classify_param(cls, name: str, param: torch.Tensor) -> int:
        if "embed" in name.lower() or "lm_head" in name.lower():
            return EXCLUDED
        if param.ndim >= 2 and "weight" in name.lower():
            return MM_PARAM
        if param.ndim <= 1:
            return PARAM
        return MM_PARAM

    @classmethod
    def get_param_plan(cls, model: nn.Module, base_seed: int = 42):
        return build_param_plan(model, cls.classify_param, base_seed)

    @classmethod
    def get_param_seeds(cls, model: nn.Module, base_seed: int = 42) -> dict:
        return {
            name: base_seed + i * 7919
            for i, (name, _) in enumerate(model.named_parameters())
        }

    @classmethod
    def perturb_model(
        cls,
        frozen: FrozenNoiserParams,
        sigma: float,
        model: nn.Module,
        epoch: int,
        thread_id: int,
        param_seeds: dict[str, int],
        original_params: dict[str, torch.Tensor],
    ) -> None:
        """Apply perturbation for a single population member in-place."""
        true_epoch = 0 if frozen.noise_reuse == 0 else epoch // frozen.noise_reuse
        true_thread = thread_id // 2
        sign = 1.0 if thread_id % 2 == 0 else -1.0
        r = frozen.rank
        eff_sigma = sign * sigma / math.sqrt(r)

        gen = torch.Generator()
        with torch.no_grad():
            for name, param in model.named_parameters():
                classification = cls.classify_param(name, param)
                if classification == EXCLUDED:
                    continue
                if classification == PARAM and frozen.freeze_nonlora:
                    continue

                seed = param_seeds[name]
                original = original_params[name]
                combined = seed ^ (true_epoch * 2654435761) ^ (true_thread * 40503)
                gen.manual_seed(combined & 0xFFFFFFFF)

                if classification == MM_PARAM:
                    a, b = original.shape
                    lora = torch.randn(
                        a + b,
                        r,
                        generator=gen,
                        dtype=original.dtype,
                        device=original.device,
                    )
                    param.copy_(original + (lora[b:] * eff_sigma) @ lora[:b].T)
                else:
                    noise = torch.randn(
                        *original.shape,
                        generator=gen,
                        dtype=original.dtype,
                        device=original.device,
                    )
                    param.copy_(original + noise * sign * sigma)

    @classmethod
    def compute_gradients(
        cls,
        frozen: FrozenNoiserParams,
        sigma: float,
        epoch: int,
        pop_size: int,
        model: nn.Module,
        param_seeds: dict[str, int],
        fitnesses: torch.Tensor,
        param_plan: list[tuple[str, int, int]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Vectorized gradient estimation for all parameters."""
        gradients = {}
        scale = math.sqrt(pop_size)
        params_dict = dict(model.named_parameters())

        items = (
            param_plan
            if param_plan
            else [
                (n, cls.classify_param(n, p), param_seeds[n])
                for n, p in model.named_parameters()
            ]
        )

        for name, classification, seed in items:
            param = params_dict[name]
            if classification == EXCLUDED:
                gradients[name] = torch.zeros_like(param)
            elif classification == MM_PARAM:
                grad = compute_lora_gradient_batched(
                    frozen, sigma, epoch, pop_size, param, seed, fitnesses
                )
                gradients[name] = -(grad * scale).to(param.dtype)
            elif classification == PARAM:
                grad = compute_full_gradient_batched(
                    frozen, sigma, epoch, pop_size, param, seed, fitnesses
                )
                gradients[name] = -(grad * scale).to(param.dtype)
        return gradients

    @classmethod
    def update_params(
        cls,
        noised: NoisedParams,
        model: nn.Module,
        gradients: dict[str, torch.Tensor],
    ) -> None:
        noised.optimizer.zero_grad()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in gradients:
                    param.grad = gradients[name].to(param.dtype)
        noised.optimizer.step()
        noised.step += 1

    @classmethod
    def convert_fitnesses(
        cls, frozen: FrozenNoiserParams, raw_scores: torch.Tensor
    ) -> torch.Tensor:
        return normalize_fitnesses(raw_scores, frozen.group_size)

    @classmethod
    def export_lora_adapters(
        cls,
        frozen: FrozenNoiserParams,
        sigma: float,
        epoch: int,
        pop_size: int,
        model: nn.Module,
        param_seeds: dict[str, int],
        output_dir: str,
        target_modules: list[str] | None = None,
        param_plan: list[tuple[str, int, int]] | None = None,
    ) -> list[str]:
        """Export population perturbations as PEFT LoRA adapters.

        Generates noise for all members in batch, then writes files.
        """
        import json
        import os

        from safetensors.torch import save_file

        r = frozen.rank
        params_dict = dict(model.named_parameters())

        items = (
            param_plan
            if param_plan
            else [
                (n, cls.classify_param(n, p), param_seeds[n])
                for n, p in model.named_parameters()
            ]
        )

        # Collect only MM_PARAM items
        mm_items = [(n, s) for n, c, s in items if c == MM_PARAM]

        if target_modules is None:
            target_modules = sorted(
                {
                    n.rsplit(".", 1)[0].rsplit(".", 1)[-1]
                    if n.endswith(".weight")
                    else n.rsplit(".", 1)[-1]
                    for n, _ in mm_items
                }
            )

        # Pre-generate all noise in batch per parameter
        # all_noise[param_name] = (all_A, all_B) each (pop_size, ...)
        all_noise: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, seed in mm_items:
            param = params_dict[name]
            A, B = batch_lora_noise(frozen, sigma, epoch, pop_size, param, seed)
            # Move to CPU for save_file
            all_noise[name] = (A.cpu(), B.cpu())

        adapter_dirs = []
        for member_id in range(pop_size):
            adapter_dir = os.path.join(output_dir, f"member_{member_id}")
            os.makedirs(adapter_dir, exist_ok=True)

            tensors = {}
            for name, _ in mm_items:
                A_all, B_all = all_noise[name]
                # A_all[member_id] is (out_dim, rank) already scaled
                # B_all[member_id] is (in_dim, rank) raw direction
                module_name = name
                if module_name.endswith(".weight"):
                    module_name = module_name[: -len(".weight")]
                peft_prefix = f"base_model.model.{module_name}"
                # lora_A: (rank, in_dim), lora_B: (out_dim, rank)
                tensors[f"{peft_prefix}.lora_A.weight"] = B_all[
                    member_id
                ].T.contiguous()
                tensors[f"{peft_prefix}.lora_B.weight"] = A_all[member_id].contiguous()

            save_file(
                tensors,
                os.path.join(adapter_dir, "adapter_model.safetensors"),
            )

            config = {
                "r": r,
                "lora_alpha": r,
                "target_modules": target_modules,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "peft_type": "LORA",
            }
            with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
                json.dump(config, f)

            adapter_dirs.append(adapter_dir)

        return adapter_dirs


class OpenES:
    """OpenAI-style ES. Full-rank perturbations for all parameters."""

    @classmethod
    def init(
        cls,
        model,
        sigma,
        lr,
        optimizer_cls=None,
        optimizer_kwargs=None,
        group_size=0,
        freeze_nonlora=False,
        noise_reuse=0,
        **kw,
    ):
        if optimizer_cls is None:
            optimizer_cls = torch.optim.SGD
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)
        frozen = FrozenNoiserParams(
            group_size=group_size,
            freeze_nonlora=freeze_nonlora,
            noise_reuse=noise_reuse,
            rank=1,
        )
        return frozen, NoisedParams(sigma=sigma, optimizer=optimizer)

    @classmethod
    def classify_param(cls, name: str, param: torch.Tensor) -> int:
        if "embed" in name.lower() or "lm_head" in name.lower():
            return EXCLUDED
        return PARAM

    @classmethod
    def get_param_plan(cls, model, base_seed=42):
        return build_param_plan(model, cls.classify_param, base_seed)

    @classmethod
    def get_param_seeds(cls, model, base_seed=42):
        return EggRoll.get_param_seeds(model, base_seed)

    perturb_model = EggRoll.perturb_model

    @classmethod
    def compute_gradients(
        cls,
        frozen,
        sigma,
        epoch,
        pop_size,
        model,
        param_seeds,
        fitnesses,
        param_plan=None,
    ):
        gradients = {}
        scale = math.sqrt(pop_size)
        params_dict = dict(model.named_parameters())
        items = (
            param_plan
            if param_plan
            else [
                (n, cls.classify_param(n, p), param_seeds[n])
                for n, p in model.named_parameters()
            ]
        )
        for name, classification, seed in items:
            param = params_dict[name]
            if classification == EXCLUDED:
                gradients[name] = torch.zeros_like(param)
            else:
                grad = compute_full_gradient_batched(
                    frozen, sigma, epoch, pop_size, param, seed, fitnesses
                )
                gradients[name] = -(grad * scale).to(param.dtype)
        return gradients

    update_params = EggRoll.update_params

    @classmethod
    def convert_fitnesses(cls, frozen, raw_scores):
        return normalize_fitnesses(raw_scores, frozen.group_size)
